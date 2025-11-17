import os, csv, json, re
import torch
from PIL import Image
from Losses.loss import CoxPHLoss


def is_global_zero(trainer):
    try:
        return bool(getattr(trainer, "is_global_zero", True))
    except Exception:
        return True

def get_label(labels, idx):
    if labels is None:
        return None
    try:
        return labels[int(idx)]
    except Exception:
        try:
            if torch.is_tensor(labels):
                val = labels[int(idx)]
                return val.detach().cpu().item() if val.ndim == 0 else str(val.detach().cpu().tolist())
            return str(labels[int(idx)])
        except Exception:
            return None

def get_lr(trainer):
    try:
        cfgs = getattr(trainer, "lr_scheduler_configs", None)
        if cfgs:
            return float(cfgs[0].scheduler.optimizer.param_groups[0]["lr"])
    except Exception:
        pass
    try:
        return float(trainer.optimizers[0].param_groups[0]["lr"])
    except Exception:
        return None

def slug(s, maxlen=50):
    if s is None:
        return "none"
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s[:maxlen] if maxlen else s

def to_uint8_slice(t2d: torch.Tensor) -> Image.Image:
    """Robust 2D -> uint8 grayscale with percentile clipping."""
    t = t2d.detach().float().cpu()
    numel = t.numel()
    try:
        lo = torch.quantile(t, 0.01) if numel >= 100 else t.min()
        hi = torch.quantile(t, 0.99) if numel >= 100 else t.max()
    except Exception:
        lo, hi = t.min(), t.max()
    denom = (hi - lo).clamp(min=1e-6)
    t = (t - lo) / denom
    t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    arr = t.numpy()
    return Image.fromarray(arr, mode="L")

def save_mid_slices_for_batch(
    vol: torch.Tensor,            # expected shape [B, 2, D1, D2, D3]
    out_root: str,                # directory to save into
    tag: str,                     # e.g., f"ep{epoch}_gs{global_step}_b{batch_idx}"
    slabels,                      # list/tensor of sample labels for nicer filenames (optional)
):
    """
    Saves mid-Z, mid-Y, mid-X slices for CT (ch=0) and Dose (ch=1) for every sample.
    Returns two lists (ct_list, dose_list) of dicts with file paths per sample.
    """
    assert vol.ndim == 5 and vol.size(1) >= 2, "Expected [B, 2, D1, D2, D3]"
    B, C, D1, D2, D3 = vol.shape
    os.makedirs(out_root, exist_ok=True)

    # Mid indices (integer floors)
    m1, m2, m3 = D1 // 2, D2 // 2, D3 // 2

    def _one_channel_paths(ch_name: str, ch_idx: int):
        out_list = []
        for i in range(B):
            lbl_raw = get_label(slabels, i)
            lbl = slug(lbl_raw)
            base = f"{tag}_{lbl}_{ch_name}"  # f"{tag}_s{i}_{lbl}_{ch_name}"

            # Planes (names chosen to reflect axis index in the tensor)
            # mid-Z equivalent -> along dim 2 (D1)
            pz = vol[i, ch_idx, m1, :, :]
            # mid-Y equivalent -> along dim 3 (D2)
            py = vol[i, ch_idx, :, m2, :]
            # mid-X equivalent -> along dim 4 (D3)
            px = vol[i, ch_idx, :, :, m3]

            # Save JPEGs
            paths = {}
            for plane_name, plane in (("midZ", pz), ("midY", py), ("midX", px)):
                img = to_uint8_slice(plane)
                fpath = os.path.join(out_root, f"{base}_{plane_name}.jpg")
                img.save(fpath, format="JPEG", quality=85, subsampling=0)
                paths[plane_name] = fpath

            out_list.append({
                "sample_index": i,
                "sample_label": lbl_raw if lbl_raw is not None else None,
                **paths
            })
        return out_list

    ct_list   = _one_channel_paths("CT",   0)
    dose_list = _one_channel_paths("DOSE", 1)
    if C == 3:
        struct_list = _one_channel_paths("STRUCT", 2)
    else:
        struct_list = None
    return ct_list, dose_list, struct_list


def test_loss_computation(prediction, label, event, lf):
    if lf[0] == 'CoxPHLoss':
        lf = CoxPHLoss(mode='implemented', reduction='none')
        loss = lf.forward(prediction.detach(), label.detach(), event.detach())
        return loss
    else:
        lf = getattr(torch.nn, lf[0])(reduction='none')
        loss = lf(prediction.detach(), label.detach())
        return loss
