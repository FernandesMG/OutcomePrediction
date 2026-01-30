##Losses
import torch
from torch.nn.modules import Module
import numpy as np
from torchmetrics import Metric
from pycox.models import loss as losses
import torch.nn.functional as F
from monai.losses import SSIMLoss


def CrossEntropy(output, target):
    log_prob = F.log_softmax(output, dim=1)
    loss = F.nll_loss(log_prob, torch.argmax(target, dim=1), reduction='none')
    return torch.mean(loss)


def MaskedMSELoss(y_pred, y_true):
    """MSE loss that ignores missing values (NaNs) in the target."""
    mask = ~torch.isnan(y_true)  # Mask where targets are not NaN
    loss = (y_pred[mask] - y_true[mask]) ** 2
    return loss.mean() if mask.any() else torch.tensor(0.0, device=y_pred.device)


def SoftDiceLoss(output, target):
    """
   Reference: Milletari, F., Navab, N., & Ahmadi, S. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric
   Medical Image Segmentation. In International Conference on 3D Vision (3DV).
   """
    output = F.logsigmoid(output).exp()
    axes = list(range(2, len(output.shape)))
    eps = 1e-10
    intersection = torch.sum(output * target + eps, axes)
    output_sum_square = torch.sum(output * output + eps, axes)
    target_sum_square = torch.sum(target * target + eps, axes)
    sum_squares = output_sum_square + target_sum_square
    return 1.0 - 2.0 * torch.mean(intersection / sum_squares)


def WeightedMSE(prediction, labels, weights=None, label_range=None):
    loss = 0
    for i, label in enumerate(labels):
        idx = (label_range == int(label.cpu().numpy())).nonzero()
        if (idx is not None) and (idx[0][0] < 60):
            loss = loss + (prediction[i] - label) ** 2 * weights[idx[0][0]]
        else:
            loss = loss + (prediction[i] - label) ** 2 * weights[-1]
    loss = loss / (i + 1)
    return loss


class CoxPHLoss(Module):
    def __init__(self, mode='pycox', reduction='mean'):
        super().__init__()
        assert reduction in ['mean', 'sum', 'none']
        self.mode = mode
        self.reduction = reduction

    def forward(self, risk_scores, durations, events):
        if self.mode == 'implemented':
            return self.manual_implementation(risk_scores, durations, events)
        elif self.mode == 'pycox':
            return self.pycox_implementation(risk_scores, durations, events)

    @staticmethod
    def manual_implementation(risk_scores, durations, events, reduction='mean'):
        # Sort by descending durations
        sorted_indices = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[sorted_indices]
        events = events[sorted_indices]

        # Compute log partial likelihood
        log_cumulative_hazard = torch.logcumsumexp(risk_scores, dim=0)  # check
        log_likelihood = risk_scores - log_cumulative_hazard
        uncensored_likelihood = log_likelihood * events  # events censored should be zero here
        if reduction == 'mean':
            loss = -torch.mean(uncensored_likelihood)
        elif reduction == 'sum':
            loss = -torch.sum(uncensored_likelihood)
        else:
            loss = -uncensored_likelihood
        return loss

    @staticmethod
    def pycox_implementation(risk_scores, durations, events, reduction='mean'):
        sorted_indices = torch.argsort(durations, descending=True)
        pycox_loss = losses.cox_ph_loss_sorted(risk_scores[sorted_indices], events[sorted_indices])
        if reduction == 'mean':
            loss = pycox_loss
        elif reduction == 'sum':
            loss = pycox_loss * events.sum()
        else:
            raise AttributeError('reduction=="none" is not implemented for pycox_implementation')
        return loss


class ConcordanceIndex(Metric):
    def __init__(self, pred_type='risk'):
        super().__init__()
        assert pred_type in ['risk', 'time']
        self.pred_type='risk'
        self.factor = -1 if pred_type == 'risk' else 1
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")  # 1 = event occurred, 0 = censored

    def update(self, preds: torch.Tensor, times: torch.Tensor, events: torch.Tensor):
        self.preds.append(preds.detach().flatten())
        self.times.append(times.detach().flatten())
        self.events.append(events.detach().flatten())

    def compute(self):
        preds = torch.cat(self.preds) * self.factor
        times = torch.cat(self.times)
        events = torch.cat(self.events)

        # Get all pairwise comparisons (i, j)
        t_i = times.unsqueeze(0)
        t_j = times.unsqueeze(1)
        e_i = events.unsqueeze(0)
        p_i = preds.unsqueeze(0)
        p_j = preds.unsqueeze(1)

        # Identify valid comparable pairs: t_i < t_j and e_i == 1
        valid = (t_i < t_j) & (e_i == 1)

        # For those, check prediction concordance
        pred_diff = p_i - p_j
        time_diff = t_i - t_j

        concordant = ((pred_diff * time_diff) > 0).float()
        ties = (pred_diff == 0).float() * 0.5

        cindex_matrix = concordant + ties
        total_valid = valid.float().sum()

        if total_valid == 0:
            return torch.tensor(0.0, device=preds.device)

        return (cindex_matrix * valid.float()).sum() / total_valid


def grad3d(t):
    dz = t[..., 1:, :, :] - t[..., :-1, :, :]
    dy = t[..., :, 1:, :] - t[..., :, :-1, :]
    dx = t[..., :, :, 1:] - t[..., :, :, :-1]
    return dz, dy, dx


class GradientDiffLoss(torch.nn.Module):
    def __init__(self, p=1, reduction='mean'):  # p=1 -> L1 on gradients
        super().__init__()
        self.p = p
        self.reduction = reduction
        assert reduction in ['mean', 'none']

    def forward(self, x, y, mask=None):
        gz = grad3d(x)
        gy = grad3d(y)
        diffs = [a - b for a, b in zip(gz, gy)]
        if mask is not None:
            m_z = mask[..., 1:, :, :]; m_y = mask[..., :, 1:, :]; m_x = mask[..., :, :, 1:]
            diffs = [d * m for d, m in zip(diffs, (m_z, m_y, m_x))]
        if self.p == 1:
            vals = (sum(d.abs().mean() for d in diffs)
                    if self.reduction == 'mean' else sum(d.abs().mean(axis=[-3, -2, -1]) for d in diffs))
        else:
            vals = (sum((d**2).mean() for d in diffs)
                    if self.reduction == 'mean' else sum((d**2).mean(axis=[-3, -2, -1]) for d in diffs))
        return vals / 3.0


class ComboLoss(torch.nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.1, w_gdl=0.1, reduction='mean'):
        super().__init__()
        self.l1 = torch.nn.L1Loss(reduction=reduction)
        self.ssim = SSIMLoss(spatial_dims=3, win_size=7, reduction=reduction)
        self.gdl = GradientDiffLoss(p=1, reduction=reduction)
        self.w_l1, self.w_ssim, self.w_gdl = w_l1, w_ssim, w_gdl
        self.reduction = reduction

    def forward(self, pred, tgt, mask=None):
        loss = self.w_l1 * (self.l1(pred, tgt) * (mask if mask is not None else 1.0))
        loss += self.w_ssim * (self.ssim(pred, tgt) * (mask if mask is not None else 1.0))[:, :, None, None, None]
        loss += self.w_gdl * self.gdl(pred, tgt, mask)[:, :, None, None, None]
        return loss

