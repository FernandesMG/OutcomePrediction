import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic depth per-sample (residual branch only)."""
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)
        assert 0.0 <= self.p < 1.0

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        # per-sample mask, broadcast over C,D,H,W
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, groups=8, se=False, p_drop=0.1, post_act=True):
        super().__init__()

        g_out = math.gcd(groups, out_ch)  # robust GroupNorm groups
        assert g_out >= 1

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(g_out, out_ch)
        self.act   = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(g_out, out_ch)

        self.se = None
        if se:
            mid = max(8, out_ch // 8)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(out_ch, mid, 1), nn.SiLU(inplace=True),
                nn.Conv3d(mid, out_ch, 1), nn.Sigmoid()
            )

        # DropPath on residual branch (better than Dropout3d on the sum)
        self.drop_path = DropPath(p_drop) if p_drop > 0 else nn.Identity()
        self.post_act = post_act  # keep your original style by default

        # Downsample if needed
        if (in_ch != out_ch) or (stride != 1):
            self.down = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(g_out, out_ch)
            )
        else:
            self.down = nn.Identity()

        # init: last norm gamma to zero -> start near identity
        nn.init.zeros_(self.gn2.weight)

    def forward(self, x):
        identity = self.down(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.gn2(out)

        if self.se is not None:
            # SE expects channel weights; multiply after norm
            out = out * self.se(out)

        out = identity + self.drop_path(out)

        if self.post_act:
            out = self.act(out)

        return out


# ---------- Helper: upsample-and-fuse block ----------
class UpFuse(nn.Module):
    """
    Upsamples 'x' to match spatial size of 'skip', aligns channels, concatenates, then fuses with a ResBlock.
    """
    def __init__(self, in_ch, skip_ch, out_ch, groups=8, drop_path=0.0):
        super().__init__()
        # 1x1 to align channels after upsampling (optional but neat)
        self.proj = nn.Conv3d(in_ch, skip_ch, kernel_size=1, bias=False)
        self.fuse = ResBlock3D(in_ch=skip_ch + skip_ch, out_ch=out_ch, stride=1,
                               groups=groups, se=False, drop_path=drop_path)

    def forward(self, x, skip):
        # upsample x to skip spatial size
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)
