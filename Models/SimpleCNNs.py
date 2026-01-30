import math
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from .ResBlocks import ResBlock3D, UpFuse


# Define CNN model with optional residual connections and dropout
class CNN(pl.LightningModule):
    def __init__(self, config, use_residual=False, use_dropout=False):
        super().__init__()
        self.save_hyperparameters()

        self.use_residual = use_residual
        self.use_dropout = use_dropout

        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

        # Define the CNN as a Sequential block
        self.cnn = nn.Sequential(
            nn.Conv3d(config['DATA']['n_channel'], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Define the feed-forward block as a Sequential block
        self.feed_forward = nn.Sequential(
            nn.Linear(256 * (config['DATA']['dim'][0] // 64 + 1) *
                      (config['DATA']['dim'][1] // 64 + 1) *
                      (config['DATA']['dim'][2] // 32), config['MODEL']['classifier_in']),
            nn.ReLU(),
            self.dropout
        )

    def forward(self, x):
        identity = x
        x = self.cnn(x)

        if self.use_residual:
            x += identity  # Add residual connection if enabled (only works if dimensions match)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.feed_forward(x)
        return x

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat = self.forward(x)
        # loss = F.cross_entropy(y_hat, y)
        # self.log('train_loss', loss)
        # return loss
        return

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# ---------- ResUNet3D using residual backbone style ----------
class ResUNet3D(nn.Module):
    """
    A small, fast ResUNet-like 3D classifier/regressor.
    Depth: 3 downs / 3 ups. Channels: 8-16-32 (match your backbone).
    Returns logits of shape [B, out_classes, D, H, W].
    """
    def __init__(self, in_channels, out_classes, base=(8,16,32), groups=8, drop_path_max=0.1, anisotropic=False):
        super().__init__()
        c1, c2, c3 = base

        # full-res stem skip (cheap but helps detail)
        g1 = math.gcd(groups, c1) or 1
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, c1),
            nn.SiLU(inplace=False),
        )

        # Encoder (down)
        # Stochastic depth schedule across encoder+decoder blocks (light)
        # Order: enc1, enc2, enc3, bottleneck, dec2, dec1, dec0  (7 blocks)
        self.encoder = ResUNet3DEncoder(in_channels, base, groups, drop_path_max, anisotropic)

        # Decoder (up + fuse)
        self.decoder = ResUNet3DDecoder(out_classes, base, groups, drop_path_max)

    def forward(self, x):
        # full-res skip
        s0 = self.stem(x)

        b, e1, e2, e3 = self.encoder(x)

        logits = self.decoder(b, e2, e1, s0)  # [B, C, D, H, W]

        return logits


class ResUNet3DWTabInput(ResUNet3D):
    """
    A small, fast ResUNet-like 3D classifier/regressor which modifies one of its input channels using external tabular
    variables. Modification is made by creating new uniform channels with the specified tabular variables.
    Depth: 3 downs / 3 ups. Channels: 8-16-32 (match your backbone).
    Returns logits of shape [B, out_classes, D, H, W].
    """
    def __init__(self, in_channels, out_classes, base=(8,16,32), groups=8, drop_path_max=0.1, anisotropic=False,
                 tab_vars=2, modified_channel_idx=0):
        super().__init__(in_channels, out_classes, base, groups, drop_path_max, anisotropic)
        self.modified_channel_idx = modified_channel_idx
        c1, c2, c3 = base

        g1 = math.gcd(groups, c1) or 1
        self.channel_mod = nn.Sequential(
            nn.Conv3d(tab_vars+1, c1, kernel_size=1, padding=1, bias=False),
            nn.GroupNorm(g1, c1),
            nn.SiLU(inplace=False),
            nn.Conv3d(c1, 1, kernel_size=1, padding=1, bias=True),
        )

    def forward(self, x, tab_vars):
        # expand tab vars to image shape
        expanded_tab = tab_vars[:, :, None, None, None] * torch.ones([tab_vars[0], tab_vars[1]] + list(x.shape[2:]))

        # generate new modified channel
        modified_channel = self.channel_mod(torch.cat([expanded_tab, x[:, self.modified_channel_idx]], dim=1))

        # substitute original channel by modified channel
        x[:, self.modified_channel_idx] = modified_channel

        # full-res skip
        s0 = self.stem(x)

        b, e1, e2, e3 = self.encoder(x)

        logits = self.decoder(b, e2, e1, s0)  # [B, C, D, H, W]

        return logits


class ResUNet3DWLateFusion(ResUNet3D):
    """
    A small, fast ResUNet-like 3D classifier/regressor which fuses external tabular variables between the encoder and
    the decoder. Modification is made by creating new uniform channels with the specified tabular variables.
    Depth: 3 downs / 3 ups. Channels: 8-16-32 (match your backbone).
    Returns logits of shape [B, out_classes, D, H, W].
    """
    def __init__(self, in_channels, out_classes, base=(8,16,32), groups=8, drop_path_max=0.1, anisotropic=False,
                 tab_vars=2, modified_channel_idx=0):
        super().__init__(in_channels, out_classes, base, groups, drop_path_max, anisotropic)
        self.modified_channel_idx = modified_channel_idx
        c1, c2, c3 = base

        new_base = (c1, c2, c3+tab_vars)

        # Decoder (up + fuse)
        self.decoder = ResUNet3DDecoder(out_classes, new_base, groups, drop_path_max)

    def forward(self, x, tab_vars):
        # full-res skip
        s0 = self.stem(x)

        b, e1, e2, e3 = self.encoder(x)

        # expand tab vars to image shape
        expanded_tab = tab_vars[:, :, None, None, None] * torch.ones(
            [tab_vars.shape[0], tab_vars.shape[1]] + list(b.shape[2:]), device=tab_vars.device, dtype=x.dtype)

        # generate new modified channel
        b = torch.cat([expanded_tab, b], dim=1)

        logits = self.decoder(b, e2, e1, s0)  # [B, C, D, H, W]

        return logits


class ResUNet3DEncoder(nn.Module):
    def __init__(self, in_channels, base=(8,16,32), groups=8, drop_path_max=0.1, anisotropic=False):
        super().__init__()

        c1, c2, c3 = base

        s1 = (2, 2, 1) if anisotropic else 2
        s2 = (2, 2, 1) if anisotropic else 2
        s3 = 2  # usually safe to downsample depth later even if anisotropic

        # Stochastic depth schedule across encoder+decoder blocks (light)
        # Order: enc1, enc2, enc3, bottleneck, dec2, dec1, dec0  (7 blocks)
        sched = torch.linspace(0.0, drop_path_max, steps=4).tolist()

        self.enc1 = nn.Sequential(
            ResBlock3D(in_channels, c1, stride=s1, groups=groups, p_drop=sched[0]),
            ResBlock3D(c1, c1, stride=1, groups=groups, p_drop=sched[0])
        )

        self.enc2 = nn.Sequential(
            ResBlock3D(c1, c2, stride=s2, groups=groups, p_drop=sched[1]),
            ResBlock3D(c2, c2, stride=1, groups=groups, p_drop=sched[1])
        )

        self.enc3 = nn.Sequential(
            ResBlock3D(c2, c3, stride=s3, groups=groups, p_drop=sched[2]),
            ResBlock3D(c3, c3, stride=1, groups=groups, p_drop=sched[2])
        )

        # Bottleneck (no downsample)
        self.bot  = ResBlock3D(c3, c3, stride=1, groups=groups, p_drop=sched[3])

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)   # c1, /2 or /(1,2,2)
        e2 = self.enc2(e1)  # c2, /4
        e3 = self.enc3(e2)  # c3, /8

        encoding  = self.bot(e3)   # c3

        return encoding, e1, e2, e3


class ResUNet3DDecoder(nn.Module):
    def __init__(self, out_classes, base=(8, 16, 32), groups=8, drop_path_max=0.1):
        super().__init__()

        c1, c2, c3 = base

        # Stochastic depth schedule across encoder+decoder blocks (light)
        # Order: enc1, enc2, enc3, bottleneck, dec2, dec1, dec0  (7 blocks)
        sched = torch.linspace(drop_path_max, 0.0, steps=4).tolist()

        # Decoder (up + fuse)
        self.up2  = UpFuse(in_ch=c3, skip_ch=c2, out_ch=c2, groups=groups, p_drop=sched[0])
        self.up2_ = ResBlock3D(c2, c2, stride=1, groups=groups, p_drop=sched[0])

        self.up1  = UpFuse(in_ch=c2, skip_ch=c1, out_ch=c1, groups=groups, p_drop=sched[1])
        self.up1_ = ResBlock3D(c1, c1, stride=1, groups=groups, p_drop=sched[1])

        # final up to full-res, fuse with stem skip
        self.up0  = UpFuse(in_ch=c1, skip_ch=c1, out_ch=c1, groups=groups, p_drop=sched[2])
        self.up0_ = ResBlock3D(c1, c1, stride=1, groups=groups, p_drop=sched[2])

        # Classification/Regression head
        self.head = nn.Conv3d(c1, out_classes, kernel_size=1, bias=True)

    def forward(self, x, e2, e1, s0):
        # decoder with skip connections
        d2 = self.up2_(self.up2(x,  e2))   # -> c2, /4
        d1 = self.up1_(self.up1(d2, e1))  # -> c1, /2
        d0 = self.up0_(self.up0(d1, s0))   # -> c1, /1

        out = self.head(d0)

        return out


class SimpleCNN(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.flatten(self.model(x))


class SimpleCNNTest(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.flatten(self.model(x))

class SimpleCNNTest2(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Dropout(p=config['MODEL']['dropout_prob']),
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNTest3(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            torch.nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Dropout(p=config['MODEL']['dropout_prob']),
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNTest4(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(

            ResBlock3D(in_channels, 8, stride=(2, 2, 1), p_drop=0.1),

            ResBlock3D(8, 16, stride=(2, 2, 1), p_drop=0.1),

            ResBlock3D(16, 32, stride=2, p_drop=0.1),

            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Dropout(p=config['MODEL']['dropout_prob']),
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNTest5(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        max_p_drop = config['MODEL']['dropout_prob']

        self.model = nn.Sequential(

            ResBlock3D(in_channels, 8, stride=(2, 2, 1), p_drop=0),

            ResBlock3D(8, 16, stride=(2, 2, 1), p_drop=max_p_drop / 2),

            ResBlock3D(16, 32, stride=2, p_drop=max_p_drop),

            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNTest6(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 32, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Dropout(p=config['MODEL']['dropout_prob']),
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNTest7(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 64, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 8, kernel_size=3, padding=1),
            torch.nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Dropout(p=config['MODEL']['dropout_prob']),
            nn.Flatten(),  # (B, 32)
            nn.Linear(8, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNTest8(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        max_p_drop = config['MODEL']['dropout_prob']

        self.model = nn.Sequential(

            ResBlock3D(in_channels, 8, stride=(2, 2, 1), p_drop=0),
            ResBlock3D(8, 8, stride=1, p_drop=0),

            ResBlock3D(8, 16, stride=(2, 2, 1), p_drop=max_p_drop / 2),
            ResBlock3D(16, 16, stride=1, p_drop=max_p_drop / 2),

            ResBlock3D(16, 32, stride=2, p_drop=max_p_drop),
            ResBlock3D(32, 32, stride=1, p_drop=max_p_drop),

            nn.AdaptiveAvgPool3d(1),  # (B, 32, 1, 1, 1)
            nn.Flatten(),  # (B, 32)
            nn.Linear(32, out_classes),  # (B, 1)
        )

    def forward(self, x):
        return self.model(x)


class SimpleCNNBackboneTest1(pl.LightningModule):
    def __init__(self, in_channels, out_classes, config):
        super().__init__()
        self.save_hyperparameters()

        max_p_drop = config['MODEL']['dropout_prob']

        self.model = nn.Sequential(

            ResBlock3D(in_channels, 8, stride=(2, 2, 1), p_drop=0),
            ResBlock3D(8, 8, stride=1, p_drop=0),

            ResBlock3D(8, 16, stride=(2, 2, 1), p_drop=max_p_drop / 2),
            ResBlock3D(16, 16, stride=1, p_drop=max_p_drop / 2),

            ResBlock3D(16, 32, stride=2, p_drop=max_p_drop),
            ResBlock3D(32, out_classes, stride=1, p_drop=max_p_drop),

            nn.AdaptiveAvgPool3d(1),  # (B, out_classes, 1, 1, 1)
            nn.Flatten(),  # (B, out_classes)
        )

    def forward(self, x):
        return self.model(x)
