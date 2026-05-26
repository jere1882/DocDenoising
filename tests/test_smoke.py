"""Smoke tests: catch shape mismatches and broken init before a real training run."""

import torch
import torch.nn.functional as F

from denoising.models.lit_module import DenoisingLitModule
from denoising.models.unet import ResidualUNet, UNet
from denoising.utils.metrics import psnr_from_mse

_BACKBONE_CFG = {"in_channels": 3, "out_channels": 3, "base_channels": 8}


def test_unet_forward_shape():
    model = UNet(base_channels=8)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == x.shape
    assert 0.0 <= y.min().item() and y.max().item() <= 1.0


def test_residual_unet_forward_shape():
    model = ResidualUNet(base_channels=8)
    x = torch.rand(2, 3, 64, 64)  # rand not randn — input must be in [0,1]
    y = model(x)
    assert y.shape == x.shape
    assert 0.0 <= y.min().item() and y.max().item() <= 1.0


def test_litmodule_forward_and_backward():
    lit = DenoisingLitModule(backbone="unet", backbone_cfg=_BACKBONE_CFG)
    noisy = torch.randn(2, 3, 64, 64)
    clean = torch.rand(2, 3, 64, 64)
    out = lit(noisy)
    assert out.shape == clean.shape
    loss = F.mse_loss(out, clean)
    loss.backward()
    assert torch.isfinite(loss)


def test_psnr_from_mse():
    assert psnr_from_mse(0.0) == float("inf")
    assert abs(psnr_from_mse(0.01) - 20.0) < 1e-6
    assert abs(psnr_from_mse(1.0) - 0.0) < 1e-6
