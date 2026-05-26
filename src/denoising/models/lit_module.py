"""LightningModule wrapping a denoising backbone. Loss, optimizer, metric logging."""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from denoising.models.swin2sr import Swin2SRDenoiser
from denoising.models.unet import PlainEncoderDecoder, ResidualUNet, UNet
from denoising.utils.metrics import psnr_from_mse

_BACKBONES: dict[str, type[nn.Module]] = {
    "unet": UNet,
    "plain_encoder_decoder": PlainEncoderDecoder,
    "residual_unet": ResidualUNet,
    "swin2sr": Swin2SRDenoiser,
}

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
}

_OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


class DenoisingLitModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "unet",
        backbone_cfg: dict | None = None,
        lr: float = 1e-3,
        optimizer: str = "adam",
        loss: str = "mse",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = _BACKBONES[backbone](**(backbone_cfg or {}))
        self.criterion = _LOSSES[loss]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        noisy, clean = batch
        loss = self.criterion(self(noisy), clean)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        noisy, clean = batch
        out = self(noisy).clamp(0, 1)
        loss = self.criterion(out, clean)
        psnr = psnr_from_mse(((out - clean) ** 2).mean().item())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return _OPTIMIZERS[self.hparams.optimizer](self.parameters(), lr=self.hparams.lr)
