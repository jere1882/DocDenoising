"""Training callbacks."""

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger


class LogPredictionsCallback(pl.Callback):
    """Logs a grid of noisy / denoised / clean image triplets to W&B each validation epoch.

    Only activates when the trainer is using WandbLogger — silently does nothing otherwise.
    """

    def __init__(self, num_samples: int = 4) -> None:
        self.num_samples = num_samples
        self._batch: tuple | None = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero:
            return
        if batch_idx == 0:
            self._batch = batch

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero:
            return
        if self._batch is None:
            return
        if not isinstance(trainer.logger, WandbLogger):
            return

        import wandb

        noisy, clean = self._batch
        n = min(self.num_samples, noisy.size(0))
        noisy, clean = noisy[:n], clean[:n]

        # Grab filenames for captions — val loader is unshuffled so batch 0 = first n names.
        try:
            names = trainer.datamodule.test_set.names[:n]
        except AttributeError:
            names = [f"sample_{i}" for i in range(n)]

        with torch.no_grad():
            denoised = pl_module(noisy).clamp(0, 1)

        images = []
        for noi, den, cln, name in zip(noisy.cpu(), denoised.cpu(), clean.cpu(), names):
            grid = torch.cat([noi, den, cln], dim=2)  # concat along width
            images.append(wandb.Image(grid, caption=f"noisy | denoised | clean — {name}"))

        # No explicit step= here: wandb's own docs warn that setting step under a
        # multi-process (DDP) run risks silent data loss (its internal step-tracker
        # can conflict with the step values Lightning's own metric logging advances
        # via other channels). Confirmed directly: a real DDP run logged this exact
        # warning, and its 4 predicted images never made it to the server (metadata
        # reference existed, the actual file upload didn't) -- letting wandb
        # auto-assign the step avoids that race.
        trainer.logger.experiment.log({"predictions": images, "epoch": trainer.current_epoch})
        self._batch = None
