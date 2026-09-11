"""Train the denoising model. Hydra-driven entry point."""

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from denoising.callbacks import LogPredictionsCallback
from denoising.data.datamodule import DenoisingDataModule
from denoising.models.lit_module import DenoisingLitModule


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = DenoisingDataModule(
        clean_dir=cfg.data.clean_dir,
        degradations=OmegaConf.to_container(cfg.data.degradations, resolve=True),
        img_size=cfg.data.img_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        test_fraction=cfg.data.test_fraction,
        max_train_samples=cfg.data.get("max_train_samples"),
        seed=cfg.seed,
    )

    model = DenoisingLitModule(
        backbone=cfg.model.backbone,
        backbone_cfg=OmegaConf.to_container(cfg.model.backbone_cfg, resolve=True),
        lr=cfg.model.lr,
        optimizer=cfg.model.optimizer,
        loss=cfg.model.loss,
    )

    run_name = (
        f"{cfg.model.backbone}"
        f"-{cfg.model.loss}"
        f"-lr{cfg.model.lr}"
        f"-bs{cfg.data.batch_size}"
    )
    logger = instantiate(cfg.logger, name=run_name)
    # Record the full config on the run — W&B only; CSVLogger's experiment has no `config`.
    if hasattr(getattr(logger, "experiment", None), "config"):
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename="denoising-{epoch:02d}-{val_psnr:.2f}",
            monitor="val_psnr",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        LogPredictionsCallback(num_samples=4),
    ]
    # Keep one checkpoint per epoch (epoch=00.ckpt, epoch=01.ckpt, ...) so the prediction
    # evolution can be regenerated after training with `denoising-viz-evolution`.
    if cfg.get("save_all_epochs", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                filename="{epoch:02d}",  # Lightning renders this as "epoch=NN.ckpt"
                save_top_k=-1,
            )
        )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
