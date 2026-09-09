"""LightningDataModule for the document denoising dataset."""

import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from denoising.data.dataset import DocumentDenoisingDataset, split_names
from denoising.data.transforms import DegradationPipeline


class DenoisingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        clean_dir: str,
        degradations: list[dict],
        img_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
        test_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        names = [f for f in os.listdir(self.hparams.clean_dir) if f.endswith(".png")]
        train_names, test_names = split_names(names, self.hparams.seed, self.hparams.test_fraction)

        pipeline = DegradationPipeline.from_config(self.hparams.degradations)

        transform = transforms.Compose([
            transforms.Resize(
                (self.hparams.img_size, self.hparams.img_size),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
        ])

        self.train_set = DocumentDenoisingDataset(
            self.hparams.clean_dir, train_names, pipeline,
            transform=transform, deterministic=False,
        )
        self.test_set = DocumentDenoisingDataset(
            self.hparams.clean_dir, test_names, pipeline,
            transform=transform, deterministic=True,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
