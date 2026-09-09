"""Dataset that loads clean crops and synthesizes noisy inputs on the fly."""

import os
import random

from PIL import Image
from torch.utils.data import Dataset

from denoising.data.transforms import DegradationPipeline, apply_deterministic


def split_names(names, seed: int, test_fraction: float) -> tuple[list[str], list[str]]:
    """Deterministic train/val filename split.

    Shared by DenoisingDataModule and predict.py so both see the exact same
    held-out validation set for a given (seed, test_fraction) — predict.py
    needs this to sample real validation crops instead of the whole pool.
    """
    names = sorted(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, int(round(len(names) * test_fraction)))
    return names[n_test:], names[:n_test]  # train, val


class DocumentDenoisingDataset(Dataset):
    """Applies a DegradationPipeline to each clean crop to produce the noisy input.

    With ``deterministic=True`` the pipeline's random state is seeded by filename
    before each call — used for the test set so PSNR is comparable across epochs.
    """

    def __init__(
        self,
        clean_dir: str,
        names,
        pipeline: DegradationPipeline,
        transform=None,
        deterministic: bool = False,
    ):
        self.clean_dir = clean_dir
        self.names = list(names)
        self.pipeline = pipeline
        self.transform = transform
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        clean = Image.open(os.path.join(self.clean_dir, name)).convert("RGB")

        if self.deterministic:
            noisy = apply_deterministic(self.pipeline, clean, name)
        else:
            noisy = self.pipeline(clean)

        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean
