"""Dataset that loads clean crops and synthesizes noisy inputs on the fly."""

import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from denoising.data.transforms import DegradationPipeline


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
            seed = hash(name) & 0xFFFFFFFF
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)
            noisy = self.pipeline(clean)
            random.setstate(py_state)
            np.random.set_state(np_state)
        else:
            noisy = self.pipeline(clean)

        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean
