"""Degradation transforms and composable pipeline for noise synthesis."""

import io
import random

import numpy as np
from PIL import Image, ImageFilter


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode img as JPEG at the given quality and decode it back. Deterministic,
    single-quality version of JPEGNoise's core op — used by predict.py for demo grids."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JPEGNoise:
    def __init__(self, quality_min: int = 20, quality_max: int = 75, prob: float = 1.0):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        q = random.randint(self.quality_min, self.quality_max)
        return jpeg_roundtrip(img, q)


class GaussianNoise:
    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 0.05, prob: float = 1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + np.random.normal(0, sigma, arr.shape).astype(np.float32), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class GaussianBlur:
    def __init__(self, radius_min: float = 0.1, radius_max: float = 1.0, prob: float = 1.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.prob = prob

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        radius = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class DegradationPipeline:
    """Composes a sequence of degradation steps, each applied in order.

    Build from a list of dicts via ``DegradationPipeline.from_config(cfg.data.degradations)``.
    Each dict must have a ``type`` key matching one of the registered transforms.
    All other keys are passed as kwargs to the transform constructor.

    Example config entry::

        - type: jpeg
          prob: 1.0
          quality_min: 20
          quality_max: 75
    """

    _REGISTRY: dict[str, type] = {
        "jpeg": JPEGNoise,
        "gaussian_noise": GaussianNoise,
        "gaussian_blur": GaussianBlur,
    }

    def __init__(self, steps: list) -> None:
        self.steps = steps

    @classmethod
    def from_config(cls, cfg: list[dict]) -> "DegradationPipeline":
        steps = []
        for entry in cfg:
            entry = dict(entry)
            kind = entry.pop("type")
            if kind not in cls._REGISTRY:
                raise ValueError(f"Unknown degradation type '{kind}'. Available: {list(cls._REGISTRY)}")
            steps.append(cls._REGISTRY[kind](**entry))
        return cls(steps)

    def __call__(self, img: Image.Image) -> Image.Image:
        for step in self.steps:
            img = step(img)
        return img
