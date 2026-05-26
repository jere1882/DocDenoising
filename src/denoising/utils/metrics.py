"""Image quality metrics."""

import math

import torch


def psnr_from_mse(mse: float, peak: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(peak ** 2 / mse)


def psnr_tensor(a: torch.Tensor, b: torch.Tensor, peak: float = 1.0) -> float:
    return psnr_from_mse(((a - b) ** 2).mean().item(), peak=peak)
