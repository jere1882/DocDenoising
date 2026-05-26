"""Denoise an arbitrary-sized document image using overlapping tiles."""

import math
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image

from denoising.models.lit_module import DenoisingLitModule
from denoising.utils.device import pick_device


def _hann_window(size: int) -> torch.Tensor:
    """2D Hann window for smooth tile blending at seams."""
    h = torch.hann_window(size, periodic=False)
    return (h.unsqueeze(0) * h.unsqueeze(1)).unsqueeze(0)  # 1, H, W


def _padded_size(size: int, tile_size: int, stride: int) -> int:
    if size <= tile_size:
        return tile_size
    return tile_size + math.ceil((size - tile_size) / stride) * stride


def tile_denoise(
    image: Image.Image,
    model: torch.nn.Module,
    tile_size: int = 256,
    overlap: int = 32,
    device: torch.device | None = None,
) -> Image.Image:
    """Denoise an arbitrary-sized image by running the model on overlapping tiles.

    Overlapping regions are blended with a 2D Hann window to eliminate seam
    artifacts. The output is cropped back to the original image dimensions.
    """
    if device is None:
        device = next(model.parameters()).device

    stride = tile_size - overlap
    img = T.ToTensor()(image)  # C, H, W
    C, H, W = img.shape

    pH = _padded_size(H, tile_size, stride)
    pW = _padded_size(W, tile_size, stride)
    img_padded = F.pad(img.unsqueeze(0), (0, pW - W, 0, pH - H), mode="reflect").squeeze(0)

    out = torch.zeros(C, pH, pW)
    wgt = torch.zeros(1, pH, pW)
    window = _hann_window(tile_size)

    for y in range(0, pH - tile_size + 1, stride):
        for x in range(0, pW - tile_size + 1, stride):
            tile = img_padded[:, y : y + tile_size, x : x + tile_size].unsqueeze(0).to(device)
            with torch.no_grad():
                denoised = model(tile).clamp(0, 1).squeeze(0).cpu()
            out[:, y : y + tile_size, x : x + tile_size] += denoised * window
            wgt[:, y : y + tile_size, x : x + tile_size] += window

    result = (out / wgt.clamp(min=1e-8))[:, :H, :W]
    return T.ToPILImage()(result)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = pick_device()
    print(f"device: {device}")

    model = DenoisingLitModule.load_from_checkpoint(cfg.infer.checkpoint, map_location=device)
    model.eval().to(device)

    input_path = Path(cfg.infer.input)
    if cfg.infer.output:
        output_path = Path(cfg.infer.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_denoised" + input_path.suffix)

    image = Image.open(input_path).convert("RGB")
    W, H = image.size
    print(f"input:  {input_path}  ({W}×{H})")

    denoised = tile_denoise(
        image, model,
        tile_size=cfg.infer.tile_size,
        overlap=cfg.infer.overlap,
        device=device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    denoised.save(output_path)
    print(f"output: {output_path}  ({W}×{H})")


if __name__ == "__main__":
    main()
