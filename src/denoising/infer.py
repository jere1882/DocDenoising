"""Denoise an arbitrary-sized document image using overlapping, batched tiles.

Works with either backend: a PyTorch checkpoint or a (fp32 or int8) ONNX export.
Tiles are collected up front and run through the model in batches, not one at a
time -- this is also how you measure whether batching actually helps on CPU.
"""

import math
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image

from denoising.models.lit_module import DenoisingLitModule
from denoising.utils.device import pick_device

Predictor = Callable[[torch.Tensor], torch.Tensor]  # (B,C,H,W) in [0,1] -> (B,C,H,W) in [0,1]


def pytorch_predictor(checkpoint: str, device: torch.device) -> Predictor:
    model = DenoisingLitModule.load_from_checkpoint(checkpoint, map_location=device)
    model.eval().to(device)

    def predict(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(batch.to(device)).clamp(0, 1).cpu()

    return predict


def onnx_predictor(onnx_path: str, threads: int = 4) -> Predictor:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])

    def predict(batch: torch.Tensor) -> torch.Tensor:
        out = sess.run(None, {"noisy": batch.numpy().astype(np.float32)})[0]
        return torch.from_numpy(out).clamp(0, 1)

    return predict


def make_predictor(
    checkpoint: str | None, onnx_path: str | None, device: torch.device, threads: int = 4
) -> Predictor:
    """ONNX is used when `onnx_path` is set, otherwise falls back to the PyTorch checkpoint."""
    if onnx_path:
        return onnx_predictor(onnx_path, threads=threads)
    if not checkpoint:
        raise ValueError("Set either infer.checkpoint or infer.onnx_path.")
    return pytorch_predictor(checkpoint, device)


def _hann_window(size: int) -> torch.Tensor:
    """2D Hann window for smooth tile blending at seams.

    `periodic=False` makes the 1D window exactly 0 at both endpoints. At the
    outer border of the whole canvas only one tile covers each pixel, so a
    weight of exactly 0 there means total weight 0 and the output silently
    goes black (confirmed directly: reconstructing a random image through an
    identity predictor left ~0.45% of pixels, concentrated at the border,
    off by up to 254/255). Flooring the window at a small epsilon fixes this
    with negligible effect on interior blending, where two windows overlap.
    """
    h = torch.hann_window(size, periodic=False).clamp(min=1e-3)
    return (h.unsqueeze(0) * h.unsqueeze(1)).unsqueeze(0)  # 1, H, W


def _padded_size(size: int, tile_size: int, stride: int) -> int:
    if size <= tile_size:
        return tile_size
    return tile_size + math.ceil((size - tile_size) / stride) * stride


def tile_denoise(
    image: Image.Image,
    predict: Predictor,
    tile_size: int = 512,
    overlap: int = 32,
    batch_size: int = 8,
) -> Image.Image:
    """Denoise an arbitrary-sized image by running `predict` on overlapping tiles.

    Tiles are gathered first and sent through `predict` in batches of `batch_size`
    (last batch may be smaller) rather than one at a time. Overlapping regions are
    blended with a 2D Hann window to eliminate seam artifacts, and the output is
    cropped back to the original image dimensions.
    """
    stride = tile_size - overlap
    img = T.ToTensor()(image)  # C, H, W
    C, H, W = img.shape

    pH = _padded_size(H, tile_size, stride)
    pW = _padded_size(W, tile_size, stride)
    img_padded = F.pad(img.unsqueeze(0), (0, pW - W, 0, pH - H), mode="reflect").squeeze(0)

    coords = [
        (y, x)
        for y in range(0, pH - tile_size + 1, stride)
        for x in range(0, pW - tile_size + 1, stride)
    ]
    tiles = torch.stack(
        [img_padded[:, y : y + tile_size, x : x + tile_size] for y, x in coords]
    )  # N, C, tile, tile

    out = torch.zeros(C, pH, pW)
    wgt = torch.zeros(1, pH, pW)
    window = _hann_window(tile_size)

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        denoised = predict(tiles[i : i + batch_size])
        for j, (y, x) in enumerate(batch_coords):
            out[:, y : y + tile_size, x : x + tile_size] += denoised[j] * window
            wgt[:, y : y + tile_size, x : x + tile_size] += window

    result = (out / wgt.clamp(min=1e-8))[:, :H, :W]
    return T.ToPILImage()(result)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = pick_device()
    backend = "onnx" if cfg.infer.get("onnx_path") else "pytorch"
    print(f"backend: {backend}  device: {device if backend == 'pytorch' else 'cpu'}")

    predict = make_predictor(cfg.infer.checkpoint, cfg.infer.get("onnx_path"), device)

    input_path = Path(cfg.infer.input)
    if cfg.infer.output:
        output_path = Path(cfg.infer.output)
    else:
        output_path = input_path.parent / (input_path.stem + "_denoised" + input_path.suffix)

    image = Image.open(input_path).convert("RGB")
    W, H = image.size
    print(f"input:  {input_path}  ({W}×{H})")

    denoised = tile_denoise(
        image, predict,
        tile_size=cfg.infer.tile_size,
        overlap=cfg.infer.overlap,
        batch_size=cfg.infer.get("batch_size", 8),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    denoised.save(output_path)
    print(f"output: {output_path}  ({W}×{H})")


if __name__ == "__main__":
    main()
