"""Build a synthetic multi-crop 'page' (real crop content, composite layout) to
exercise and demo tiled inference across multiple 512x512 tiles with overlap,
and to measure whether batching tiles together helps on CPU.

Expects two things to already exist (see the main README and
outputs/experiments/summary.csv for how they were produced):
  - outputs/dataset_50k/crops50k/clean/  the 50k-crop dataset
  - outputs/experiments/<run>/checkpoints|export/  trained checkpoints + ONNX exports

Writes intermediate files to outputs/demo_tiling/ (gitignored) and copies the
final comparison figure into demo_samples/tiling/ (tracked).
"""
import json
import os
import shutil
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms

import denoising
from denoising.data.dataset import split_names
from denoising.data.transforms import DegradationPipeline, apply_deterministic
from denoising.infer import make_predictor, tile_denoise

REPO = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO / "outputs/dataset_50k/crops50k/clean"
OUT = REPO / "outputs/demo_tiling"
DEMO_SAMPLES = REPO / "demo_samples/tiling"
EXP = REPO / "outputs/experiments"

GRID_ROWS, GRID_COLS = 2, 3  # -> 1536x1024 composite, well over one 512 tile

RUNS = {
    "pytorch_fp32_residual_bc64": dict(
        checkpoint=str(EXP / "residual_unet_bc64/checkpoints/denoising-epoch=09-val_psnr=30.82.ckpt"),
        onnx_path=None,
    ),
    "onnx_int8_residual_bc32": dict(
        checkpoint=None,
        onnx_path=str(EXP / "residual_unet_bc32/export/model_int8.onnx"),
    ),
}


def build_composite() -> tuple[Image.Image, Image.Image]:
    names = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith(".png"))
    _, val_names = split_names(names, 42, 0.1)
    picked = val_names[: GRID_ROWS * GRID_COLS]
    print("picked crops:", picked)

    composite = Image.new("RGB", (512 * GRID_COLS, 512 * GRID_ROWS))
    for idx, name in enumerate(picked):
        r, c = divmod(idx, GRID_COLS)
        crop = Image.open(CLEAN_DIR / name).convert("RGB").resize((512, 512))
        composite.paste(crop, (c * 512, r * 512))

    data_cfg = OmegaConf.load(Path(denoising.__file__).parent / "conf/data/default.yaml")
    pipeline = DegradationPipeline.from_config(OmegaConf.to_container(data_cfg.degradations, resolve=True))
    noisy = apply_deterministic(pipeline, composite, "demo_tiling_composite")
    return composite, noisy


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DEMO_SAMPLES.mkdir(parents=True, exist_ok=True)

    composite, noisy = build_composite()
    composite.save(OUT / "clean_composite.png")
    noisy.save(OUT / "noisy_composite.png")
    print("composite size:", composite.size)

    device = torch.device("cpu")
    torch.set_num_threads(4)  # match this project's latency methodology
    timing = {}
    denoised_paths = {}
    for label, cfg in RUNS.items():
        predict = make_predictor(cfg["checkpoint"], cfg["onnx_path"], device)
        timing[label] = {}
        for bs in (1, 8):
            t0 = time.perf_counter()
            out = tile_denoise(noisy, predict, tile_size=512, overlap=32, batch_size=bs)
            timing[label][f"batch_size_{bs}"] = round(time.perf_counter() - t0, 3)
            if bs == 1:
                p = OUT / f"denoised_{label}.png"
                out.save(p)
                denoised_paths[label] = p
        print(label, timing[label])

    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))

    # Stacked noisy / each backend / clean comparison figure for the docs.
    panels = [("noisy input", noisy), ("fp32 PyTorch (bc64)", Image.open(denoised_paths["pytorch_fp32_residual_bc64"])),
              ("int8 ONNX (bc32)", Image.open(denoised_paths["onnx_int8_residual_bc32"])), ("clean target", composite)]
    w, h = noisy.size
    scale = 0.5
    stacked = Image.new("RGB", (int(w * scale), (int(h * scale) + 30) * len(panels)), "white")
    draw = ImageDraw.Draw(stacked)
    y = 0
    for label, img in panels:
        draw.text((5, y), label, fill="black")
        y += 20
        stacked.paste(img.resize((int(w * scale), int(h * scale))), (0, y))
        y += int(h * scale) + 10
    stacked.save(OUT / "before_after_stack.png")
    shutil.copy(OUT / "before_after_stack.png", DEMO_SAMPLES / "before_after_stack.png")
    print("wrote", DEMO_SAMPLES / "before_after_stack.png")


if __name__ == "__main__":
    main()
