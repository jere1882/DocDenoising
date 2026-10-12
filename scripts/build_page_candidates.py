"""Build a few candidate 'full page' composites (real crop content, mosaic layout),
run tiled inference on each, and save noisy/denoised/clean paths for manual inspection.

Candidates are picked for visual variety using the same colorfulness ranking used
earlier in this project, so some candidates are text-heavy and some are photo/color-heavy.
"""
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

import denoising
from denoising.data.dataset import split_names
from denoising.data.transforms import DegradationPipeline, apply_deterministic
from denoising.infer import make_predictor, tile_denoise

REPO = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO / "outputs/dataset_50k/crops50k/clean"
OUT = REPO / "outputs/page_candidates"
EXP = REPO / "outputs/experiments"

GRID_ROWS, GRID_COLS = 2, 3  # 1536x1024 composite
N_CANDIDATES = 4
PREFIX_SCAN = 500
CHECKPOINT = EXP / "residual_unet_bc64/checkpoints/denoising-epoch=09-val_psnr=30.82.ckpt"


def colorfulness(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith(".png"))
    _, val_names = split_names(names, 42, 0.1)
    candidates = val_names[:PREFIX_SCAN]

    scored = sorted(((colorfulness(Image.open(CLEAN_DIR / n)), n) for n in candidates), reverse=True)
    needed = N_CANDIDATES * GRID_ROWS * GRID_COLS
    top = [n for _, n in scored[:needed]]
    print(f"picked {len(top)} crops across {N_CANDIDATES} candidates")

    data_cfg = OmegaConf.load(Path(denoising.__file__).parent / "conf/data/default.yaml")
    pipeline = DegradationPipeline.from_config(OmegaConf.to_container(data_cfg.degradations, resolve=True))

    device = torch.device("cpu")
    torch.set_num_threads(4)
    predict = make_predictor(str(CHECKPOINT), None, device)

    per_page = GRID_ROWS * GRID_COLS
    for i in range(N_CANDIDATES):
        page_names = top[i * per_page : (i + 1) * per_page]
        composite = Image.new("RGB", (512 * GRID_COLS, 512 * GRID_ROWS))
        for idx, name in enumerate(page_names):
            r, c = divmod(idx, GRID_COLS)
            crop = Image.open(CLEAN_DIR / name).convert("RGB").resize((512, 512))
            composite.paste(crop, (c * 512, r * 512))

        noisy = apply_deterministic(pipeline, composite, f"page_candidate_{i}")
        denoised = tile_denoise(noisy, predict, tile_size=512, overlap=32, batch_size=1)

        clean_p = OUT / f"candidate_{i}_clean.png"
        noisy_p = OUT / f"candidate_{i}_noisy.png"
        den_p = OUT / f"candidate_{i}_denoised.png"
        composite.save(clean_p)
        noisy.save(noisy_p)
        denoised.save(den_p)
        print(f"candidate {i}: crops={page_names}")
        print(f"  before (noisy):    {noisy_p}")
        print(f"  after  (denoised): {den_p}")
        print(f"  reference (clean): {clean_p}")


if __name__ == "__main__":
    main()
