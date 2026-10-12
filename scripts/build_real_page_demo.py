"""Find the held-out crops where denoising helps the most, then render and tile-denoise
their REAL full source pages (not a synthetic mosaic) at the verified render scale.

Scale verification: rendering page 59 of 0000231.pdf at fitz.Matrix(4,4) and cropping at
(x=565, y=1572, 512, 512) exactly matches the stored crop 0000231_page59_x565_y1572.png,
pixel for pixel (max abs diff 0). So PDF_RENDER_SCALE below is not a guess.
"""
import os
import re
from pathlib import Path

import fitz
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

import denoising
from denoising.data.dataset import split_names
from denoising.data.transforms import DegradationPipeline, apply_deterministic
from denoising.infer import make_predictor, tile_denoise
from denoising.utils.metrics import psnr_tensor

REPO = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO / "outputs/dataset_50k/crops50k/clean"
PDF_DIR = Path("/Users/jere/Downloads/0000")
OUT = REPO / "outputs/real_page_demo"
EXP = REPO / "outputs/experiments"
CHECKPOINT = EXP / "residual_unet_bc64/checkpoints/denoising-epoch=09-val_psnr=30.82.ckpt"

PDF_RENDER_SCALE = 4.0  # verified exact match against a stored crop, see module docstring
PREFIX_SCAN = 800
N_PICK = 2
CROP_RE = re.compile(r"(\d+)_page(\d+)_x(\d+)_y(\d+)\.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith(".png"))
    _, val_names = split_names(names, 42, 0.1)
    scan = val_names[:PREFIX_SCAN]

    data_cfg = OmegaConf.load(Path(denoising.__file__).parent / "conf/data/default.yaml")
    pipeline = DegradationPipeline.from_config(OmegaConf.to_container(data_cfg.degradations, resolve=True))
    tf = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    device = torch.device("cpu")
    torch.set_num_threads(4)
    predict = make_predictor(str(CHECKPOINT), None, device)

    print(f"scanning {len(scan)} held-out crops for denoising gain (PSNR_denoised - PSNR_noisy)...")
    gains = []
    for name in scan:
        clean = Image.open(CLEAN_DIR / name).convert("RGB")
        noisy_pil = apply_deterministic(pipeline, clean, name)
        clean_t, noisy_t = tf(clean), tf(noisy_pil)
        with torch.no_grad():
            denoised_t = predict(noisy_t.unsqueeze(0))[0]
        gain = psnr_tensor(denoised_t, clean_t) - psnr_tensor(noisy_t, clean_t)
        gains.append((gain, name))

    gains.sort(reverse=True)
    print("top 5 gains:", gains[:5])
    picked = gains[:N_PICK]

    for rank, (gain, name) in enumerate(picked):
        m = CROP_RE.match(name)
        stem, page_num, x, y = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        pdf_path = PDF_DIR / f"{stem}.pdf"
        print(f"\nrank {rank}: crop={name} gain={gain:.2f}dB  source={pdf_path} page={page_num} crop@({x},{y})")

        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE))
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3]
        elif pix.n == 1:
            arr = np.repeat(arr, 3, axis=2)
        clean_page = Image.fromarray(arr, mode="RGB")
        doc.close()
        print(f"  rendered page size: {clean_page.size}")

        noisy_page = apply_deterministic(pipeline, clean_page, f"{stem}_page{page_num}_fullpage")
        denoised_page = tile_denoise(noisy_page, predict, tile_size=512, overlap=32, batch_size=1)

        tag = f"rank{rank}_{stem}_page{page_num}"
        clean_page.save(OUT / f"{tag}_clean.png")
        noisy_page.save(OUT / f"{tag}_noisy.png")
        denoised_page.save(OUT / f"{tag}_denoised.png")
        print(f"  wrote {OUT}/{tag}_{{clean,noisy,denoised}}.png")


if __name__ == "__main__":
    main()
