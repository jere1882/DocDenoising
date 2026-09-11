"""Regenerate a prediction-evolution figure from per-epoch checkpoints.

Point this at a run directory produced by ``denoising-train ... save_all_epochs=true``.
It reads that run's own Hydra config (so the validation split, degradation pipeline,
and image size exactly match training), runs every ``epoch*.ckpt`` on a handful of
fixed held-out crops, and saves one grid PNG showing how the output sharpens over epochs.

    denoising-viz-evolution --run-dir outputs/run_10k
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from denoising.data.dataset import split_names
from denoising.data.transforms import DegradationPipeline, apply_deterministic
from denoising.models.lit_module import DenoisingLitModule


def _epoch_of(path: str) -> int:
    m = re.search(r"epoch=?(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="a denoising-train run directory")
    parser.add_argument("--num-samples", type=int, default=6, help="held-out crops to visualize")
    parser.add_argument("--output", default=None, help="output PNG (default: <run-dir>/evolution.png)")
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, ".hydra", "config.yaml"))
    clean_dir = cfg.data.clean_dir
    img_size = cfg.data.img_size
    seed = cfg.seed

    # Rebuild the exact validation set training used, then take a fixed handful.
    names = [f for f in os.listdir(clean_dir) if f.endswith(".png")]
    _, val_names = split_names(names, seed, cfg.data.test_fraction)
    sample_names = val_names[: args.num_samples]

    pipeline = DegradationPipeline.from_config(OmegaConf.to_container(cfg.data.degradations, resolve=True))
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])

    noisy, clean = [], []
    for name in sample_names:
        c = Image.open(os.path.join(clean_dir, name)).convert("RGB")
        noisy.append(transform(apply_deterministic(pipeline, c, name)))
        clean.append(transform(c))
    noisy = torch.stack(noisy)
    clean = torch.stack(clean)

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints", "epoch*.ckpt")), key=_epoch_of)
    if not ckpts:
        raise SystemExit(
            f"no epoch*.ckpt in {args.run_dir}/checkpoints — train with save_all_epochs=true"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = []  # (epoch_label, [n,H,W,3] uint8)
    for ck in ckpts:
        model = DenoisingLitModule.load_from_checkpoint(ck, map_location=device).eval().to(device)
        with torch.no_grad():
            out = model(noisy.to(device)).clamp(0, 1).cpu()
        preds.append((f"epoch {_epoch_of(ck)}", (out.permute(0, 2, 3, 1).numpy() * 255).astype("uint8")))

    n = len(sample_names)
    ncol = 1 + len(preds) + 1  # noisy | each epoch | clean
    fig, axes = plt.subplots(n, ncol, figsize=(1.8 * ncol, 1.8 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        axes[r, 0].imshow(noisy[r].permute(1, 2, 0).numpy())
        if r == 0:
            axes[r, 0].set_title("noisy (input)", fontsize=8)
        for c, (label, imgs) in enumerate(preds, start=1):
            axes[r, c].imshow(imgs[r])
            if r == 0:
                axes[r, c].set_title(label, fontsize=8)
        axes[r, -1].imshow(clean[r].permute(1, 2, 0).numpy())
        if r == 0:
            axes[r, -1].set_title("clean (target)", fontsize=8)
        for c in range(ncol):
            axes[r, c].axis("off")

    fig.suptitle(f"prediction evolution — {os.path.basename(os.path.normpath(args.run_dir))}", y=1.0)
    fig.tight_layout()
    out_path = args.output or os.path.join(args.run_dir, "evolution.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {out_path}  ({len(preds)} epochs x {n} samples)")


if __name__ == "__main__":
    main()
