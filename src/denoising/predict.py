"""Generate the demo grid: noisy / denoised / clean for top-N samples by PSNR gain."""

import random
from pathlib import Path

import hydra
import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from denoising.data.transforms import jpeg_roundtrip
from denoising.models.lit_module import DenoisingLitModule
from denoising.utils.device import pick_device
from denoising.utils.metrics import psnr_tensor


def _panel(img: Image.Image, label: str, font, scale: int = 2) -> Image.Image:
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    bar = 32
    w, h = img.size
    out = Image.new("RGB", (w, h + bar), "white")
    out.paste(img, (0, bar))
    ImageDraw.Draw(out).text((10, 6), label, fill="black", font=font)
    return out


def _hcat(imgs, gap: int = 12) -> Image.Image:
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    h = max(i.height for i in imgs)
    out = Image.new("RGB", (w, h), "white")
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def _vcat(imgs, gap: int = 18) -> Image.Image:
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), "white")
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    rng = random.Random(cfg.predict.seed)
    device = pick_device()
    print(f"device: {device}")

    model = DenoisingLitModule.load_from_checkpoint(cfg.predict.checkpoint, map_location=device)
    model.eval().to(device)

    transform = T.Compose([T.Resize((cfg.data.img_size, cfg.data.img_size)), T.ToTensor()])

    clean_dir = Path(cfg.data.clean_dir)
    all_clean = sorted(
        (p for p in clean_dir.iterdir() if p.suffix == ".png"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    pool = all_clean[: max(cfg.predict.num_candidates * 4, 100)]
    rng.shuffle(pool)
    candidates = pool[: cfg.predict.num_candidates]

    results = []
    for clean_path in candidates:
        clean_pil = Image.open(clean_path).convert("RGB")
        noisy_pil = jpeg_roundtrip(clean_pil, cfg.predict.demo_quality)

        n = transform(noisy_pil).unsqueeze(0).to(device)
        c = transform(clean_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            d = model(n).clamp(0, 1)

        pn, pd = psnr_tensor(n, c), psnr_tensor(d, c)
        gain = pd - pn
        print(
            f"{clean_path.name[:48]:50s}  noisy={pn:5.2f}dB  denoised={pd:5.2f}dB  gain={gain:+.2f}dB"
        )
        results.append({
            "name": clean_path.name, "psnr_n": pn, "psnr_d": pd, "gain": gain,
            "n_pil": T.ToPILImage()(n.squeeze(0).cpu()),
            "d_pil": T.ToPILImage()(d.squeeze(0).cpu()),
            "c_pil": T.ToPILImage()(c.squeeze(0).cpu()),
        })

    results.sort(key=lambda r: r["gain"], reverse=True)
    picked = results[: cfg.predict.num_samples]
    print()
    for r in picked:
        print(f"picked: {r['name']}  gain={r['gain']:+.2f}dB")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()

    rows = []
    for r in picked:
        rows.append(_hcat([
            _panel(r["n_pil"], f"Noisy — PSNR {r['psnr_n']:.1f} dB", font, cfg.predict.display_scale),
            _panel(r["d_pil"], f"Denoised — PSNR {r['psnr_d']:.1f} dB  (gain {r['gain']:+.1f} dB)", font, cfg.predict.display_scale),
            _panel(r["c_pil"], "Clean (ground truth)", font, cfg.predict.display_scale),
        ]))
    out_path = Path(cfg.predict.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _vcat(rows).save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
