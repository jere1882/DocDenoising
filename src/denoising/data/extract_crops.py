"""Generate clean crops from PDF pages.

For each PDF, render randomly-sampled pages to images and pull random
high-quality square crops. Crops that are mostly blank (low std or near-white
mean) are rejected. Noise is generated on-the-fly during training
(see train.py); the noisy/ folder is optional and off by default.

Output layout:
    <output-dir>/clean/<pdf-stem>_page<N>_x<X>_y<Y>.png
    <output-dir>/noisy/<pdf-stem>_page<N>_x<X>_y<Y>.png  (only with --also-noisy)
"""

import argparse
import os
import random
import tempfile
from pathlib import Path

import cv2
import fitz
import numpy as np


def add_compression_artifacts(image, quality):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_filename = tmp_file.name
        cv2.imwrite(temp_filename, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        degraded_image = cv2.imread(temp_filename)
    os.remove(temp_filename)
    return degraded_image


def page_to_image(doc, page_num, scale):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def is_high_quality(crop, min_std, max_mean):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return gray.std() >= min_std and gray.mean() <= max_mean


def sample_random_crops_from_page(image, crop_size, n_samples, min_std, max_mean, rng):
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return [], 0
    crops = []
    rejected = 0
    max_attempts = n_samples * 4
    attempts = 0
    while len(crops) < n_samples and attempts < max_attempts:
        y = rng.randint(0, h - crop_size)
        x = rng.randint(0, w - crop_size)
        crop = image[y : y + crop_size, x : x + crop_size]
        if is_high_quality(crop, min_std, max_mean):
            crops.append((x, y, crop))
        else:
            rejected += 1
        attempts += 1
    return crops, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default="data/source_pdfs")
    parser.add_argument("--output-dir", default="data/crops")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--scale", type=float, default=3.0, help="PDF render scale (3.0 ≈ 216 DPI, 2.0 ≈ 144 DPI)")
    parser.add_argument("--max-pages-per-pdf", type=int, default=None,
                        help="randomly sample at most this many pages from each PDF")
    parser.add_argument("--crops-per-page", type=int, default=6,
                        help="number of random crops to attempt per page (after quality filter)")
    parser.add_argument("--min-std", type=float, default=15.0,
                        help="reject crops with grayscale std below this (filters blank/uniform regions)")
    parser.add_argument("--max-mean", type=float, default=245.0,
                        help="reject crops with grayscale mean above this (filters mostly-white blanks)")
    parser.add_argument("--target-crops", type=int, default=None,
                        help="stop after writing this many crops")
    parser.add_argument("--target-size-gb", type=float, default=None,
                        help="stop after written crops exceed this many GB")
    parser.add_argument("--also-noisy", action="store_true",
                        help="also write a paired noisy crop (mostly for the legacy demo). "
                             "Training generates noise on-the-fly.")
    parser.add_argument("--quality", type=int, default=10,
                        help="JPEG quality for the paired noisy crop (only used with --also-noisy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-pdfs", type=int, default=None,
                        help="cap on number of PDFs to process (useful for quick smoke tests)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    pdf_dir = Path(args.pdf_dir).expanduser()
    out_dir = Path(args.output_dir)
    clean_dir = out_dir / "clean"
    noisy_dir = out_dir / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    if args.also_noisy:
        noisy_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    rng.shuffle(pdf_files)
    if args.limit_pdfs:
        pdf_files = pdf_files[: args.limit_pdfs]
    if not pdf_files:
        raise SystemExit(f"no PDFs found in {pdf_dir}")

    target_size_bytes = int(args.target_size_gb * 1024**3) if args.target_size_gb else None

    print(f"processing up to {len(pdf_files)} PDFs from {pdf_dir}")
    if args.target_crops or target_size_bytes:
        print(f"  stop targets: crops={args.target_crops}, size_gb={args.target_size_gb}")

    total_crops = 0
    total_bytes = 0
    rejected = 0

    def hit_target():
        if args.target_crops and total_crops >= args.target_crops:
            return True
        if target_size_bytes and total_bytes >= target_size_bytes:
            return True
        return False

    for i, pdf_path in enumerate(pdf_files):
        if hit_target():
            break
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  skipping {pdf_path.name}: {e}")
            continue
        n_pages = len(doc)
        if args.max_pages_per_pdf is not None and n_pages > args.max_pages_per_pdf:
            page_indices = sorted(rng.sample(range(n_pages), args.max_pages_per_pdf))
        else:
            page_indices = list(range(n_pages))

        for page_num in page_indices:
            if hit_target():
                break
            try:
                pix = page_to_image(doc, page_num, args.scale)
            except Exception as e:
                print(f"  skipping page {page_num} of {pdf_path.name}: {e}")
                continue

            page_crops, page_rejected = sample_random_crops_from_page(
                pix, args.crop_size, args.crops_per_page, args.min_std, args.max_mean, rng,
            )
            rejected += page_rejected

            for x, y, crop in page_crops:
                if hit_target():
                    break
                filename = f"{pdf_path.stem}_page{page_num}_x{x}_y{y}.png"
                out_path = clean_dir / filename
                cv2.imwrite(str(out_path), crop)
                total_crops += 1
                total_bytes += out_path.stat().st_size
                if args.also_noisy:
                    noisy = add_compression_artifacts(crop, args.quality)
                    cv2.imwrite(str(noisy_dir / filename), noisy, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        doc.close()

        if (i + 1) % 25 == 0 or i == len(pdf_files) - 1 or hit_target():
            mb = total_bytes / 1024**2
            print(f"  [{i + 1}/{len(pdf_files)}] crops: {total_crops}, size: {mb:.1f} MB, rejected: {rejected}")

    gb = total_bytes / 1024**3
    print(f"\nWrote {total_crops} clean crops ({gb:.2f} GB) to {clean_dir}"
          f"{f' (+ noisy in {noisy_dir})' if args.also_noisy else ''}")
    print(f"Rejected {rejected} low-quality crop attempts.")


if __name__ == "__main__":
    main()
