"""denoising-export: checkpoint -> fp32 ONNX -> int8 ONNX, with PSNR, size, and CPU latency report."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

import denoising
from denoising.data.dataset import split_names
from denoising.data.transforms import DegradationPipeline, apply_deterministic
from denoising.export.onnx_export import export_onnx, load_backbone, max_abs_diff
from denoising.export.onnx_quantize import quantize_int8
from denoising.utils.metrics import psnr_tensor

import onnxruntime as ort


def _to_tensor(img, img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])
    return tf(img)


def _load_pair(clean_dir, name, pipeline, img_size):
    c = Image.open(os.path.join(clean_dir, name)).convert("RGB")
    noisy = _to_tensor(apply_deterministic(pipeline, c, name), img_size)
    return noisy, _to_tensor(c, img_size)


def _latency_ms(fn, x, repeats=30):
    fn(x)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def export_run(checkpoint, clean_dir, out_dir, n_calib, n_eval, seed, test_fraction, img_size):
    torch.set_num_threads(4)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backbone, hparams = load_backbone(checkpoint)
    base_channels = hparams.get("backbone_cfg", {}).get("base_channels")

    fp32_path = str(out / "model_fp32.onnx")
    int8_path = str(out / "model_int8.onnx")
    export_onnx(backbone, fp32_path, img_size)

    data_cfg = OmegaConf.load(Path(denoising.__file__).parent / "conf" / "data" / "default.yaml")
    pipeline = DegradationPipeline.from_config(OmegaConf.to_container(data_cfg.degradations, resolve=True))

    names = sorted(f for f in os.listdir(clean_dir) if f.endswith(".png"))
    train_names, val_names = split_names(names, seed, test_fraction)
    eval_names = val_names if n_eval <= 0 else val_names[:n_eval]

    calib_arrays = [_load_pair(clean_dir, n, pipeline, img_size)[0].unsqueeze(0).numpy()
                    for n in train_names[:n_calib]]
    quantize_int8(fp32_path, int8_path, calib_arrays)
    del calib_arrays

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess_fp32 = ort.InferenceSession(fp32_path, sess_options=so, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, sess_options=so, providers=["CPUExecutionProvider"])

    def run_pt(x):
        with torch.no_grad():
            return backbone(x).numpy()

    def run_ort(sess):
        return lambda x: sess.run(None, {"noisy": x.numpy()})[0]

    paths = {
        "pytorch_fp32": run_pt,
        "onnx_fp32": run_ort(sess_fp32),
        "onnx_int8": run_ort(sess_int8),
    }

    first_noisy, _ = _load_pair(clean_dir, eval_names[0], pipeline, img_size)
    parity = max_abs_diff(backbone, fp32_path, first_noisy.unsqueeze(0))

    sq_sum = {k: 0.0 for k in paths}
    n_px = 0
    per_img_mse = {k: [] for k in paths}
    per_img_psnr = {k: [] for k in paths}
    for name in eval_names:
        noisy, clean = _load_pair(clean_dir, name, pipeline, img_size)
        x = noisy.unsqueeze(0)
        for k, fn in paths.items():
            pred = torch.from_numpy(fn(x)).clamp(0, 1)[0]
            se = (pred - clean) ** 2
            sq_sum[k] += float(se.sum())
            mse_i = float(se.mean())
            per_img_mse[k].append(mse_i)
            per_img_psnr[k].append(10 * np.log10(1.0 / mse_i))
        n_px += clean.numel()
    np.savez(out / "per_image_mse.npz", **{k: np.array(v) for k, v in per_img_mse.items()})

    global_psnr = {k: 10 * np.log10(1.0 / (sq_sum[k] / n_px)) for k in paths}

    rng = np.random.default_rng(0)
    n_img = len(eval_names)
    mse_int8 = np.array(per_img_mse["onnx_int8"])
    mse_fp32 = np.array(per_img_mse["pytorch_fp32"])
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, n_img, n_img)
        boot.append(10 * np.log10(1.0 / mse_int8[idx].mean()) - 10 * np.log10(1.0 / mse_fp32[idx].mean()))
    boot = np.array(boot)
    paired = {
        "int8_minus_fp32_global_psnr_db": round(float(global_psnr["onnx_int8"] - global_psnr["pytorch_fp32"]), 4),
        "bootstrap_ci95_db": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "bootstrap_resamples": 2000,
    }

    lat_noisy, _ = _load_pair(clean_dir, eval_names[0], pipeline, img_size)
    lat_x = lat_noisy.unsqueeze(0)
    latency = {k: _latency_ms(fn, lat_x) for k, fn in paths.items()}
    sizes = {"onnx_fp32_bytes": os.path.getsize(fp32_path), "onnx_int8_bytes": os.path.getsize(int8_path)}

    report = {
        "checkpoint": checkpoint,
        "base_channels": base_channels,
        "n_calib": n_calib,
        "n_eval": len(eval_names),
        "img_size": img_size,
        "psnr_global_db": {k: round(float(v), 3) for k, v in global_psnr.items()},
        "psnr_per_image_mean_db": {k: round(float(np.mean(v)), 3) for k, v in per_img_psnr.items()},
        "int8_vs_fp32": paired,
        "latency_ms_per_image": {k: round(v, 1) for k, v in latency.items()},
        "file_sizes": sizes,
        "onnx_fp32_max_abs_diff_vs_pytorch": parity,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--calib", type=int, default=500, help="training crops used for int8 calibration")
    p.add_argument("--eval", type=int, default=0, help="validation crops used for the report (0 = full split)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--img-size", type=int, default=512)
    a = p.parse_args()
    report = export_run(a.checkpoint, a.clean_dir, a.out_dir, a.calib, a.eval,
                        a.seed, a.test_fraction, a.img_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
