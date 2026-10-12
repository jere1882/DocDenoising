"""Checkpoint -> backbone -> ONNX fp32, with a parity check against PyTorch."""

import numpy as np
import onnxruntime as ort
import torch

from denoising.models.lit_module import DenoisingLitModule

OPSET = 17


def load_backbone(checkpoint: str) -> tuple[torch.nn.Module, dict]:
    lit = DenoisingLitModule.load_from_checkpoint(checkpoint, map_location="cpu")
    return lit.model.eval(), dict(lit.hparams)


def export_onnx(backbone: torch.nn.Module, out_path: str, img_size: int = 512) -> None:
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        backbone,
        dummy,
        out_path,
        input_names=["noisy"],
        output_names=["denoised"],
        dynamic_axes={
            "noisy": {0: "batch", 2: "height", 3: "width"},
            "denoised": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=OPSET,
        dynamo=False,
    )


def max_abs_diff(backbone: torch.nn.Module, onnx_path: str, x: torch.Tensor) -> float:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = backbone(x).numpy()
    out = sess.run(None, {"noisy": x.numpy()})[0]
    return float(np.abs(ref - out).max())
