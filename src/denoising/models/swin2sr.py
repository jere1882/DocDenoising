"""Swin2SR backbone wrapper for artifact removal (upscale=1, fine-tuned from pretrained)."""

import torch
import torch.nn as nn
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution

_CHECKPOINTS = {
    "compressed": "caidas/swin2SR-compressed-sr-x4-48",   # 12M params, best for JPEG
    "lightweight": "caidas/swin2SR-lightweight-x2-64",    # 1M params, fits on MPS
}


class Swin2SRDenoiser(nn.Module):
    """Swin2SR fine-tuned for JPEG artifact removal.

    Initialises with upscale=1 (same-resolution output) and loads the transformer
    backbone weights from the pretrained compressed-SR checkpoint. Only the final
    output convolution is randomly initialised — everything else is pretrained.
    """

    def __init__(
        self,
        checkpoint: str = "lightweight",
        embed_dim: int = 60,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        resi_connection: str = "1conv",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        depths = depths or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]

        cfg = Swin2SRConfig(
            upscale=1,
            upsampler="",
            image_size=64,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            resi_connection=resi_connection,
        )
        self.model = Swin2SRForImageSuperResolution(cfg)

        if pretrained:
            hf_id = _CHECKPOINTS[checkpoint]
            src = Swin2SRForImageSuperResolution.from_pretrained(hf_id)
            backbone_weights = {
                k: v for k, v in src.state_dict().items() if k.startswith("swin2sr.")
            }
            missing, _ = self.model.load_state_dict(backbone_weights, strict=False)
            assert all("final_convolution" in k for k in missing), \
                f"Unexpected missing keys: {missing}"
            del src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).reconstruction.clamp(0, 1)
