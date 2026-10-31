"""Saliency maps over the spectrogram [aspect 4] (Phase 2).

Grad-CAM for the CNN and integrated gradients for the Transformer, so the web app can show
which regions of the spectrogram pushed the score towards "AI".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch


def saliency_map(
    model: torch.nn.Module,
    spectrogram: np.ndarray,
    method: Literal["gradcam", "integrated_gradients"] = "gradcam",
    target_class: int = 1,
) -> np.ndarray:
    """Return a map with the spectrogram's shape, scaled to [0, 1].

    Grad-CAM (Selvaraju et al. 2017) for the CNN from ``models.deep.build_cnn``: the last
    convolution block's activations, weighted by the average gradient of the target class,
    summed over channels, kept where positive and stretched back to the input size.
    Integrated gradients is not implemented yet (it is meant for the Transformer).
    """
    import torch
    import torch.nn.functional as F

    if method != "gradcam":
        raise NotImplementedError("integrated gradients arrives with the Transformer")
    layer = model[4][5]  # ReLU before the last block's pooling
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
        output.retain_grad()
        captured["act"] = output

    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        device = next(model.parameters()).device
        x = torch.as_tensor(np.asarray(spectrogram, dtype=np.float32), device=device)[None, None]
        model.zero_grad()
        model(x)[0, target_class].backward()
    finally:
        handle.remove()
    act = captured["act"]
    weights = act.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam
