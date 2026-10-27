"""Deep models on spectrograms [aspect 2: deep learning] (Phase 2).

* A compact CNN on log-mel or CQT inputs: a fast baseline that can run on CPU in the demo.
* A spectrogram Transformer (patch embeddings + encoder).
* SpecTTTra from SONICS (Rahman et al., ICLR 2025) as the reference baseline. Reuse the
  authors' MIT-licensed code and checkpoints (github.com/awsaf49/sonics) rather than
  re-implementing it.

torch is only imported for type checking here, so the rest of the package works without it.
Train on Colab; save checkpoints under cfg.paths.checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


def build_cnn(n_mels: int = 128, n_classes: int = 2) -> torch.nn.Module:
    """Small convolutional network over ``(1, n_mels, frames)`` inputs."""
    raise NotImplementedError("Phase 2")


def build_transformer(
    n_mels: int = 128, patch_size: tuple[int, int] = (16, 16), depth: int = 6, n_classes: int = 2
) -> torch.nn.Module:
    """Spectrogram Transformer: patch embedding, positional encoding, encoder, class head."""
    raise NotImplementedError("Phase 2")


def load_spectttra(model_id: str = "awsaf49/sonics-spectttra-alpha-5s") -> torch.nn.Module:
    """Load a pretrained SpecTTTra checkpoint (alpha/beta/gamma, each for 5 s or 120 s)."""
    raise NotImplementedError("Phase 2")


def train_deep(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 3e-4,
    device: str = "cuda",
) -> dict[str, list[float]]:
    """Train with early stopping on validation loss; return the loss/metric history."""
    raise NotImplementedError("Phase 2")
