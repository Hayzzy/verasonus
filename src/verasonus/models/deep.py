"""Deep models on spectrograms [aspect 2: deep learning] (Phase 2).

* A compact CNN on log-mel or CQT inputs: a fast baseline that can run on CPU in the demo.
* A spectrogram Transformer (patch embeddings + encoder).
* SpecTTTra from SONICS (Rahman et al., ICLR 2025) as the reference baseline. Reuse the
  authors' MIT-licensed code and checkpoints (github.com/awsaf49/sonics) rather than
  re-implementing it.

torch is imported inside the functions, so the rest of the package works without it.
Train on the local GPU (8 GB VRAM: use mixed precision and modest batch sizes) and save
checkpoints under cfg.paths.checkpoints so runs can resume.

Inputs are the cached spectrograms (features.cache.build_spectrogram_cache): an array
``(windows, n_mels, frames)``, in memory or memory-mapped from disk when it is larger than
RAM, with labels 1 = AI. Training and prediction take row indices into that array, so no
split is ever copied.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    import torch


def build_cnn(n_classes: int = 2) -> torch.nn.Module:
    """Small convolutional network over ``(1, n_mels, frames)`` inputs.

    Four blocks of two 3x3 convolutions (each with batch norm and ReLU) and 2x2 max pooling,
    widening from 32 to 256 channels, then an average over time and frequency and a linear
    head. About 1.2 M parameters, so it also runs on CPU. Any spectrogram size works.
    """
    from torch import nn

    def block(c_in: int, c_out: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        nn.BatchNorm2d(1),  # scales the dB input with statistics learnt during training
        block(1, 32),
        block(32, 64),
        block(64, 128),
        block(128, 256),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(256, n_classes),
    )


def build_transformer(
    n_mels: int = 128, patch_size: tuple[int, int] = (16, 16), depth: int = 6, n_classes: int = 2
) -> torch.nn.Module:
    """Spectrogram Transformer: patch embedding, positional encoding, encoder, class head."""
    raise NotImplementedError("Phase 2")


SPECTTTRA_RATE = 16_000


def load_spectttra(model_id: str = "awsaf49/sonics-spectttra-alpha-120s") -> torch.nn.Module:
    """Load a pretrained SpecTTTra checkpoint (alpha/beta/gamma, each for 5 s or 120 s).

    Uses the authors' MIT-licensed package (pip install git+https://github.com/awsaf49/sonics,
    see requirements/ml.txt). The weights download from Hugging Face on first use. The
    SONICS models were trained on Suno and Udio songs at 16 kHz, without our 7 kHz low-pass.
    """
    from sonics import HFAudioClassifier

    return HFAudioClassifier.from_pretrained(model_id).eval()


def spectttra_scores(
    model: torch.nn.Module,
    audio: np.ndarray,
    seconds: float = 120.0,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """AI probability for windows of ``seconds`` across 16 kHz mono ``audio``.

    Windows overlap by half. Audio shorter than one window is repeated to fill it rather than
    padded with silence: the SONICS models were trained on full songs and score long silent
    padding as AI. Each window is scaled to unit standard deviation, as in SONICS.
    Returns ``(start_seconds, probabilities)``.
    """
    import torch

    device = device or default_device()
    model.to(device)
    n = round(seconds * SPECTTTRA_RATE)
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("empty audio")
    if len(audio) < n:
        audio = np.tile(audio, -(-n // len(audio)))[:n]
    starts = list(range(0, len(audio) - n + 1, n // 2))
    if starts[-1] + n < len(audio):
        starts.append(len(audio) - n)  # cover the end of the track too
    batch = np.stack([audio[s : s + n] for s in starts])
    batch /= np.maximum(batch.std(axis=1, keepdims=True), 1e-6)
    with torch.inference_mode():
        logits = model(torch.from_numpy(batch).to(device))
    probs = torch.sigmoid(logits.float()).reshape(-1).cpu().numpy()
    return np.asarray(starts) / SPECTTTRA_RATE, probs


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def spec_augment(x: torch.Tensor, max_freq: int = 16, max_time: int = 40) -> torch.Tensor:
    """Hide one random band of mel bins and one random stretch of frames in each example.

    The hidden values are replaced by the example's mean (SpecAugment, Park et al. 2019), so
    the model cannot rely on any single frequency band or moment.
    """
    import torch

    batch, _, n_freq, n_time = x.shape

    def band(size: int, max_width: int) -> torch.Tensor:
        width = torch.randint(0, max_width + 1, (batch, 1), device=x.device)
        start = (torch.rand(batch, 1, device=x.device) * (size - width + 1)).long()
        pos = torch.arange(size, device=x.device)
        return (pos >= start) & (pos < start + width)

    hidden = band(n_freq, max_freq)[:, None, :, None] | band(n_time, max_time)[:, None, None, :]
    return torch.where(hidden, x.mean(dim=(2, 3), keepdim=True), x)


def _batch(x: np.ndarray, rows: np.ndarray, device: str) -> torch.Tensor:
    """Rows of the spectrogram array as a float tensor shaped ``(batch, 1, n_mels, frames)``.

    Rows are read in sorted order (fast on a memory-mapped array) and put back in order.
    """
    import torch

    order = np.argsort(rows)
    data = np.empty((len(rows), *x.shape[1:]), dtype=x.dtype)
    data[order] = x[rows[order]]
    return torch.from_numpy(data).to(device).float().unsqueeze(1)


def predict_deep(
    model: torch.nn.Module,
    x: np.ndarray,
    rows: np.ndarray,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """AI probability for each of ``x[rows]``."""
    import torch

    device = device or default_device()
    model.to(device).eval()
    scores = []
    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"),
    ):
        for start in range(0, len(rows), batch_size):
            logits = model(_batch(x, rows[start : start + batch_size], device))
            scores.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0)


def train_deep(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    *,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    patience: int = 4,
    augment: bool = True,
    balance: bool = False,
    device: str | None = None,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> dict[str, list[float]]:
    """Train on ``x[train_rows]`` and keep the epoch with the best val ROC-AUC.

    AdamW with a one-cycle learning-rate schedule, bfloat16 mixed precision on CUDA, and
    early stopping after ``patience`` epochs without improvement. ``balance`` weights the
    loss so real and AI windows count equally overall. The model is left holding the best
    epoch's weights. Returns the per-epoch training loss and val ROC-AUC.
    """
    import torch
    from torch import nn

    device = device or default_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = -(-len(train_rows) // batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=lr, total_steps=epochs * steps)
    weight = None
    if balance:
        counts = np.bincount(y[train_rows], minlength=2).astype(float)
        weight = torch.as_tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float)
        log(f"class weights real {weight[0]:.2f}, AI {weight[1]:.2f}")
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
    labels = torch.as_tensor(y, dtype=torch.long)

    history: dict[str, list[float]] = {"train_loss": [], "val_roc_auc": []}
    best_auc, best_state, waited = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.0
        order = rng.permutation(train_rows)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            xb = _batch(x, rows, device)
            if augment:
                xb = spec_augment(xb)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model(xb).float(), labels[rows].to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(rows)

        val_auc = float(roc_auc_score(y[val_rows], predict_deep(model, x, val_rows, device=device)))
        history["train_loss"].append(total / len(train_rows))
        history["val_roc_auc"].append(val_auc)
        log(
            f"epoch {epoch:>2}/{epochs}  train loss {total / len(train_rows):.4f}  "
            f"val ROC-AUC {val_auc:.4f}  ({time.perf_counter() - started:.0f} s)"
        )
        if val_auc > best_auc:
            best_auc, waited = val_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                log(f"no improvement for {patience} epochs: stopping")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
