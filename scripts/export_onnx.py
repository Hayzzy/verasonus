"""Export trained models to ONNX for fast CPU inference in the API (Phase 3).

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/cnn_best.pt --out checkpoints/cnn.onnx
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.parse_args()
    raise NotImplementedError(
        "Phase 3: torch.onnx.export the model, then check that ONNX Runtime output matches "
        "PyTorch and time a 30-second clip on CPU."
    )


if __name__ == "__main__":
    main()
