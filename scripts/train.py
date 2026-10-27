"""Train one model family (Phases 1-2). Runs locally or in a Colab notebook cell.

Usage:
    python scripts/train.py --model random_forest
    python scripts/train.py --model cnn --epochs 20
"""

import argparse

MODELS = ["svm", "random_forest", "gradient_boosting", "cnn", "transformer", "probe_mert"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VeraSonus model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--epochs", type=int, default=20, help="deep models only")
    parser.parse_args()
    raise NotImplementedError(
        "Phase 1-2: load features for the train/val splits, then call "
        "verasonus.models.classical.train_classical or verasonus.models.deep.train_deep."
    )


if __name__ == "__main__":
    main()
