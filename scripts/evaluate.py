"""Run an evaluation protocol and save figures and tables to reports/ (Phases 1-2).

Usage:
    python scripts/evaluate.py --protocol seen
    python scripts/evaluate.py --protocol unseen
    python scripts/evaluate.py --protocol robustness
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VeraSonus models.")
    parser.add_argument("--protocol", choices=["seen", "unseen", "robustness"], required=True)
    parser.parse_args()
    raise NotImplementedError(
        "Phase 1-2: score the test data, then call verasonus.evaluation.protocols and "
        "save outputs with verasonus.evaluation.reports."
    )


if __name__ == "__main__":
    main()
