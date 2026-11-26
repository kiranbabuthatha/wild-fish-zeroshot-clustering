"""Run zero-shot feature extraction, DR, clustering, evaluation, and visual export."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for running the full thesis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the zero-shot fish clustering and cross-dataset pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke_test.yaml"),
        help="Path to experiment YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the end-to-end clustering and evaluation pipeline."""
    args = parse_args()
    raise NotImplementedError(
        f"Pipeline runner will be wired to {args.config} once core modules are ready."
    )


if __name__ == "__main__":
    main()
