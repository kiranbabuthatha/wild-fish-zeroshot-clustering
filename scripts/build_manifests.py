"""Build standardized CSV manifests for open wild fish and SCHUFI datasets."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest generation."""
    parser = argparse.ArgumentParser(
        description="Build unified image manifests from local wild fish datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to experiment YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for manifest generation."""
    args = parse_args()
    raise NotImplementedError(
        f"Manifest builder will be wired to {args.config} in the next update."
    )


if __name__ == "__main__":
    main()
