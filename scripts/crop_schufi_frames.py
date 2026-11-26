"""Detect and crop fish instances from unboxed SCHUFI and HODOR frames."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for automated frame cropping."""
    parser = argparse.ArgumentParser(
        description="Crop fish bounding boxes from raw underwater frames."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to experiment YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for automated fish detection and cropping."""
    args = parse_args()
    raise NotImplementedError(
        f"Frame cropper will be wired to {args.config} once detector choice is confirmed."
    )


if __name__ == "__main__":
    main()
