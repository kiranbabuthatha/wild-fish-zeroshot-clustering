"""Build standardized CSV manifests for open wild fish and SCHUFI datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fish_clustering.config import load_config
from fish_clustering.data import apply_sampling_filters, build_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path overriding data.manifest_path from config.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip configured datasets whose local directory is not present yet.",
    )
    parser.add_argument(
        "--apply-sampling",
        action="store_true",
        help="Apply max_crops_per_video and max_images_per_species caps before saving.",
    )
    return parser.parse_args()


def main() -> None:
    """Build manifest CSV from configured local datasets."""
    args = parse_args()
    cfg = load_config(args.config)
    output_path = args.output or cfg.data.manifest_path

    df = build_manifest(
        cfg.data.raw_root,
        cfg.data.datasets,
        skip_missing=args.skip_missing,
    )

    if args.apply_sampling and not df.empty:
        seed = cfg.experiment.seeds[0]
        df = apply_sampling_filters(
            df,
            max_crops_per_video=cfg.data.max_crops_per_video,
            max_images_per_species=cfg.data.max_images_per_species,
            seed=seed,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    mask_count = int((df["mask_path"].fillna("").astype(str).str.strip() != "").sum())
    species_count = int(df["is_species_level"].sum()) if not df.empty else 0
    LOGGER.info(
        "Wrote %d rows (%d with masks, %d species-level) to %s",
        len(df),
        mask_count,
        species_count,
        output_path,
    )


if __name__ == "__main__":
    main()
