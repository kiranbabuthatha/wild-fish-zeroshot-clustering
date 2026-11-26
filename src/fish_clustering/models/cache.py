"""Deterministic disk caching for extracted image embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fish_clustering.models.backbones import BaseFeatureExtractor
from fish_clustering.models.extractor import extract_embeddings


@dataclass
class CachedEmbeddings:
    """Container returned when loading or computing cached embeddings."""

    embeddings: np.ndarray
    image_ids: list[str]
    metadata: dict[str, Any]
    cache_path: Path | None = None


def build_cache_key(
    *,
    backbone: str,
    pooling: str,
    image_size: int,
    l2_normalize: bool,
    use_foreground_mask: bool,
    equalize_resolution: int | None,
    image_ids: list[str],
) -> str:
    """Create a collision-free cache filename stem from extraction config and ordered image IDs."""
    payload = {
        "backbone": backbone,
        "pooling": pooling,
        "image_size": int(image_size),
        "l2_normalize": bool(l2_normalize),
        "use_foreground_mask": bool(use_foreground_mask),
        "equalize_resolution": equalize_resolution,
        "image_ids": image_ids,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    safe_backbone = backbone.replace("/", "_").replace(":", "_")
    return f"{safe_backbone}_{pooling}_{image_size}_{digest}"


def save_embeddings(
    cache_path: str | Path,
    embeddings: np.ndarray,
    image_ids: list[str],
    metadata: dict[str, Any],
) -> Path:
    """Save embeddings, aligned image IDs, and metadata as a compressed NPZ file."""
    path = Path(cache_path)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(image_ids):
        raise ValueError(
            f"Embeddings row count ({arr.shape[0]}) must match image_ids length ({len(image_ids)})."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=arr,
        image_ids=np.asarray(image_ids, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_embeddings(cache_path: str | Path) -> CachedEmbeddings:
    """Load embeddings, image IDs, and metadata from a cached NPZ archive."""
    path = Path(cache_path)
    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache file not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        image_ids = [str(item) for item in archive["image_ids"].tolist()]
        metadata_raw = str(archive["metadata_json"].item())

    if embeddings.shape[0] != len(image_ids):
        raise ValueError(
            f"Corrupt cache {path}: {embeddings.shape[0]} rows vs {len(image_ids)} image IDs."
        )

    return CachedEmbeddings(
        embeddings=embeddings,
        image_ids=image_ids,
        metadata=json.loads(metadata_raw),
        cache_path=path,
    )


def extract_or_load_embeddings(
    manifest: pd.DataFrame,
    extractor: BaseFeatureExtractor,
    *,
    cache_dir: str | Path,
    raw_root: Path | None = None,
    batch_size: int = 32,
    l2_normalize: bool = True,
    use_foreground_mask: bool = False,
    equalize_size: int | None = None,
    crop_from_bbox: bool = True,
    force_recompute: bool = False,
    show_progress: bool = False,
) -> CachedEmbeddings:
    """Return cached embeddings if present and aligned, otherwise compute and cache them."""
    if manifest.empty:
        return CachedEmbeddings(
            embeddings=np.empty((0, 0), dtype=np.float32),
            image_ids=[],
            metadata={"backbone": extractor.backbone_name, "pooling": extractor.pooling},
            cache_path=None,
        )

    ordered_ids = [str(item) for item in manifest["image_id"].tolist()]
    key = build_cache_key(
        backbone=extractor.backbone_name,
        pooling=extractor.pooling,
        image_size=extractor.image_size,
        l2_normalize=l2_normalize,
        use_foreground_mask=use_foreground_mask,
        equalize_resolution=equalize_size,
        image_ids=ordered_ids,
    )
    cache_path = Path(cache_dir) / f"{key}.npz"

    if cache_path.is_file() and not force_recompute:
        cached = load_embeddings(cache_path)
        if cached.image_ids == ordered_ids:
            return cached

    embeddings, extracted_ids = extract_embeddings(
        manifest,
        extractor,
        raw_root=raw_root,
        batch_size=batch_size,
        l2_normalize=l2_normalize,
        use_foreground_mask=use_foreground_mask,
        equalize_size=equalize_size,
        crop_from_bbox=crop_from_bbox,
        show_progress=show_progress,
    )
    metadata = {
        "backbone": extractor.backbone_name,
        "pooling": extractor.pooling,
        "image_size": extractor.image_size,
        "l2_normalize": l2_normalize,
        "use_foreground_mask": use_foreground_mask,
        "equalize_resolution": equalize_size,
        "crop_from_bbox": crop_from_bbox,
        "num_samples": len(extracted_ids),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
    }
    save_embeddings(cache_path, embeddings, extracted_ids, metadata)
    return CachedEmbeddings(
        embeddings=embeddings,
        image_ids=extracted_ids,
        metadata=metadata,
        cache_path=cache_path,
    )
