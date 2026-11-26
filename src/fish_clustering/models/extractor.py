"""Batched feature extraction and L2 normalization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fish_clustering.data.preprocessing import load_image_sample
from fish_clustering.models.backbones import BaseFeatureExtractor


def l2_normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embedding rows cleanly, guarding against zero-norm rows."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        return arr.copy()

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, eps)
    return (arr / safe_norms).astype(np.float32, copy=False)


def extract_embeddings(
    manifest: pd.DataFrame,
    extractor: BaseFeatureExtractor,
    *,
    raw_root: Path | None = None,
    batch_size: int = 32,
    l2_normalize: bool = True,
    use_foreground_mask: bool = False,
    equalize_size: int | None = None,
    crop_from_bbox: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Run batched feature extraction across a manifest DataFrame."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if manifest.empty:
        return np.empty((0, 0), dtype=np.float32), []

    records = manifest.to_dict(orient="records")
    image_ids = [str(rec["image_id"]) for rec in records]
    batches = range(0, len(records), batch_size)
    if show_progress:
        from tqdm import tqdm

        batches = tqdm(batches, desc=f"Extracting {extractor.backbone_name}", unit="batch")

    encoded_chunks: list[np.ndarray] = []
    for start in batches:
        chunk_records = records[start : start + batch_size]
        pil_batch: list[Image.Image] = [
            load_image_sample(
                rec,
                raw_root=raw_root,
                use_foreground_mask=use_foreground_mask,
                equalize_size=equalize_size,
                crop_from_bbox=crop_from_bbox,
            )
            for rec in chunk_records
        ]
        feats = np.asarray(extractor.encode_images(pil_batch), dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != len(pil_batch):
            raise ValueError(
                f"Extractor returned unexpected shape {feats.shape} for batch size {len(pil_batch)}."
            )
        encoded_chunks.append(feats)

    all_embeddings = np.concatenate(encoded_chunks, axis=0).astype(np.float32, copy=False)
    if l2_normalize:
        all_embeddings = l2_normalize_embeddings(all_embeddings)

    return all_embeddings, image_ids
