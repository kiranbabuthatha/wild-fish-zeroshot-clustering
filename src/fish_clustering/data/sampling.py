"""Sampling caps and taxonomic filters for wild fish image manifests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _is_missing(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _resolve_class_key(dataset: object, species: object, label_raw: object, idx: int) -> str:
    if not _is_missing(species):
        return f"species::{str(species).strip()}"
    if not _is_missing(label_raw):
        return f"{dataset}::label::{str(label_raw).strip()}"
    return f"row::{idx}"


def _sample_by_group(
    df: pd.DataFrame,
    group_keys: pd.Series,
    max_per_group: int,
    seed: int,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    work["_group_key"] = group_keys.values
    work = work.sort_values(by=["_group_key", "dataset", "image_id"], kind="mergesort")

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    for _, group in work.groupby("_group_key", sort=True):
        idx = group.index.to_numpy()
        if len(idx) > max_per_group:
            chosen = rng.choice(len(idx), size=max_per_group, replace=False)
            chosen.sort()
            idx = idx[chosen]
        selected_indices.extend(idx.tolist())

    result = df.loc[selected_indices].reset_index(drop=True)
    return result


def cap_crops_per_video(
    df: pd.DataFrame,
    max_per_video: int | None,
    seed: int = 42,
) -> pd.DataFrame:
    """Cap crops per (dataset, video_id, class) so multi-species clips stay balanced."""
    if max_per_video is None:
        return df.copy()
    if max_per_video <= 0:
        raise ValueError("max_per_video must be positive when provided.")
    if df.empty:
        return df.copy()

    keys: list[str] = []
    for idx, (ds, vid, sp, raw_lbl) in enumerate(
        zip(df["dataset"], df["video_id"], df["species"], df["label_raw"])
    ):
        if _is_missing(vid):
            keys.append(f"row::{idx}")
        else:
            cls_key = _resolve_class_key(ds, sp, raw_lbl, idx)
            keys.append(f"{ds}::vid::{vid}::{cls_key}")

    return _sample_by_group(df, pd.Series(keys, index=df.index), max_per_video, seed)


def cap_images_per_species(
    df: pd.DataFrame,
    max_per_species: int | None,
    seed: int = 42,
) -> pd.DataFrame:
    """Cap images per species label to reduce long-tail class imbalance."""
    if max_per_species is None:
        return df.copy()
    if max_per_species <= 0:
        raise ValueError("max_per_species must be positive when provided.")
    if df.empty:
        return df.copy()

    keys = [
        _resolve_class_key(ds, sp, raw_lbl, idx)
        for idx, (ds, sp, raw_lbl) in enumerate(
            zip(df["dataset"], df["species"], df["label_raw"])
        )
    ]
    return _sample_by_group(df, pd.Series(keys, index=df.index), max_per_species, seed)


def filter_min_images_per_species(
    df: pd.DataFrame,
    min_per_species: int | None,
) -> pd.DataFrame:
    """Drop rare classes having fewer than min_per_species rows."""
    if min_per_species is None:
        return df.copy()
    if min_per_species <= 0:
        raise ValueError("min_per_species must be positive when provided.")
    if df.empty:
        return df.copy()

    keys = pd.Series(
        [
            _resolve_class_key(ds, sp, raw_lbl, idx)
            for idx, (ds, sp, raw_lbl) in enumerate(
                zip(df["dataset"], df["species"], df["label_raw"])
            )
        ],
        index=df.index,
    )
    counts = keys.value_counts()
    keep_keys = set(counts[counts >= min_per_species].index)
    return df.loc[keys.isin(keep_keys)].reset_index(drop=True)


def filter_top_n_species(
    df: pd.DataFrame,
    top_n: int | None,
) -> pd.DataFrame:
    """Retain rows belonging to the top_n most frequent species/classes."""
    if top_n is None:
        return df.copy()
    if top_n <= 0:
        raise ValueError("top_n must be positive when provided.")
    if df.empty:
        return df.copy()

    keys = pd.Series(
        [
            _resolve_class_key(ds, sp, raw_lbl, idx)
            for idx, (ds, sp, raw_lbl) in enumerate(
                zip(df["dataset"], df["species"], df["label_raw"])
            )
        ],
        index=df.index,
    )
    counts = keys.value_counts()
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    keep_keys = {k for k, _ in ranked[:top_n]}
    return df.loc[keys.isin(keep_keys)].reset_index(drop=True)


def filter_species_level(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only rows annotated at Latin binomial species level."""
    if df.empty:
        return df.copy()
    mask = df["is_species_level"].apply(
        lambda val: val is True or (isinstance(val, str) and val.strip().lower() == "true")
    )
    return df.loc[mask].reset_index(drop=True)


def filter_shared_species(df: pd.DataFrame, min_datasets: int = 2) -> pd.DataFrame:
    """Retain only species that appear across at least min_datasets datasets."""
    if min_datasets < 1:
        raise ValueError("min_datasets must be at least 1.")
    if df.empty:
        return df.copy()

    valid_species = df["species"].apply(lambda val: not _is_missing(val))
    subset = df.loc[valid_species]
    counts = subset.groupby("species")["dataset"].nunique()
    shared = set(counts[counts >= min_datasets].index)
    mask = df["species"].isin(shared)
    return df.loc[mask].reset_index(drop=True)


def apply_sampling_filters(
    df: pd.DataFrame,
    *,
    species_level_only: bool = False,
    max_crops_per_video: int | None = None,
    min_images_per_species: int | None = None,
    top_n_species: int | None = None,
    max_images_per_species: int | None = None,
    shared_species_min_datasets: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply taxonomic, duplicate-video, class-support, and cap filters in order."""
    filtered = df.copy()
    if species_level_only:
        filtered = filter_species_level(filtered)
    filtered = cap_crops_per_video(filtered, max_crops_per_video, seed=seed)
    filtered = filter_min_images_per_species(filtered, min_images_per_species)
    filtered = filter_top_n_species(filtered, top_n_species)
    filtered = cap_images_per_species(filtered, max_images_per_species, seed=seed)
    if shared_species_min_datasets is not None:
        filtered = filter_shared_species(filtered, min_datasets=shared_species_min_datasets)
    return filtered.reset_index(drop=True)
