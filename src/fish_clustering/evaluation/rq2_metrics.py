"""High-dimensional RQ2 mixing, cross-dataset matching, and OOD scoring."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _is_valid_species(val: object) -> bool:
    if val is None or pd.isna(val):
        return False
    return str(val).strip() != ""


def compute_dataset_mixing_entropy(
    embeddings: np.ndarray,
    datasets: Sequence[str],
    *,
    k_neighbors: int = 15,
    metric: str = "cosine",
) -> dict[str, Any]:
    """Measure normalized k-NN dataset entropy and same-dataset neighbor rate in high-D."""
    arr = np.asarray(embeddings, dtype=np.float32)
    ds = np.asarray([str(d) for d in datasets])
    if arr.shape[0] != ds.shape[0]:
        raise ValueError(
            f"Length mismatch: embeddings={arr.shape[0]} vs datasets={ds.shape[0]}."
        )
    if arr.shape[0] < 2:
        raise ValueError("At least 2 samples are required to compute mixing entropy.")

    unique_datasets = np.unique(ds)
    n_datasets = len(unique_datasets)
    k = min(int(k_neighbors), arr.shape[0] - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(arr)
    indices = nn.kneighbors(return_distance=False)

    entropies = np.zeros(arr.shape[0], dtype=np.float32)
    same_rates = np.zeros(arr.shape[0], dtype=np.float32)

    for i in range(arr.shape[0]):
        nbrs = [int(j) for j in indices[i] if j != i][:k]
        nbr_ds = ds[nbrs]
        same_rates[i] = float(np.mean(nbr_ds == ds[i]))

        if n_datasets <= 1:
            entropies[i] = 0.0
            continue

        _, counts = np.unique(nbr_ds, return_counts=True)
        probs = counts / float(len(nbrs))
        entropy = -float(np.sum(probs * np.log(probs)))
        entropies[i] = float(entropy / np.log(n_datasets))

    return {
        "mean_normalized_entropy": float(np.mean(entropies)),
        "mean_same_dataset_rate": float(np.mean(same_rates)),
        "n_datasets": int(n_datasets),
        "k_neighbors": int(k),
        "per_sample_entropy": entropies,
        "per_sample_same_dataset_rate": same_rates,
    }


def compute_cross_dataset_species_match(
    embeddings: np.ndarray,
    datasets: Sequence[str],
    species: Sequence[str],
    *,
    k_neighbors: int = 15,
    metric: str = "cosine",
) -> dict[str, float]:
    """Evaluate how often shared species match the same species in other datasets."""
    arr = np.asarray(embeddings, dtype=np.float32)
    ds = np.asarray([str(d) for d in datasets])
    sp = np.asarray([str(s).strip() if _is_valid_species(s) else "" for s in species])
    if arr.shape[0] != ds.shape[0] or arr.shape[0] != sp.shape[0]:
        raise ValueError("embeddings, datasets, and species must share the same length.")

    valid_mask = sp != ""
    if not np.any(valid_mask):
        return {
            "n_shared_species": 0,
            "n_evaluated_samples": 0,
            "top1_cross_dataset_accuracy": float("nan"),
            "knn_cross_dataset_precision": float("nan"),
        }

    df_sp = pd.DataFrame({"dataset": ds[valid_mask], "species": sp[valid_mask]})
    ds_counts = df_sp.groupby("species")["dataset"].nunique()
    shared_species = set(ds_counts[ds_counts >= 2].index.tolist())
    if not shared_species:
        return {
            "n_shared_species": 0,
            "n_evaluated_samples": 0,
            "top1_cross_dataset_accuracy": float("nan"),
            "knn_cross_dataset_precision": float("nan"),
        }

    shared_mask = np.isin(sp, list(shared_species))
    top1_hits: list[float] = []
    knn_precisions: list[float] = []

    for query_ds in np.unique(ds[shared_mask]):
        query_idx = np.where((ds == query_ds) & shared_mask)[0]
        ref_idx = np.where((ds != query_ds) & valid_mask)[0]
        if len(query_idx) == 0 or len(ref_idx) == 0:
            continue

        k = min(int(k_neighbors), len(ref_idx))
        nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(arr[ref_idx])
        nbr_local = nn.kneighbors(arr[query_idx], return_distance=False)

        for local_i, q_i in enumerate(query_idx):
            q_species = sp[q_i]
            matched_species = sp[ref_idx[nbr_local[local_i]]]
            top1_hits.append(float(matched_species[0] == q_species))
            knn_precisions.append(float(np.mean(matched_species == q_species)))

    if not top1_hits:
        return {
            "n_shared_species": len(shared_species),
            "n_evaluated_samples": 0,
            "top1_cross_dataset_accuracy": float("nan"),
            "knn_cross_dataset_precision": float("nan"),
        }

    return {
        "n_shared_species": int(len(shared_species)),
        "n_evaluated_samples": int(len(top1_hits)),
        "top1_cross_dataset_accuracy": float(np.mean(top1_hits)),
        "knn_cross_dataset_precision": float(np.mean(knn_precisions)),
    }


def compute_knn_ood_scores(
    reference_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    k_neighbors: int = 15,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute mean k-NN distance from each query embedding to a reference pool."""
    ref = np.asarray(reference_embeddings, dtype=np.float32)
    qry = np.asarray(query_embeddings, dtype=np.float32)
    if ref.ndim != 2 or qry.ndim != 2:
        raise ValueError("reference_embeddings and query_embeddings must be 2-D arrays.")
    if ref.shape[0] == 0:
        raise ValueError("reference_embeddings must contain at least one sample.")
    if qry.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    k = min(int(k_neighbors), ref.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(ref)
    dists, _ = nn.kneighbors(qry, return_distance=True)
    return np.mean(dists, axis=1).astype(np.float32)


def compute_self_knn_ood_scores(
    embeddings: np.ndarray,
    *,
    k_neighbors: int = 15,
    metric: str = "cosine",
) -> np.ndarray:
    """Compute leave-one-out mean k-NN distance within an embedding set."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("At least 2 embeddings are required for self k-NN OOD scoring.")

    k = min(int(k_neighbors), arr.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(arr)
    dists, indices = nn.kneighbors(return_distance=True)

    scores = np.zeros(arr.shape[0], dtype=np.float32)
    for i in range(arr.shape[0]):
        kept = [float(d) for idx, d in zip(indices[i], dists[i]) if idx != i][:k]
        scores[i] = float(np.mean(kept))
    return scores
