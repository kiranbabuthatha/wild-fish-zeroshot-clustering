"""Projection quality metrics: Trustworthiness, Continuity, and local distortion."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _clamp_k(n_samples: int, n_neighbors: int) -> int:
    if n_samples < 5:
        raise ValueError(
            f"At least 5 samples are required to compute k-NN projection quality, got {n_samples}."
        )
    max_k = (n_samples - 1) // 2
    while max_k >= 1 and (2 * n_samples - 3 * max_k - 1) <= 0:
        max_k -= 1
    k = min(int(n_neighbors), max_k)
    if k < 1:
        raise ValueError(f"Could not determine a valid neighbor count for n_samples={n_samples}.")
    return k


def compute_trustworthiness(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> float:
    """Compute scikit-learn Trustworthiness between high-D and low-D embeddings."""
    hi = np.asarray(high_dim, dtype=np.float32)
    lo = np.asarray(low_dim, dtype=np.float32)
    if hi.shape[0] != lo.shape[0]:
        raise ValueError(
            f"Sample count mismatch: high_dim={hi.shape[0]} vs low_dim={lo.shape[0]}."
        )
    k = _clamp_k(hi.shape[0], n_neighbors)
    return float(sklearn_trustworthiness(hi, lo, n_neighbors=k, metric=metric))


def compute_continuity(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    n_neighbors: int = 15,
    high_dim_metric: str = "cosine",
    low_dim_metric: str = "euclidean",
    chunk_size: int = 512,
) -> float:
    """Compute Venna and Kaski (2006) Continuity using memory-safe chunked distance blocks."""
    hi = np.asarray(high_dim, dtype=np.float32)
    lo = np.asarray(low_dim, dtype=np.float32)
    if hi.shape[0] != lo.shape[0]:
        raise ValueError(
            f"Sample count mismatch: high_dim={hi.shape[0]} vs low_dim={lo.shape[0]}."
        )

    n_samples = hi.shape[0]
    k = _clamp_k(n_samples, n_neighbors)
    penalty = 0.0

    for start in range(0, n_samples, chunk_size):
        end = min(n_samples, start + chunk_size)
        block_size = end - start

        d_hi = pairwise_distances(hi[start:end], hi, metric=high_dim_metric)
        d_lo = pairwise_distances(lo[start:end], lo, metric=low_dim_metric)

        row_idx = np.arange(block_size)
        col_idx = np.arange(start, end)
        d_hi[row_idx, col_idx] = np.inf
        d_lo[row_idx, col_idx] = np.inf

        u_k = np.argpartition(d_hi, kth=k - 1, axis=1)[:, :k]
        ranks_lo = np.argsort(np.argsort(d_lo, axis=1, kind="mergesort"), axis=1, kind="mergesort") + 1
        neighbor_ranks = ranks_lo[row_idx[:, None], u_k]
        excess = np.maximum(neighbor_ranks - k, 0)
        penalty += float(np.sum(excess))

    normalizer = 2.0 / (n_samples * k * (2 * n_samples - 3 * k - 1))
    continuity = 1.0 - normalizer * penalty
    return float(np.clip(continuity, 0.0, 1.0))


def compute_pointwise_distortion(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    n_neighbors: int = 15,
    high_dim_metric: str = "cosine",
    low_dim_metric: str = "euclidean",
) -> np.ndarray:
    """Compute per-sample neighborhood Jaccard distortion in [0, 1]."""
    hi = np.asarray(high_dim, dtype=np.float32)
    lo = np.asarray(low_dim, dtype=np.float32)
    if hi.shape[0] != lo.shape[0]:
        raise ValueError(
            f"Sample count mismatch: high_dim={hi.shape[0]} vs low_dim={lo.shape[0]}."
        )

    n_samples = hi.shape[0]
    k = _clamp_k(n_samples, n_neighbors)

    nn_hi = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric=high_dim_metric).fit(hi)
    nn_lo = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric=low_dim_metric).fit(lo)

    idx_hi = nn_hi.kneighbors(return_distance=False)
    idx_lo = nn_lo.kneighbors(return_distance=False)

    distortion = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        u = {int(j) for j in idx_hi[i] if j != i}
        v = {int(j) for j in idx_lo[i] if j != i}
        if len(u) > k:
            u = set(list(u)[:k])
        if len(v) > k:
            v = set(list(v)[:k])
        union_size = len(u | v)
        if union_size == 0:
            distortion[i] = 0.0
        else:
            distortion[i] = 1.0 - (len(u & v) / union_size)
    return distortion


def compute_projection_quality(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    *,
    n_neighbors: int = 15,
    high_dim_metric: str = "cosine",
) -> dict[str, Any]:
    """Compute Trustworthiness, Continuity, and per-point neighborhood distortion."""
    k = _clamp_k(np.asarray(high_dim).shape[0], n_neighbors)
    trust = compute_trustworthiness(high_dim, low_dim, n_neighbors=k, metric=high_dim_metric)
    cont = compute_continuity(
        high_dim,
        low_dim,
        n_neighbors=k,
        high_dim_metric=high_dim_metric,
    )
    distortion = compute_pointwise_distortion(
        high_dim,
        low_dim,
        n_neighbors=k,
        high_dim_metric=high_dim_metric,
    )
    return {
        "trustworthiness": trust,
        "continuity": cont,
        "mean_neighborhood_jaccard": float(1.0 - np.mean(distortion)),
        "k_neighbors": k,
        "per_point_distortion": distortion,
    }
