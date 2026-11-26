"""Clustering wrappers for HDBSCAN, Ward agglomerative, k-means, and GMM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

SUPPORTED_CLUSTERERS = {"hdbscan", "ward", "kmeans", "gmm"}


@dataclass
class ClusteringResult:
    """Output container from a clustering run."""

    method: str
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    space: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)


def _validate_features(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D feature array, got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError(f"At least 2 samples are required for clustering, got {arr.shape[0]}.")
    return arr


def _summarize_labels(labels: np.ndarray) -> tuple[int, int]:
    int_labels = np.asarray(labels, dtype=np.int64)
    n_noise = int(np.sum(int_labels == -1))
    unique_clusters = {int(lbl) for lbl in int_labels.tolist() if lbl >= 0}
    return len(unique_clusters), n_noise


def cluster_hdbscan(
    features: np.ndarray,
    *,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
    metric: str = "euclidean",
    seed: int = 42,
    space: str = "high_dim",
) -> ClusteringResult:
    """Run density-based HDBSCAN clustering (with noise label -1)."""
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2.")

    arr = _validate_features(features)
    try:
        import hdbscan

        estimator = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        backend = "hdbscan"
    except ImportError:
        from sklearn.cluster import HDBSCAN

        estimator = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        backend = "sklearn.cluster.HDBSCAN"

    labels = np.asarray(estimator.fit_predict(arr), dtype=np.int64)
    n_clusters, n_noise = _summarize_labels(labels)
    return ClusteringResult(
        method="hdbscan",
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        space=space,
        seed=seed,
        params={
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "backend": backend,
        },
    )


def cluster_ward(
    features: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 42,
    space: str = "high_dim",
) -> ClusteringResult:
    """Run Ward hierarchical agglomerative clustering."""
    arr = _validate_features(features)
    if n_clusters < 1 or n_clusters > arr.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} must be between 1 and {arr.shape[0]}.")

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = np.asarray(model.fit_predict(arr), dtype=np.int64)
    count, n_noise = _summarize_labels(labels)
    return ClusteringResult(
        method="ward",
        labels=labels,
        n_clusters=count,
        n_noise=n_noise,
        space=space,
        seed=seed,
        params={"n_clusters": n_clusters, "linkage": "ward"},
    )


def cluster_kmeans(
    features: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 42,
    space: str = "high_dim",
) -> ClusteringResult:
    """Run k-means clustering."""
    arr = _validate_features(features)
    if n_clusters < 1 or n_clusters > arr.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} must be between 1 and {arr.shape[0]}.")

    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = np.asarray(model.fit_predict(arr), dtype=np.int64)
    count, n_noise = _summarize_labels(labels)
    return ClusteringResult(
        method="kmeans",
        labels=labels,
        n_clusters=count,
        n_noise=n_noise,
        space=space,
        seed=seed,
        params={"n_clusters": n_clusters},
    )


def cluster_gmm(
    features: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 42,
    covariance_type: str = "diag",
    space: str = "high_dim",
) -> ClusteringResult:
    """Run diagonal-covariance Gaussian Mixture clustering."""
    arr = _validate_features(features)
    if n_clusters < 1 or n_clusters > arr.shape[0]:
        raise ValueError(f"n_clusters={n_clusters} must be between 1 and {arr.shape[0]}.")

    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        random_state=seed,
        reg_covar=1e-5,
    )
    labels = np.asarray(model.fit_predict(arr), dtype=np.int64)
    count, n_noise = _summarize_labels(labels)
    return ClusteringResult(
        method="gmm",
        labels=labels,
        n_clusters=count,
        n_noise=n_noise,
        space=space,
        seed=seed,
        params={"n_clusters": n_clusters, "covariance_type": covariance_type},
    )


def run_clustering_suite(
    features: np.ndarray,
    methods: list[str],
    *,
    n_clusters: int | None = None,
    hdbscan_min_cluster_size: int = 15,
    seed: int = 42,
    space: str = "high_dim",
) -> dict[str, ClusteringResult]:
    """Run requested clustering methods, optionally inferring fixed k from HDBSCAN."""
    if not methods:
        raise ValueError("methods must contain at least one clustering method.")
    unknown = [m for m in methods if m not in SUPPORTED_CLUSTERERS]
    if unknown:
        raise ValueError(
            f"Unsupported clustering methods {unknown}; expected subset of {sorted(SUPPORTED_CLUSTERERS)}."
        )

    raw_results: dict[str, ClusteringResult] = {}
    inferred_k = n_clusters

    if "hdbscan" in methods:
        hdb_res = cluster_hdbscan(
            features,
            min_cluster_size=hdbscan_min_cluster_size,
            seed=seed,
            space=space,
        )
        raw_results["hdbscan"] = hdb_res
        if inferred_k is None and hdb_res.n_clusters >= 2:
            inferred_k = hdb_res.n_clusters

    for method in methods:
        if method == "hdbscan":
            continue
        if inferred_k is None or inferred_k < 2:
            raise ValueError(
                f"Clustering method {method!r} requires n_clusters >= 2 (provided {n_clusters!r} and HDBSCAN inferred {inferred_k!r})."
            )
        if method == "ward":
            raw_results[method] = cluster_ward(
                features, n_clusters=inferred_k, seed=seed, space=space
            )
        elif method == "kmeans":
            raw_results[method] = cluster_kmeans(
                features, n_clusters=inferred_k, seed=seed, space=space
            )
        elif method == "gmm":
            raw_results[method] = cluster_gmm(
                features, n_clusters=inferred_k, seed=seed, space=space
            )

    return {method: raw_results[method] for method in methods}
