"""Dimensionality reduction wrappers supporting PCA, openTSNE, and UMAP."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
SUPPORTED_REDUCERS = {"pca", "tsne", "umap"}


@dataclass
class ReducerResult:
    """Output container from a fitted dimensionality reduction method."""

    method: str
    coords: np.ndarray
    n_components: int
    seed: int
    params: dict[str, Any] = field(default_factory=dict)
    model: Any = None


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings array, got shape {arr.shape}.")
    if arr.shape[0] < 2:
        raise ValueError(f"At least 2 samples are required for reduction, got {arr.shape[0]}.")
    return arr


def fit_pca(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    seed: int = 42,
) -> ReducerResult:
    """Fit PCA and return reduced coordinates alongside the fitted model."""
    from sklearn.decomposition import PCA

    arr = _validate_embeddings(embeddings)
    effective_components = min(int(n_components), arr.shape[0], arr.shape[1])
    if effective_components < 1:
        raise ValueError("n_components must be at least 1.")

    model = PCA(n_components=effective_components, random_state=seed)
    coords = np.asarray(model.fit_transform(arr), dtype=np.float32)
    return ReducerResult(
        method="pca",
        coords=coords,
        n_components=effective_components,
        seed=seed,
        params={"n_components": effective_components},
        model=model,
    )


def fit_tsne(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    seed: int = 42,
    metric: str = "cosine",
) -> ReducerResult:
    """Fit openTSNE (preserving out-of-sample transform support) on embeddings."""
    from openTSNE import TSNE

    arr = _validate_embeddings(embeddings)
    max_perplexity = max(1.0, (arr.shape[0] - 1) / 3.0)
    effective_perplexity = min(float(perplexity), max_perplexity)
    if effective_perplexity < float(perplexity):
        LOGGER.warning(
            "Clamped t-SNE perplexity from %.1f to %.2f for n_samples=%d.",
            perplexity,
            effective_perplexity,
            arr.shape[0],
        )

    tsne = TSNE(
        n_components=int(n_components),
        perplexity=effective_perplexity,
        metric=metric,
        initialization="pca",
        random_state=seed,
        n_jobs=-1,
    )
    embedding = tsne.fit(arr)
    coords = np.asarray(embedding, dtype=np.float32)
    return ReducerResult(
        method="tsne",
        coords=coords,
        n_components=int(n_components),
        seed=seed,
        params={"perplexity": effective_perplexity, "metric": metric},
        model=embedding,
    )


def fit_umap(
    embeddings: np.ndarray,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    metric: str = "cosine",
) -> ReducerResult:
    """Fit UMAP on embeddings and return reduced coordinates plus fitted reducer."""
    import umap

    arr = _validate_embeddings(embeddings)
    effective_neighbors = min(int(n_neighbors), max(2, arr.shape[0] - 1))
    if effective_neighbors < int(n_neighbors):
        LOGGER.warning(
            "Clamped UMAP n_neighbors from %d to %d for n_samples=%d.",
            n_neighbors,
            effective_neighbors,
            arr.shape[0],
        )

    model = umap.UMAP(
        n_components=int(n_components),
        n_neighbors=effective_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        random_state=seed,
        n_jobs=1,
    )
    coords = np.asarray(model.fit_transform(arr), dtype=np.float32)
    return ReducerResult(
        method="umap",
        coords=coords,
        n_components=int(n_components),
        seed=seed,
        params={
            "n_neighbors": effective_neighbors,
            "min_dist": float(min_dist),
            "metric": metric,
        },
        model=model,
    )


def transform_new_points(result: ReducerResult, new_embeddings: np.ndarray) -> np.ndarray:
    """Project unseen embeddings into an already fitted PCA, openTSNE, or UMAP space."""
    if result.model is None:
        raise ValueError(f"ReducerResult for {result.method!r} has no fitted model.")

    new_arr = np.asarray(new_embeddings, dtype=np.float32)
    if new_arr.ndim != 2:
        raise ValueError(f"Expected 2-D array of unseen embeddings, got shape {new_arr.shape}.")
    if new_arr.shape[0] == 0:
        return np.empty((0, result.n_components), dtype=np.float32)

    transformed = result.model.transform(new_arr)
    return np.asarray(transformed, dtype=np.float32)


def run_reductions(
    embeddings: np.ndarray,
    methods: list[str],
    *,
    n_components: int = 2,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    seed: int = 42,
) -> dict[str, ReducerResult]:
    """Run all configured dimensionality reduction methods on an embedding matrix."""
    if not methods:
        raise ValueError("methods must contain at least one reduction method.")

    results: dict[str, ReducerResult] = {}
    for method in methods:
        if method not in SUPPORTED_REDUCERS:
            raise ValueError(
                f"Unsupported reduction method {method!r}; expected one of {sorted(SUPPORTED_REDUCERS)}."
            )
        if method == "pca":
            results[method] = fit_pca(embeddings, n_components=n_components, seed=seed)
        elif method == "tsne":
            results[method] = fit_tsne(
                embeddings,
                n_components=n_components,
                perplexity=tsne_perplexity,
                seed=seed,
            )
        elif method == "umap":
            results[method] = fit_umap(
                embeddings,
                n_components=n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                seed=seed,
            )
    return results
