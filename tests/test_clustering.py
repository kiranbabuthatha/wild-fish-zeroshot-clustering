"""Unit tests for dimensionality reduction (PCA, openTSNE, UMAP) and clustering."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from fish_clustering.clustering import (
    run_clustering_suite,
    run_reductions,
    transform_new_points,
)
from fish_clustering.models import l2_normalize_embeddings


def _make_clustered_embeddings(
    n_per_cluster: int = 20,
    dim: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, dim), dtype=np.float32)
    centers[0, 0] = 1.0
    centers[1, 5] = 1.0
    centers[2, 10] = 1.0

    train_chunks = []
    labels = []
    heldout_chunks = []
    for cid in range(3):
        noise = rng.normal(0.0, 0.08, size=(n_per_cluster, dim)).astype(np.float32)
        train_chunks.append(centers[cid] + noise)
        labels.extend([cid] * n_per_cluster)

        test_noise = rng.normal(0.0, 0.08, size=(2, dim)).astype(np.float32)
        heldout_chunks.append(centers[cid] + test_noise)

    X_train = l2_normalize_embeddings(np.vstack(train_chunks))
    X_heldout = l2_normalize_embeddings(np.vstack(heldout_chunks))
    return X_train, np.asarray(labels, dtype=np.int64), X_heldout


def test_run_reductions_and_out_of_sample_transform() -> None:
    X_train, _, X_heldout = _make_clustered_embeddings(n_per_cluster=20, dim=16, seed=42)

    results_a = run_reductions(
        X_train,
        methods=["pca", "tsne", "umap"],
        n_components=2,
        tsne_perplexity=10.0,
        umap_n_neighbors=10,
        umap_min_dist=0.1,
        seed=42,
    )
    results_b = run_reductions(
        X_train,
        methods=["pca", "tsne", "umap"],
        n_components=2,
        tsne_perplexity=10.0,
        umap_n_neighbors=10,
        umap_min_dist=0.1,
        seed=42,
    )

    assert set(results_a.keys()) == {"pca", "tsne", "umap"}
    for method in ("pca", "tsne", "umap"):
        res = results_a[method]
        assert res.coords.shape == (60, 2)
        assert np.all(np.isfinite(res.coords))
        assert np.allclose(res.coords, results_b[method].coords, atol=1e-4)

        projected = transform_new_points(res, X_heldout)
        assert projected.shape == (6, 2)
        assert np.all(np.isfinite(projected))

    with pytest.raises(ValueError, match="Unsupported reduction method"):
        run_reductions(X_train, methods=["unknown"])


def test_run_clustering_suite_explicit_and_inferred_k() -> None:
    X_train, y_true, _ = _make_clustered_embeddings(n_per_cluster=20, dim=16, seed=7)

    explicit = run_clustering_suite(
        X_train,
        methods=["hdbscan", "ward", "kmeans", "gmm"],
        n_clusters=3,
        hdbscan_min_cluster_size=8,
        seed=7,
        space="high_dim",
    )
    assert set(explicit.keys()) == {"hdbscan", "ward", "kmeans", "gmm"}
    for method, res in explicit.items():
        assert res.labels.shape == (60,)
        assert res.n_clusters == 3
        ari = adjusted_rand_score(y_true, res.labels)
        assert ari > 0.9, f"Unexpectedly low ARI for {method}: {ari}"

    inferred = run_clustering_suite(
        X_train,
        methods=["hdbscan", "ward", "kmeans", "gmm"],
        n_clusters=None,
        hdbscan_min_cluster_size=8,
        seed=7,
        space="high_dim",
    )
    for method in ("ward", "kmeans", "gmm"):
        assert inferred[method].n_clusters == 3
        assert adjusted_rand_score(y_true, inferred[method].labels) > 0.9

    with pytest.raises(ValueError, match="Unsupported clustering methods"):
        run_clustering_suite(X_train, methods=["dbscan_bad"], n_clusters=3)

    with pytest.raises(ValueError, match="requires n_clusters >= 2"):
        run_clustering_suite(X_train, methods=["kmeans"], n_clusters=None)
