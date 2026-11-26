"""Unit tests for clustering metrics, projection quality, and RQ2 OOD/mixing."""

import numpy as np
import pytest

from fish_clustering.clustering import ClusteringResult, fit_pca
from fish_clustering.evaluation import (
    compute_continuity,
    compute_cross_dataset_species_match,
    compute_dataset_mixing_entropy,
    compute_knn_ood_scores,
    compute_label_free_metrics,
    compute_projection_quality,
    compute_self_knn_ood_scores,
    compute_supervised_clustering_metrics,
    compute_trustworthiness,
    evaluate_clustering_results,
)
from fish_clustering.models import l2_normalize_embeddings


def test_supervised_and_label_free_clustering_metrics() -> None:
    true_labels = ["A", "A", "A", "B", "B", "B", "", None]
    pred_exact = np.array([0, 0, 0, 1, 1, 1, 0, -1], dtype=np.int64)
    metrics = compute_supervised_clustering_metrics(true_labels, pred_exact)
    assert metrics["n_samples"] == 6
    assert metrics["v_measure"] == pytest.approx(1.0)
    assert metrics["ari"] == pytest.approx(1.0)
    assert metrics["ami"] == pytest.approx(1.0)
    assert metrics["noise_fraction"] == pytest.approx(0.0)

    empty = compute_supervised_clustering_metrics(["", None], np.array([0, 1], dtype=np.int64))
    assert empty["n_samples"] == 0
    assert np.isnan(empty["v_measure"])

    # Well-separated 2-D blobs should produce strong silhouette and valid DBCV.
    rng = np.random.default_rng(42)
    coords_2d = np.vstack(
        [
            rng.normal([0.0, 0.0], 0.1, size=(12, 2)),
            rng.normal([5.0, 5.0], 0.1, size=(12, 2)),
        ]
    )
    preds_2d = np.array([0] * 12 + [1] * 12, dtype=np.int64)
    lf_2d = compute_label_free_metrics(coords_2d, preds_2d)
    assert lf_2d["silhouette"] > 0.9
    assert np.isfinite(lf_2d["calinski_harabasz"])
    assert np.isfinite(lf_2d["dbcv"])

    # High-dimensional features (>10-D) skip DBCV cleanly.
    feats_16d = rng.normal(size=(24, 16))
    lf_16d = compute_label_free_metrics(feats_16d, preds_2d)
    assert np.isnan(lf_16d["dbcv"])

    evaluated = evaluate_clustering_results(
        {"kmeans": ClusteringResult("kmeans", preds_2d, 2, 0, "umap", 42)},
        coords_2d,
        true_labels=["sp0"] * 12 + ["sp1"] * 12,
    )
    assert len(evaluated) == 1
    assert evaluated[0]["v_measure"] == pytest.approx(1.0)


def test_projection_quality_metrics() -> None:
    rng = np.random.default_rng(7)
    base_2d = np.vstack(
        [
            rng.normal([-3.0, -3.0], 0.25, size=(15, 2)),
            rng.normal([0.0, 4.0], 0.25, size=(15, 2)),
            rng.normal([4.0, -2.0], 0.25, size=(15, 2)),
        ]
    ).astype(np.float32)
    # Embed 2-D manifold cleanly into 10-D so PCA 2-D preserves neighborhoods.
    high_10d = np.hstack([base_2d, np.ones((45, 1), dtype=np.float32) * 3.0, np.zeros((45, 7), dtype=np.float32)])
    high_10d = l2_normalize_embeddings(high_10d)
    pca_res = fit_pca(high_10d, n_components=2, seed=7)

    pq = compute_projection_quality(high_10d, pca_res.coords, n_neighbors=8)
    assert pq["trustworthiness"] > 0.9
    assert pq["continuity"] > 0.9
    assert pq["per_point_distortion"].shape == (45,)
    assert np.all((pq["per_point_distortion"] >= 0.0) & (pq["per_point_distortion"] <= 1.0))

    shuffled = rng.normal(size=(45, 2)).astype(np.float32)
    assert compute_trustworthiness(high_10d, shuffled, n_neighbors=8) < 0.7
    assert compute_continuity(high_10d, shuffled, n_neighbors=8) < 0.7


def test_rq2_mixing_cross_species_and_ood() -> None:
    rng = np.random.default_rng(12)
    c0 = np.zeros(12, dtype=np.float32)
    c0[0] = 1.0
    c1 = np.zeros(12, dtype=np.float32)
    c1[4] = 1.0

    # Two datasets overlapping inside each species cluster -> high mixing & cross-match.
    embs_mixed = l2_normalize_embeddings(
        np.vstack(
            [
                c0 + rng.normal(0.0, 0.05, size=(20, 12)),
                c1 + rng.normal(0.0, 0.05, size=(20, 12)),
            ]
        )
    )
    datasets_mixed = ["ozfish"] * 10 + ["fishwio"] * 10 + ["ozfish"] * 10 + ["fishwio"] * 10
    species = ["Caranx melampygus"] * 20 + ["Chromis chromis"] * 20

    mix = compute_dataset_mixing_entropy(embs_mixed, datasets_mixed, k_neighbors=6)
    assert mix["mean_normalized_entropy"] > 0.75

    cross = compute_cross_dataset_species_match(
        embs_mixed, datasets_mixed, species, k_neighbors=5
    )
    assert cross["n_shared_species"] == 2
    assert cross["top1_cross_dataset_accuracy"] == pytest.approx(1.0)
    assert cross["knn_cross_dataset_precision"] == pytest.approx(1.0)

    # Disjoint dataset blobs -> near-zero cross-dataset mixing.
    datasets_sep = ["ozfish"] * 20 + ["fishwio"] * 20
    sep_mix = compute_dataset_mixing_entropy(embs_mixed, datasets_sep, k_neighbors=6)
    assert sep_mix["mean_normalized_entropy"] < 0.1

    # OOD scoring assigns much larger cosine distance to orthogonal vectors.
    outlier = np.zeros((3, 12), dtype=np.float32)
    outlier[:, 9] = 1.0
    in_scores = compute_knn_ood_scores(embs_mixed, embs_mixed[:4], k_neighbors=5)
    ood_scores = compute_knn_ood_scores(embs_mixed, outlier, k_neighbors=5)
    assert float(np.mean(ood_scores)) > float(np.mean(in_scores)) + 0.5

    self_scores = compute_self_knn_ood_scores(embs_mixed, k_neighbors=5)
    assert self_scores.shape == (40,)
    assert np.all(self_scores >= 0.0)
