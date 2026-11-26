"""Supervised and label-free clustering evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)

from fish_clustering.clustering.algorithms import ClusteringResult


def _is_missing_label(val: object) -> bool:
    if val is None or pd.isna(val):
        return True
    return isinstance(val, str) and val.strip() == ""


def compute_supervised_clustering_metrics(
    true_labels: Sequence[Any],
    pred_labels: np.ndarray,
) -> dict[str, float]:
    """Compute V-measure, homogeneity, completeness, ARI, and AMI over labeled samples."""
    preds = np.asarray(pred_labels, dtype=np.int64)
    if len(true_labels) != preds.shape[0]:
        raise ValueError(
            f"Length mismatch between true_labels ({len(true_labels)}) and pred_labels ({preds.shape[0]})."
        )

    valid_idx = [i for i, lbl in enumerate(true_labels) if not _is_missing_label(lbl)]
    if not valid_idx:
        return {
            "v_measure": float("nan"),
            "homogeneity": float("nan"),
            "completeness": float("nan"),
            "ari": float("nan"),
            "ami": float("nan"),
            "noise_fraction": 0.0,
            "n_true_classes": 0,
            "n_pred_clusters": 0,
            "n_samples": 0,
        }

    y_true = [str(true_labels[i]).strip() for i in valid_idx]
    y_pred = preds[valid_idx]

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    n_pred_clusters = len({int(lbl) for lbl in y_pred.tolist() if lbl >= 0})
    return {
        "v_measure": float(v_measure),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "ari": float(ari),
        "ami": float(ami),
        "noise_fraction": float(np.mean(y_pred == -1)),
        "n_true_classes": len(set(y_true)),
        "n_pred_clusters": int(n_pred_clusters),
        "n_samples": int(len(y_true)),
    }


def compute_label_free_metrics(
    features: np.ndarray,
    pred_labels: np.ndarray,
    *,
    metric: str = "euclidean",
    compute_dbcv: bool = True,
    max_dbcv_dim: int = 10,
) -> dict[str, float]:
    """Compute silhouette, Calinski-Harabasz, Davies-Bouldin, and low-D DBCV scores."""
    arr = np.asarray(features, dtype=np.float64)
    preds = np.asarray(pred_labels, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[0] != preds.shape[0]:
        raise ValueError(
            f"Feature matrix shape {arr.shape} does not match pred_labels shape {preds.shape}."
        )

    valid_mask = preds != -1
    valid_features = arr[valid_mask]
    valid_labels = preds[valid_mask]
    unique_clusters = np.unique(valid_labels)

    metrics: dict[str, float] = {
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "dbcv": float("nan"),
    }

    if len(unique_clusters) >= 2 and valid_features.shape[0] > len(unique_clusters):
        sil_metric = metric if metric in {"euclidean", "cosine", "manhattan"} else "euclidean"
        metrics["silhouette"] = float(
            silhouette_score(valid_features, valid_labels, metric=sil_metric)
        )
        metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(valid_features, valid_labels)
        )
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(valid_features, valid_labels)
        )

        if compute_dbcv and arr.shape[1] <= max_dbcv_dim:
            try:
                from hdbscan.validity import validity_index

                metrics["dbcv"] = float(validity_index(arr, preds, metric=metric))
            except Exception:
                metrics["dbcv"] = float("nan")

    return metrics


def evaluate_clustering_results(
    results: dict[str, ClusteringResult],
    features: np.ndarray,
    true_labels: Sequence[Any] | None = None,
    *,
    compute_label_free: bool = True,
    metric: str = "euclidean",
) -> list[dict[str, Any]]:
    """Build flat evaluation records for a dictionary of ClusteringResult objects."""
    rows: list[dict[str, Any]] = []
    for res in results.values():
        row: dict[str, Any] = {
            "method": res.method,
            "space": res.space,
            "seed": res.seed,
            "n_clusters": res.n_clusters,
            "n_noise": res.n_noise,
        }
        if true_labels is not None:
            row.update(compute_supervised_clustering_metrics(true_labels, res.labels))
        if compute_label_free:
            row.update(
                compute_label_free_metrics(
                    features,
                    res.labels,
                    metric=metric,
                )
            )
        rows.append(row)
    return rows
