"""Publication-ready static Matplotlib figures for thesis chapters and slides."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure, silhouette_score

from fish_clustering.data.preprocessing import load_image_sample


def _save_fig(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _categorical_colors(n: int) -> list[Any]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def plot_embedding_scatter(
    coords: np.ndarray,
    labels: Sequence[Any],
    output_path: str | Path,
    *,
    title: str,
    legend_title: str = "Label",
    dpi: int = 300,
) -> Path:
    """Plot a clean 2-D scatter map grouped by discrete cluster or species/dataset labels."""
    xy = np.asarray(coords, dtype=np.float32)
    raw_labels = [str(lbl) if lbl is not None and not pd.isna(lbl) and str(lbl).strip() != "" else "unlabeled" for lbl in labels]

    counts = Counter(raw_labels)
    non_noise = [(lbl, cnt) for lbl, cnt in counts.most_common() if lbl not in {"-1", "noise", "Noise (-1)"}]
    top_labels = [lbl for lbl, _ in non_noise[:18]]
    top_set = set(top_labels)

    mapped_labels = []
    for lbl in raw_labels:
        if lbl in {"-1", "noise", "Noise (-1)"}:
            mapped_labels.append("Noise (-1)")
        elif lbl in top_set:
            mapped_labels.append(lbl)
        else:
            mapped_labels.append("Other")

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    unique_order = [lbl for lbl in top_labels if lbl in mapped_labels]
    if "Other" in mapped_labels:
        unique_order.append("Other")
    if "Noise (-1)" in mapped_labels:
        unique_order.append("Noise (-1)")

    colors = _categorical_colors(len(unique_order))
    arr_lbl = np.asarray(mapped_labels)

    for idx, lbl in enumerate(unique_order):
        mask = arr_lbl == lbl
        is_noise = lbl == "Noise (-1)"
        color = "#9aa5b1" if is_noise else ("#cbd5e1" if lbl == "Other" else colors[idx])
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=14 if is_noise else 22,
            alpha=0.45 if is_noise else 0.82,
            color=color,
            edgecolors="none",
            label=f"{lbl} ({int(mask.sum())})",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        title=legend_title,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    return _save_fig(fig, output_path, dpi=dpi)


def plot_thumbnail_map(
    coords: np.ndarray,
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    raw_root: Path | None = None,
    max_thumbs: int = 64,
    thumb_px: int = 42,
    title: str = "Thumbnail Embedding Map",
    dpi: int = 300,
) -> Path:
    """Place non-overlapping fish crop thumbnails onto a 2-D embedding scatter."""
    xy = np.asarray(coords, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color="#64748b", alpha=0.35, edgecolors="none")

    if len(xy) > 0:
        x_span = max(float(np.ptp(xy[:, 0])), 1e-6)
        y_span = max(float(np.ptp(xy[:, 1])), 1e-6)
        norm_xy = np.column_stack(
            [(xy[:, 0] - xy[:, 0].min()) / x_span, (xy[:, 1] - xy[:, 1].min()) / y_span]
        )

        # Greedy minimum-spacing placement in normalized [0, 1]^2 space.
        min_dist_sq = 0.085 ** 2
        placed: list[np.ndarray] = []
        records = manifest.to_dict(orient="records")
        step = max(1, len(records) // max(max_thumbs * 4, 1))
        candidate_indices = list(range(0, len(records), step))

        for idx in candidate_indices:
            if len(placed) >= max_thumbs:
                break
            pt = norm_xy[idx]
            if any(float(np.sum((pt - prev) ** 2)) < min_dist_sq for prev in placed):
                continue

            rec = records[idx]
            raw_img = str(rec.get("image_path") or "")
            if not raw_img:
                continue
            p_direct = Path(raw_img)
            exists = p_direct.is_file() or (raw_root is not None and (raw_root / p_direct).is_file())
            if not exists:
                continue

            img = load_image_sample(rec, raw_root=raw_root, crop_from_bbox=True)
            img.thumbnail((thumb_px, thumb_px), resample=Image.BICUBIC)
            ab = AnnotationBbox(
                OffsetImage(np.asarray(img), zoom=0.75),
                (float(xy[idx, 0]), float(xy[idx, 1])),
                frameon=True,
                pad=0.1,
                bboxprops={"edgecolor": "#334155", "linewidth": 0.6},
            )
            ax.add_artist(ab)
            placed.append(pt)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, linestyle=":", alpha=0.35)
    return _save_fig(fig, output_path, dpi=dpi)


def plot_cluster_composition(
    pred_labels: Sequence[int],
    group_labels: Sequence[Any],
    output_path: str | Path,
    *,
    title: str = "Cluster Composition",
    top_n_groups: int = 12,
    dpi: int = 300,
) -> Path:
    """Plot a horizontal stacked bar chart of species or dataset share inside each cluster."""
    preds = [f"Noise (-1)" if int(lbl) == -1 else f"Cluster {int(lbl)}" for lbl in pred_labels]
    groups = [str(g) if g is not None and not pd.isna(g) and str(g).strip() != "" else "unlabeled" for g in group_labels]

    df = pd.DataFrame({"cluster": preds, "group": groups})
    top_groups = [g for g, _ in Counter(groups).most_common(top_n_groups)]
    df["group_clipped"] = np.where(df["group"].isin(top_groups), df["group"], "Other")

    ctab = pd.crosstab(df["cluster"], df["group_clipped"])
    if ctab.shape[0] > 20:
        keep_clusters = ctab.sum(axis=1).sort_values(ascending=False).head(20).index
        ctab = ctab.loc[keep_clusters]

    ctab_pct = ctab.div(ctab.sum(axis=1).clip(lower=1), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(8.8, max(3.8, 0.36 * len(ctab_pct) + 1.5)))
    ctab_pct.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        colormap="tab20",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Share of Cluster Samples (%)")
    ax.set_ylabel("Assigned Cluster")
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8)
    return _save_fig(fig, output_path, dpi=dpi)


def plot_projection_quality_comparison(
    projection_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Projection Quality Comparison",
    dpi: int = 300,
) -> Path:
    """Compare Trustworthiness, Continuity, and Neighborhood Jaccard across DR methods."""
    metrics = ["trustworthiness", "continuity", "mean_neighborhood_jaccard"]
    labels = ["Trustworthiness", "Continuity", "Neighborhood Jaccard"]
    grouped = projection_df.groupby("reduction", sort=False)[metrics].agg(["mean", "std"]).fillna(0.0)

    methods = [str(m).upper() for m in grouped.index.tolist()]
    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, (metric_col, display_name) in enumerate(zip(metrics, labels)):
        means = grouped[(metric_col, "mean")].to_numpy()
        stds = grouped[(metric_col, "std")].to_numpy()
        ax.bar(
            x + (i - 1) * width,
            means,
            width=width,
            yerr=stds if len(projection_df) > len(methods) else None,
            capsize=3,
            label=display_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(frameon=False, loc="lower right")
    return _save_fig(fig, output_path, dpi=dpi)


def plot_rq2_diagnostics(
    ood_scores: np.ndarray,
    datasets: Sequence[str],
    output_path: str | Path,
    *,
    mixing_entropies: np.ndarray | None = None,
    title: str = "High-D OOD and Dataset Mixing Diagnostics",
    dpi: int = 300,
) -> Path:
    """Plot per-dataset high-D k-NN OOD score distributions and optional mixing entropy."""
    ds_arr = np.asarray([str(d) for d in datasets])
    unique_ds = sorted(np.unique(ds_arr).tolist())
    n_panels = 2 if mixing_entropies is not None and len(unique_ds) >= 2 else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.8))
    ax_ood = axes[0] if n_panels == 2 else axes

    ood_groups = [np.asarray(ood_scores)[ds_arr == d] for d in unique_ds]
    ax_ood.boxplot(ood_groups, tick_labels=unique_ds, patch_artist=True)
    ax_ood.set_title("High-D k-NN Cosine OOD Distance", fontsize=11, fontweight="bold")
    ax_ood.set_ylabel("Mean k-NN Cosine Distance")
    ax_ood.tick_params(axis="x", rotation=20)
    ax_ood.grid(True, axis="y", linestyle=":", alpha=0.4)

    if n_panels == 2 and mixing_entropies is not None:
        ax_mix = axes[1]
        ax_mix.hist(np.asarray(mixing_entropies), bins=15, range=(0.0, 1.0), color="#2563eb", alpha=0.82, edgecolor="white")
        ax_mix.set_title("Normalized Local Dataset Mixing Entropy", fontsize=11, fontweight="bold")
        ax_mix.set_xlabel("Normalized Shannon Entropy (0 = isolated, 1 = mixed)")
        ax_mix.set_ylabel("Samples")
        ax_mix.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    return _save_fig(fig, output_path, dpi=dpi)


def plot_ward_dendrogram(
    features: np.ndarray,
    output_path: str | Path,
    *,
    max_samples: int = 200,
    truncate_p: int = 25,
    seed: int = 42,
    title: str = "Ward Hierarchical Dendrogram",
    dpi: int = 300,
) -> Path:
    """Plot a truncated Ward linkage dendrogram over high-D embeddings."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(arr.shape[0], size=max_samples, replace=False))
        arr = arr[idx]

    z_mat = linkage(arr, method="ward")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    dendrogram(
        z_mat,
        truncate_mode="lastp",
        p=min(truncate_p, max(2, arr.shape[0] - 1)),
        ax=ax,
        color_threshold=0.0,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster Size / Leaf Index")
    ax.set_ylabel("Ward Linkage Distance")
    return _save_fig(fig, output_path, dpi=dpi)


def plot_k_sweep(
    features: np.ndarray,
    output_path: str | Path,
    *,
    true_labels: Sequence[Any] | None = None,
    seed: int = 42,
    k_min: int = 2,
    k_max: int = 12,
    title: str = "K-Means Sweep Across Cluster Counts",
    dpi: int = 300,
) -> Path:
    """Plot silhouette, V-measure, and ARI curves across candidate k values."""
    arr = np.asarray(features, dtype=np.float32)
    upper = min(int(k_max), arr.shape[0] - 1)
    ks = list(range(max(2, int(k_min)), max(2, upper) + 1))

    sil_scores: list[float] = []
    v_scores: list[float] = []
    ari_scores: list[float] = []

    has_labels = true_labels is not None and any(
        lbl is not None and not pd.isna(lbl) and str(lbl).strip() != "" for lbl in true_labels
    )
    if has_labels and true_labels is not None:
        valid_mask = np.asarray(
            [lbl is not None and not pd.isna(lbl) and str(lbl).strip() != "" for lbl in true_labels],
            dtype=bool,
        )
        clean_true = [str(true_labels[i]).strip() for i in np.where(valid_mask)[0]]
    else:
        valid_mask = np.zeros(arr.shape[0], dtype=bool)
        clean_true = []

    for k in ks:
        preds = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit_predict(arr)
        sil_scores.append(float(silhouette_score(arr, preds)))
        if has_labels and len(clean_true) >= 2:
            sub_preds = preds[valid_mask]
            _, _, vm = homogeneity_completeness_v_measure(clean_true, sub_preds)
            v_scores.append(float(vm))
            ari_scores.append(float(adjusted_rand_score(clean_true, sub_preds)))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(ks, sil_scores, marker="o", label="Silhouette")
    if v_scores:
        ax.plot(ks, v_scores, marker="s", label="V-Measure")
        ax.plot(ks, ari_scores, marker="^", label="ARI")

    ax.set_xticks(ks)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    return _save_fig(fig, output_path, dpi=dpi)


def export_static_figures(
    *,
    experiment_name: str,
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    run_data: dict[int, dict[str, dict[str, Any]]],
    projection_rows: list[dict[str, Any]],
    ood_scores: np.ndarray,
    output_dir: str | Path,
    raw_root: Path | None = None,
    mixing_entropies: np.ndarray | None = None,
    true_labels: Sequence[Any] | None = None,
    primary_seed: int | None = None,
) -> list[str]:
    """Generate the full suite of static thesis figures for an experiment run."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    seed = primary_seed if primary_seed is not None else sorted(run_data.keys())[0]
    seed_runs = run_data.get(seed, {})
    datasets_col = manifest["dataset"].tolist() if "dataset" in manifest.columns else ["unknown"] * len(manifest)
    has_multi_dataset = len(set(datasets_col)) > 1
    comp_labels = datasets_col if has_multi_dataset else (true_labels if true_labels is not None else datasets_col)

    for red_name, entry in seed_runs.items():
        coords = np.asarray(entry["coords"], dtype=np.float32)
        clusterings = entry.get("clusterings", {})
        primary_method = "hdbscan" if "hdbscan" in clusterings else next(iter(clusterings.keys()), None)

        if primary_method is not None:
            pred_labels = clusterings[primary_method]["labels"]
            p_cl = plot_embedding_scatter(
                coords,
                pred_labels,
                out_dir / f"map_{red_name}_by_{primary_method}.png",
                title=f"{experiment_name}: {red_name.upper()} ({primary_method.upper()})",
                legend_title="Cluster",
            )
            saved.append(str(p_cl))

            p_comp = plot_cluster_composition(
                pred_labels,
                comp_labels,
                out_dir / f"composition_{red_name}_{primary_method}.png",
                title=f"{experiment_name}: {red_name.upper()} {primary_method.upper()} Composition",
            )
            saved.append(str(p_comp))

        p_ds = plot_embedding_scatter(
            coords,
            datasets_col,
            out_dir / f"map_{red_name}_by_dataset.png",
            title=f"{experiment_name}: {red_name.upper()} by Dataset",
            legend_title="Dataset",
        )
        saved.append(str(p_ds))

        p_thumb = plot_thumbnail_map(
            coords,
            manifest,
            out_dir / f"thumbnails_{red_name}.png",
            raw_root=raw_root,
            title=f"{experiment_name}: {red_name.upper()} Thumbnail Atlas",
        )
        saved.append(str(p_thumb))

    if projection_rows:
        p_pq = plot_projection_quality_comparison(
            pd.DataFrame(projection_rows),
            out_dir / "projection_quality.png",
            title=f"{experiment_name}: Projection Quality",
        )
        saved.append(str(p_pq))

    p_rq2 = plot_rq2_diagnostics(
        ood_scores,
        datasets_col,
        out_dir / "rq2_diagnostics.png",
        mixing_entropies=mixing_entropies,
        title=f"{experiment_name}: High-D OOD & Mixing Diagnostics",
    )
    saved.append(str(p_rq2))

    if len(embeddings) >= 4:
        p_dend = plot_ward_dendrogram(
            embeddings,
            out_dir / "ward_dendrogram.png",
            seed=seed,
            title=f"{experiment_name}: Ward Dendrogram (High-D)",
        )
        saved.append(str(p_dend))

    if len(embeddings) >= 5:
        p_ksweep = plot_k_sweep(
            embeddings,
            out_dir / "ksweep_high_dim.png",
            true_labels=true_labels,
            seed=seed,
            title=f"{experiment_name}: K-Means Sweep (High-D)",
        )
        saved.append(str(p_ksweep))

    return saved
