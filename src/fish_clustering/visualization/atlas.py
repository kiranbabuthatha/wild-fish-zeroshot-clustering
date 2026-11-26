"""Interactive HTML embedding explorer and distortion/OOD overlay builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_MANIFEST_ATLAS_COLS = [
    "image_id",
    "dataset",
    "source_type",
    "label_raw",
    "species",
    "video_id",
    "image_path",
]


def build_atlas_dataframe(
    manifest: pd.DataFrame,
    coords: np.ndarray,
    *,
    cluster_labels: dict[str, np.ndarray] | None = None,
    distortion: np.ndarray | None = None,
    ood_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assemble 2-D projection coordinates, labels, distortion, and OOD scores into one table."""
    xy = np.asarray(coords, dtype=np.float32)
    if xy.ndim != 2 or xy.shape != (len(manifest), 2):
        raise ValueError(
            f"Expected coords of shape ({len(manifest)}, 2), got {xy.shape}."
        )

    keep_cols = [col for col in _MANIFEST_ATLAS_COLS if col in manifest.columns]
    atlas = manifest[keep_cols].reset_index(drop=True).copy()
    atlas["x"] = xy[:, 0]
    atlas["y"] = xy[:, 1]

    if cluster_labels:
        for method, labels in cluster_labels.items():
            arr = np.asarray(labels, dtype=np.int64)
            if arr.shape[0] != len(manifest):
                raise ValueError(
                    f"Cluster label length {arr.shape[0]} for {method!r} does not match manifest ({len(manifest)})."
                )
            atlas[f"cluster_{method}"] = [
                "noise (-1)" if int(lbl) == -1 else str(int(lbl)) for lbl in arr.tolist()
            ]

    if distortion is not None:
        dist_arr = np.asarray(distortion, dtype=np.float32)
        if dist_arr.shape[0] != len(manifest):
            raise ValueError("distortion length must match manifest row count.")
        atlas["distortion"] = dist_arr

    if ood_scores is not None:
        ood_arr = np.asarray(ood_scores, dtype=np.float32)
        if ood_arr.shape[0] != len(manifest):
            raise ValueError("ood_scores length must match manifest row count.")
        atlas["ood_score"] = ood_arr

    return atlas


def export_visual_atlas_html(
    atlas_df: pd.DataFrame,
    output_path: str | Path,
    *,
    color_by: str = "label_raw",
    title: str = "Wild Fish Embedding Atlas",
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Write a standalone interactive HTML scatter plot colored by species, dataset, or cluster."""
    import plotly.express as px

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    color_col = color_by if color_by in atlas_df.columns else None
    hover_cols = [
        col
        for col in ("image_id", "dataset", "label_raw", "species", "distortion", "ood_score")
        if col in atlas_df.columns
    ]

    fig = px.scatter(
        atlas_df,
        x="x",
        y="y",
        color=color_col,
        hover_data=hover_cols,
        title=title,
        template="plotly_white",
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    fig.update_layout(height=720, legend_title_text=color_col or "")
    fig.write_html(str(out_path), include_plotlyjs=include_plotlyjs)
    return out_path


def export_diagnostic_atlas_html(
    atlas_df: pd.DataFrame,
    output_path: str | Path,
    *,
    color_by: str = "label_raw",
    title: str = "Projection Quality and OOD Diagnostic Atlas",
    include_plotlyjs: str | bool = "cdn",
) -> Path:
    """Write a three-panel interactive atlas showing labels, projection distortion, and OOD score."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Embeddings ({color_by})",
            "Neighborhood Distortion",
            "High-D k-NN OOD Score",
        ),
    )

    hover_text = [
        "<br>".join(
            f"{col}: {row[col]}"
            for col in ("image_id", "dataset", "label_raw", "species")
            if col in atlas_df.columns and str(row[col]).strip() != ""
        )
        for _, row in atlas_df.iterrows()
    ]

    if color_by in atlas_df.columns:
        for group_val, sub_df in atlas_df.groupby(color_by, dropna=False):
            sub_idx = sub_df.index.tolist()
            fig.add_trace(
                go.Scatter(
                    x=sub_df["x"],
                    y=sub_df["y"],
                    mode="markers",
                    name=str(group_val),
                    text=[hover_text[i] for i in sub_idx],
                    hoverinfo="text",
                    marker={"size": 6, "opacity": 0.8},
                ),
                row=1,
                col=1,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=atlas_df["x"],
                y=atlas_df["y"],
                mode="markers",
                name="samples",
                text=hover_text,
                hoverinfo="text",
                marker={"size": 6, "opacity": 0.8},
            ),
            row=1,
            col=1,
        )

    dist_vals = (
        atlas_df["distortion"]
        if "distortion" in atlas_df.columns
        else np.zeros(len(atlas_df), dtype=np.float32)
    )
    fig.add_trace(
        go.Scatter(
            x=atlas_df["x"],
            y=atlas_df["y"],
            mode="markers",
            showlegend=False,
            text=hover_text,
            hoverinfo="text",
            marker={
                "size": 6,
                "color": dist_vals,
                "colorscale": "Turbo",
                "showscale": True,
                "colorbar": {"title": "Distortion", "x": 0.65, "len": 0.85},
            },
        ),
        row=1,
        col=2,
    )

    ood_vals = (
        atlas_df["ood_score"]
        if "ood_score" in atlas_df.columns
        else np.zeros(len(atlas_df), dtype=np.float32)
    )
    fig.add_trace(
        go.Scatter(
            x=atlas_df["x"],
            y=atlas_df["y"],
            mode="markers",
            showlegend=False,
            text=hover_text,
            hoverinfo="text",
            marker={
                "size": 6,
                "color": ood_vals,
                "colorscale": "Plasma",
                "showscale": True,
                "colorbar": {"title": "OOD", "x": 1.02, "len": 0.85},
            },
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title_text=title,
        template="plotly_white",
        height=620,
        width=1500,
    )
    fig.write_html(str(out_path), include_plotlyjs=include_plotlyjs)
    return out_path
