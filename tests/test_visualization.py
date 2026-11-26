"""Unit tests for interactive HTML visual atlas builders."""

from pathlib import Path

import numpy as np
import pandas as pd

from fish_clustering.visualization import (
    build_atlas_dataframe,
    export_diagnostic_atlas_html,
    export_visual_atlas_html,
)


def test_build_and_export_html_atlases(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "image_id": [f"img_{i}" for i in range(6)],
            "dataset": ["ozfish", "ozfish", "ozfish", "fishwio", "fishwio", "schufi"],
            "source_type": ["wild"] * 5 + ["controlled"],
            "label_raw": ["sp_a", "sp_a", "sp_b", "sp_a", "sp_b", "sp_c"],
            "species": ["Caranx sp", "Caranx sp", "Chromis sp", "Caranx sp", "Chromis sp", ""],
            "video_id": ["v1"] * 6,
            "image_path": [f"p_{i}.png" for i in range(6)],
        }
    )
    coords = np.array(
        [[0.0, 0.1], [0.2, 0.0], [1.0, 1.2], [0.1, 0.2], [1.1, 1.0], [3.0, 3.0]],
        dtype=np.float32,
    )
    clusters = {"kmeans": np.array([0, 0, 1, 0, 1, -1], dtype=np.int64)}
    distortion = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.8], dtype=np.float32)
    ood = np.array([0.05, 0.06, 0.08, 0.07, 0.09, 0.75], dtype=np.float32)

    atlas_df = build_atlas_dataframe(
        manifest,
        coords,
        cluster_labels=clusters,
        distortion=distortion,
        ood_scores=ood,
    )
    assert {"x", "y", "cluster_kmeans", "distortion", "ood_score"}.issubset(atlas_df.columns)
    assert atlas_df["cluster_kmeans"].iloc[-1] == "noise (-1)"

    atlas_html = export_visual_atlas_html(
        atlas_df,
        tmp_path / "atlas.html",
        color_by="cluster_kmeans",
        title="Test Visual Atlas",
    )
    assert atlas_html.is_file()
    atlas_text = atlas_html.read_text(encoding="utf-8")
    assert "Test Visual Atlas" in atlas_text
    assert "plotly" in atlas_text.lower()

    diag_html = export_diagnostic_atlas_html(
        atlas_df,
        tmp_path / "diag.html",
        color_by="dataset",
        title="Test Diagnostic Atlas",
    )
    assert diag_html.is_file()
    diag_text = diag_html.read_text(encoding="utf-8")
    assert "Test Diagnostic Atlas" in diag_text
    assert "Neighborhood Distortion" in diag_text
