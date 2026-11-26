"""Interactive HTML embedding explorer and distortion/OOD overlay builders."""

from fish_clustering.visualization.atlas import (
    build_atlas_dataframe,
    export_diagnostic_atlas_html,
    export_visual_atlas_html,
)
from fish_clustering.visualization.explorer import (
    compute_cluster_geometry,
    compute_top_k_neighbors,
    export_interactive_explorer,
    export_sample_thumbnails,
)

__all__ = [
    "build_atlas_dataframe",
    "compute_cluster_geometry",
    "compute_top_k_neighbors",
    "export_diagnostic_atlas_html",
    "export_interactive_explorer",
    "export_sample_thumbnails",
    "export_visual_atlas_html",
]
