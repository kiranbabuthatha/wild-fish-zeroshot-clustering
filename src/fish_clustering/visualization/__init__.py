"""Interactive HTML embedding explorer and publication static figure builders."""

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
from fish_clustering.visualization.figures import (
    export_static_figures,
    plot_cluster_composition,
    plot_embedding_scatter,
    plot_k_sweep,
    plot_projection_quality_comparison,
    plot_rq2_diagnostics,
    plot_thumbnail_map,
    plot_ward_dendrogram,
)

__all__ = [
    "build_atlas_dataframe",
    "compute_cluster_geometry",
    "compute_top_k_neighbors",
    "export_diagnostic_atlas_html",
    "export_interactive_explorer",
    "export_sample_thumbnails",
    "export_static_figures",
    "export_visual_atlas_html",
    "plot_cluster_composition",
    "plot_embedding_scatter",
    "plot_k_sweep",
    "plot_projection_quality_comparison",
    "plot_rq2_diagnostics",
    "plot_thumbnail_map",
    "plot_ward_dendrogram",
]
