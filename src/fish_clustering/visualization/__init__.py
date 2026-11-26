"""Interactive HTML embedding explorer and distortion/OOD overlay builders."""

from fish_clustering.visualization.atlas import (
    build_atlas_dataframe,
    export_diagnostic_atlas_html,
    export_visual_atlas_html,
)

__all__ = [
    "build_atlas_dataframe",
    "export_diagnostic_atlas_html",
    "export_visual_atlas_html",
]
