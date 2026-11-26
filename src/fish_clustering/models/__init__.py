"""Frozen Vision Transformer and Vision-Language feature extractors."""

from fish_clustering.models.backbones import (
    ALLOWED_POOLING,
    BACKBONE_SPECS,
    BackboneSpec,
    BaseFeatureExtractor,
    HuggingFaceViTExtractor,
    OpenCLIPExtractor,
    TimmViTExtractor,
    create_extractor,
    get_backbone_spec,
    list_supported_backbones,
)
from fish_clustering.models.cache import (
    CachedEmbeddings,
    build_cache_key,
    extract_or_load_embeddings,
    load_embeddings,
    save_embeddings,
)
from fish_clustering.models.extractor import (
    extract_embeddings,
    l2_normalize_embeddings,
)

__all__ = [
    "ALLOWED_POOLING",
    "BACKBONE_SPECS",
    "BackboneSpec",
    "BaseFeatureExtractor",
    "CachedEmbeddings",
    "HuggingFaceViTExtractor",
    "OpenCLIPExtractor",
    "TimmViTExtractor",
    "build_cache_key",
    "create_extractor",
    "extract_embeddings",
    "extract_or_load_embeddings",
    "get_backbone_spec",
    "l2_normalize_embeddings",
    "list_supported_backbones",
    "load_embeddings",
    "save_embeddings",
]
