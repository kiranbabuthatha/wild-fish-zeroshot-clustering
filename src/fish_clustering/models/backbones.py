"""Registry and wrappers for frozen ViT and BioCLIP vision feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from fish_clustering.config import resolve_device

ALLOWED_POOLING = {"cls", "patch_mean"}


@dataclass(frozen=True)
class BackboneSpec:
    """Metadata describing a supported frozen vision backbone."""

    name: str
    family: str
    model_id: str
    default_image_size: int
    description: str


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "dinov3_vitb16": BackboneSpec(
        name="dinov3_vitb16",
        family="timm",
        model_id="vit_base_patch16_dinov3.lvd1689m",
        default_image_size=224,
        description="DINOv3 ViT-B/16 self-supervised backbone pretrained on LVD-1689M.",
    ),
    "dinov3_vitl16": BackboneSpec(
        name="dinov3_vitl16",
        family="timm",
        model_id="vit_large_patch16_dinov3.lvd1689m",
        default_image_size=224,
        description="DINOv3 ViT-L/16 self-supervised backbone pretrained on LVD-1689M.",
    ),
    "dinov3_vith16plus": BackboneSpec(
        name="dinov3_vith16plus",
        family="timm",
        model_id="vit_huge_plus_patch16_dinov3.lvd1689m",
        default_image_size=224,
        description="DINOv3 ViT-H+/16 self-supervised backbone pretrained on LVD-1689M.",
    ),
    "bioclip2_vitl14": BackboneSpec(
        name="bioclip2_vitl14",
        family="open_clip",
        model_id="hf-hub:imageomics/bioclip-2",
        default_image_size=224,
        description="BioCLIP 2 ViT-L/14 biological vision-language encoder.",
    ),
    "dinov2_vitb14": BackboneSpec(
        name="dinov2_vitb14",
        family="timm",
        model_id="vit_base_patch14_dinov2.lvd142m",
        default_image_size=224,
        description="DINOv2 ViT-B/14 baseline self-supervised backbone.",
    ),
    "dinov2_vitl14": BackboneSpec(
        name="dinov2_vitl14",
        family="timm",
        model_id="vit_large_patch14_dinov2.lvd142m",
        default_image_size=224,
        description="DINOv2 ViT-L/14 baseline self-supervised backbone.",
    ),
}


def list_supported_backbones() -> list[str]:
    """Return sorted names of registered vision backbones."""
    return sorted(BACKBONE_SPECS.keys())


def get_backbone_spec(backbone: str) -> BackboneSpec:
    """Look up a registered BackboneSpec by short name."""
    if backbone not in BACKBONE_SPECS:
        raise ValueError(
            f"Unsupported backbone {backbone!r}; expected one of {list_supported_backbones()}."
        )
    return BACKBONE_SPECS[backbone]


class BaseFeatureExtractor(ABC):
    """Common interface for frozen image feature extractors."""

    def __init__(
        self,
        backbone_name: str,
        *,
        pooling: str = "cls",
        device: str = "cpu",
        image_size: int = 224,
    ) -> None:
        if pooling not in ALLOWED_POOLING:
            raise ValueError(
                f"Unsupported pooling {pooling!r}; expected one of {sorted(ALLOWED_POOLING)}."
            )
        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        self.backbone_name = backbone_name
        self.pooling = pooling
        self.device = device
        self.image_size = image_size

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of RGB PIL images into a float32 array of shape (N, D)."""


class TimmViTExtractor(BaseFeatureExtractor):
    """Frozen timm ViT feature extractor supporting cls and patch_mean pooling."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        pooling: str = "cls",
        device: str = "cpu",
        image_size: int | None = None,
        pretrained: bool = True,
    ) -> None:
        target_size = image_size or spec.default_image_size
        super().__init__(
            spec.name,
            pooling=pooling,
            device=device,
            image_size=target_size,
        )
        import timm
        import torch

        self._torch = torch
        create_kwargs: dict[str, object] = {
            "pretrained": pretrained,
            "num_classes": 0,
        }
        if target_size != spec.default_image_size:
            create_kwargs["dynamic_img_size"] = True

        try:
            self.model = timm.create_model(spec.model_id, **create_kwargs)
        except TypeError:
            create_kwargs.pop("dynamic_img_size", None)
            self.model = timm.create_model(spec.model_id, **create_kwargs)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

        data_cfg = timm.data.resolve_model_data_config(self.model)
        data_cfg["input_size"] = (3, target_size, target_size)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Extract frozen ViT features from a batch of PIL images."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch = self._torch.stack(tensors, dim=0).to(self.device)

        with self._torch.inference_mode():
            if hasattr(self.model, "forward_features"):
                feats = self.model.forward_features(batch)
                if feats.ndim == 3:
                    if self.pooling == "patch_mean":
                        n_prefix = int(getattr(self.model, "num_prefix_tokens", 1))
                        pooled = feats[:, n_prefix:, :].mean(dim=1)
                    elif hasattr(self.model, "forward_head"):
                        pooled = self.model.forward_head(feats, pre_logits=True)
                    else:
                        pooled = feats[:, 0, :]
                else:
                    pooled = feats
            else:
                if self.pooling == "patch_mean":
                    raise ValueError(
                        f"Backbone {self.backbone_name!r} does not expose token sequence features for patch_mean."
                    )
                pooled = self.model(batch)

        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


class OpenCLIPExtractor(BaseFeatureExtractor):
    """Frozen OpenCLIP visual encoder supporting BioCLIP 2 models."""

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        pooling: str = "cls",
        device: str = "cpu",
        image_size: int | None = None,
        pretrained: bool = True,
    ) -> None:
        if pooling != "cls":
            raise ValueError(
                f"OpenCLIP backbone {spec.name!r} uses projected image embeddings and only supports pooling='cls'."
            )
        target_size = image_size or spec.default_image_size
        super().__init__(
            spec.name,
            pooling=pooling,
            device=device,
            image_size=target_size,
        )
        import open_clip
        import torch

        self._torch = torch
        model_id = spec.model_id if pretrained else "ViT-L-14"
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)

        self.model = model
        self.transform = preprocess

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode PIL images using the frozen OpenCLIP visual encoder."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch = self._torch.stack(tensors, dim=0).to(self.device)

        with self._torch.inference_mode():
            feats = self.model.encode_image(batch)

        return feats.detach().cpu().numpy().astype(np.float32, copy=False)


def create_extractor(
    backbone: str,
    *,
    pooling: str = "cls",
    device: str = "auto",
    image_size: int | None = None,
    pretrained: bool = True,
) -> BaseFeatureExtractor:
    """Instantiate a frozen feature extractor from a backbone name and config."""
    spec = get_backbone_spec(backbone)
    if pooling not in ALLOWED_POOLING:
        raise ValueError(
            f"Unsupported pooling {pooling!r}; expected one of {sorted(ALLOWED_POOLING)}."
        )
    resolved_device = resolve_device(device)

    if spec.family == "timm":
        return TimmViTExtractor(
            spec,
            pooling=pooling,
            device=resolved_device,
            image_size=image_size,
            pretrained=pretrained,
        )
    if spec.family == "open_clip":
        return OpenCLIPExtractor(
            spec,
            pooling=pooling,
            device=resolved_device,
            image_size=image_size,
            pretrained=pretrained,
        )
    raise ValueError(f"Unsupported backbone family {spec.family!r} for {backbone!r}.")
