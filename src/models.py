"""ResNet-18 and DeiT-tiny feature extractors with forward hooks.

ResNet-18 layers (global-average-pooled to feature vectors):
    layer1 -> (B, 64)
    layer2 -> (B, 128)
    layer3 -> (B, 256)
    layer4 -> (B, 512)
    avgpool -> (B, 512)

DeiT-tiny features (all 192-dim):
    cls_token   - CLS embedding after the final norm
    patch_mean  - mean of patch tokens after the final norm
    cls_block6  - CLS embedding after transformer block 6
    block{2,5,8,11} - CLS embedding after the indicated block
"""

import timm
import torch
import torch.nn as nn
import torchvision.models as tv_models


class ResNetExtractor(nn.Module):
    """Pretrained ResNet-18 with intermediate layer outputs exposed via hooks."""

    LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4", "avgpool")

    def __init__(self, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tv_models.resnet18(weights=weights)
        self.backbone.eval()

        self._gap = nn.AdaptiveAvgPool2d(1)
        self._cache = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name in self.LAYER_NAMES:
            module = getattr(self.backbone, name)
            handle = module.register_forward_hook(self._hook(name))
            self._handles.append(handle)

    def _hook(self, name):
        def fn(module, inputs, output):
            if output.dim() == 4:
                self._cache[name] = self._gap(output).flatten(1)
            else:
                self._cache[name] = output.flatten(1)
        return fn

    @torch.no_grad()
    def forward(self, x):
        self._cache.clear()
        self.backbone(x)
        return {k: v.cpu() for k, v in self._cache.items()}

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class DeiTExtractor(nn.Module):
    """Pretrained DeiT-tiny with per-block CLS outputs exposed via hooks."""

    BLOCK_INDICES = (2, 5, 8, 11)

    def __init__(self, model_name="deit_tiny_patch16_224", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.backbone.eval()

        self._cache = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Final norm produces (B, 197, 192): index 0 is CLS, 1: are patch tokens.
        self._handles.append(
            self.backbone.norm.register_forward_hook(self._hook("_norm_out"))
        )
        # Per-block hooks for depth analysis.
        for idx in (6,) + self.BLOCK_INDICES:
            self._handles.append(
                self.backbone.blocks[idx].register_forward_hook(
                    self._hook(f"_block{idx}_out")
                )
            )

    def _hook(self, name):
        def fn(module, inputs, output):
            self._cache[name] = output
        return fn

    @torch.no_grad()
    def forward(self, x):
        self._cache.clear()
        self.backbone(x)

        norm_out = self._cache["_norm_out"]
        b6_out = self._cache["_block6_out"]

        features = {
            "cls_token": norm_out[:, 0].cpu(),
            "patch_mean": norm_out[:, 1:].mean(dim=1).cpu(),
            "cls_block6": b6_out[:, 0].cpu(),
        }
        for idx in self.BLOCK_INDICES:
            key = f"_block{idx}_out"
            if key in self._cache:
                features[f"block{idx}"] = self._cache[key][:, 0].cpu()
        return features

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
