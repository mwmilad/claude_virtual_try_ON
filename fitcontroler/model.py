"""A best-effort reconstruction of FitControler (Yang, Liu, Li, Bai, Lu --
"FitControler: Toward Fit-Aware Virtual Try-On", arXiv:2512.24016, Dec 2025).

**Read fitcontroler/README.md before trusting anything in this file.** In short: the
paper's own code/data ("Fit4Men", ~13k body-garment pairs) is not publicly released as
of when this was written, and the paper's full text was not directly fetchable either
(arXiv and Hugging Face were both blocked by this environment's network policy) -- this
module was written from abstract-level search-engine summaries plus GitHub-hosted source
of the base VTON model (IDM-VTON), not from the paper's equations, figures, or ablations.
It is a plausible, from-scratch architecture matching the two components the abstract
names -- a **fit-aware layout generator** and a **multi-scale fit injector** -- built to
be trainable and pluggable, NOT a verified reproduction of the paper's actual numbers or
exact design choices (channel counts, exact conditioning mechanism, exact loss terms are
all invented here). Weights are randomly initialized; there is no pretrained checkpoint.
See fitcontroler/README.md's "What's faithful vs. invented" table for the full breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# Invented here: the paper's abstract doesn't specify a fit vocabulary. A 5-level
# ordinal scale (garment-industry-standard "ease" categories) is a reasonable stand-in
# for whatever discrete/continuous fit signal Fit4Men actually used.
FIT_LEVELS = ("tight", "fitted", "regular", "loose", "oversized")


def fit_level_to_index(name: str) -> int:
    try:
        return FIT_LEVELS.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown fit level {name!r}, expected one of {FIT_LEVELS}") from exc


@dataclass
class FitControlerConfig:
    """Hyperparameters for FitControler. Saved alongside weights so a checkpoint is
    self-describing (see save()/load() in this module)."""

    in_channels: int = 7  # agnostic RGB (3) + current/default mask (1) + densepose viz (3)
    base_channels: int = 64
    num_stages: int = 3  # matches SDXL UNet's 3 down_blocks (320/640/1280 channels)
    backbone_channels: Sequence[int] = field(default_factory=lambda: (320, 640, 1280))
    num_fit_levels: int = len(FIT_LEVELS)
    cond_dim: int = 128


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = min(32, out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLM(nn.Module):
    """Feature-wise linear modulation -- how the fit-condition embedding steers the
    layout generator's features at each resolution. The paper doesn't specify its
    conditioning mechanism at the abstract level; FiLM is a standard, cheap choice for
    injecting a discrete/global condition into a conv encoder-decoder."""

    def __init__(self, cond_dim: int, out_ch: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class FitAwareLayoutGenerator(nn.Module):
    """Redraws the body-garment layout conditioned on a garment-agnostic representation
    of the person plus a target fit level -- the first of FitControler's two named
    components. A compact FiLM-conditioned U-Net: encoder-decoder with skip connections,
    modulated by the fit embedding at every resolution.

    Exposes its decoder's per-stage feature maps (finest resolution first) for
    MultiScaleFitInjector to deliver into a frozen VTON backbone.
    """

    def __init__(self, config: FitControlerConfig):
        super().__init__()
        self.config = config
        self.fit_embedding = nn.Embedding(config.num_fit_levels, config.cond_dim)

        chs = [config.base_channels * (2 ** i) for i in range(config.num_stages)]
        self.stage_channels = chs  # finest -> coarsest, e.g. [64, 128, 256]

        self.stem = ConvBlock(config.in_channels, chs[0])
        self.down_blocks = nn.ModuleList()
        self.down_films = nn.ModuleList()
        for i in range(config.num_stages - 1):
            self.down_blocks.append(ConvBlock(chs[i], chs[i + 1]))
            self.down_films.append(FiLM(config.cond_dim, chs[i + 1]))

        self.up_blocks = nn.ModuleList()
        self.up_films = nn.ModuleList()
        for i in reversed(range(config.num_stages - 1)):
            self.up_blocks.append(ConvBlock(chs[i + 1] + chs[i], chs[i]))
            self.up_films.append(FiLM(config.cond_dim, chs[i]))

        self.layout_head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, agnostic_repr: torch.Tensor, fit_level: torch.Tensor):
        """
        agnostic_repr: (B, in_channels, H, W).
        fit_level: (B,) long tensor of indices into FIT_LEVELS.

        Returns (layout_logits: (B,1,H,W), multi_scale_features: list[Tensor], finest
        resolution first -- e.g. [64ch@H, 128ch@H/2, 256ch@H/4] for the default config).
        """
        cond = self.fit_embedding(fit_level)

        x = self.stem(agnostic_repr)
        skips = [x]
        for block, film in zip(self.down_blocks, self.down_films):
            x = F.avg_pool2d(x, 2)
            x = film(block(x), cond)
            skips.append(x)

        decoder_outputs = [skips[-1]]
        x = skips[-1]
        for block, film, skip in zip(self.up_blocks, self.up_films, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = film(block(torch.cat([x, skip], dim=1)), cond)
            decoder_outputs.append(x)

        layout_logits = self.layout_head(x)
        multi_scale_features = list(reversed(decoder_outputs))  # finest first
        return layout_logits, multi_scale_features


class MultiScaleFitInjector(nn.Module):
    """Delivers the layout generator's multi-scale features into a frozen VTON UNet --
    the second of FitControler's two named components ("multi-scale fit injector...to
    enable layout-driven VTON"). Implemented ControlNet-style: one zero-initialized 1x1
    conv per injection point, so an untrained injector is an exact no-op and training
    only ever perturbs the frozen backbone additively, never destructively, from the
    start of training.
    """

    def __init__(self, source_channels: Sequence[int], target_channels: Sequence[int]):
        super().__init__()
        if len(source_channels) != len(target_channels):
            raise ValueError(
                f"source/target channel lists must be the same length "
                f"({len(source_channels)} vs {len(target_channels)}) -- one entry per "
                f"backbone injection point"
            )
        self.projections = nn.ModuleList()
        for c_in, c_out in zip(source_channels, target_channels):
            conv = nn.Conv2d(c_in, c_out, 1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            self.projections.append(conv)

    def forward(self, multi_scale_features: Sequence[torch.Tensor], target_shapes: Sequence[torch.Size]):
        """One residual tensor per injection point, resized to target_shapes and projected."""
        residuals = []
        for feat, proj, shape in zip(multi_scale_features, self.projections, target_shapes):
            feat = F.interpolate(feat, size=shape, mode="bilinear", align_corners=False)
            residuals.append(proj(feat))
        return residuals


class FitControler(nn.Module):
    """Top-level plug-in: owns the layout generator + injector, and manages attaching
    them to a frozen diffusers UNet2DConditionModel via forward hooks on its
    `down_blocks`. Mirrors how ControlNet-style plug-ins are conventionally wired into
    diffusers pipelines -- additive residuals on frozen backbone activations, backbone
    weights untouched.

    Usage:
        fc = FitControler(FitControlerConfig())
        fc.load_state_dict(torch.load("fitcontroler.pt"))  # once you've trained one
        layout = fc.prepare(agnostic_repr, fit_level)   # call once per generation
        fc.attach(pipe.unet)                              # before pipe(...)
        ... run the base VTON pipeline's forward pass ...
        fc.detach()                                        # after pipe(...)
    """

    def __init__(self, config: FitControlerConfig | None = None):
        super().__init__()
        self.config = config or FitControlerConfig()
        self.layout_generator = FitAwareLayoutGenerator(self.config)
        self.injector = MultiScaleFitInjector(
            self.layout_generator.stage_channels, list(self.config.backbone_channels)
        )
        self._hooks: list = []
        self._cached_features: list[torch.Tensor] | None = None

    @torch.no_grad()
    def prepare(self, agnostic_repr: torch.Tensor, fit_level: torch.Tensor) -> torch.Tensor:
        """Runs the layout generator and caches its multi-scale features for the next
        attached UNet forward pass. Returns the predicted layout as a (B,1,H,W)
        probability map (sigmoid of the raw logits) for preview/debugging."""
        layout_logits, feats = self.layout_generator(agnostic_repr, fit_level)
        self._cached_features = feats
        return torch.sigmoid(layout_logits)

    def prepare_for_training(self, agnostic_repr: torch.Tensor, fit_level: torch.Tensor):
        """Same as prepare() but keeps gradients (for train.py); returns raw logits,
        not sigmoid, so a caller can use BCEWithLogitsLoss directly."""
        layout_logits, feats = self.layout_generator(agnostic_repr, fit_level)
        self._cached_features = feats
        return layout_logits, feats

    def attach(self, unet: nn.Module) -> "FitControler":
        """Registers a forward hook on each of `unet.down_blocks[:num_stages-1]` that
        adds the corresponding injected residual to that block's output hidden states.
        Call prepare()/prepare_for_training() before the UNet forward pass this is meant
        to affect -- the hooks read whatever was cached most recently."""
        if not hasattr(unet, "down_blocks"):
            raise TypeError(
                "FitControler.attach() expects a diffusers-style UNet2DConditionModel "
                "with a `down_blocks` attribute (e.g. IDM-VTON's TryonNet unet)"
            )
        self.detach()
        num_points = len(self.injector.projections)
        for level, block in enumerate(unet.down_blocks[:num_points]):
            self._hooks.append(block.register_forward_hook(self._make_hook(level)))
        return self

    def _make_hook(self, level: int):
        def _hook(module, inputs, output):
            if self._cached_features is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            feat = F.interpolate(
                self._cached_features[level], size=hidden.shape[-2:], mode="bilinear", align_corners=False
            )
            hidden = hidden + self.injector.projections[level](feat)
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        return _hook

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._cached_features = None

    def save(self, path: str) -> None:
        torch.save({"config": self.config.__dict__, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "FitControler":
        checkpoint = torch.load(path, map_location=map_location)
        config = FitControlerConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model


def build_garment_agnostic_input(
    agnostic_image: torch.Tensor,
    default_mask: torch.Tensor,
    densepose_image: torch.Tensor,
) -> torch.Tensor:
    """Assembles the "garment-agnostic representation" fed to the layout generator.

    The paper says this representation is "delicately processed" without detailing how
    -- this is a plain, defensible stand-in: the same garment-agnostic image and default
    auto-generated mask IDM-VTON's own mask stage already produces (see
    idm-vton/inference.py's run_tryon, `mask_gray` and `mask`), plus the DensePose
    visualization for body-shape cues, simply concatenated on the channel axis.

    agnostic_image: (B,3,H,W) in [-1,1] (e.g. IDM-VTON's mask_gray, renormalized).
    default_mask:   (B,1,H,W) in [0,1] (e.g. IDM-VTON's auto-generated `mask`).
    densepose_image:(B,3,H,W) in [-1,1] (e.g. IDM-VTON's `pose_img` tensor).
    """
    return torch.cat([agnostic_image, default_mask, densepose_image], dim=1)
