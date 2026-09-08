"""Shared glue between FitControler and IDMVTONPipeline.run_tryon()'s
on_conditioning_ready extension point. Used by both inference.py (single-shot CLI) and
train.py (per-epoch sample grids) so there's one definition of how a conditioning dict
becomes a garment-agnostic representation + an attached injector.
"""
import torch
from torchvision.transforms.functional import to_tensor

from model import FitControler, build_garment_agnostic_input


def make_conditioning_hook(fit_controler: FitControler, unet, fit_level_idx: int, device: str, dtype):
    """Returns the on_conditioning_ready callback IDMVTONPipeline.run_tryon() invokes
    right after it has computed mask/pose_img/mask_gray -- builds the garment-agnostic
    representation from those same outputs (see model.build_garment_agnostic_input's
    docstring for why these specific pieces) and attaches the plug-in's injector hooks
    onto the base pipeline's UNet before the diffusion pass runs."""

    def _hook(conditioning: dict) -> None:
        tt = conditioning["tensor_transform"]  # ToTensor + Normalize([0.5],[0.5]) -> ~[-1,1]
        mask_t = to_tensor(conditioning["mask"])[:1]  # keep raw [0,1] range, single channel
        agnostic_repr = build_garment_agnostic_input(
            agnostic_image=tt(conditioning["mask_gray"]).unsqueeze(0).to(device, dtype),
            default_mask=mask_t.unsqueeze(0).to(device, dtype),
            densepose_image=tt(conditioning["pose_img"]).unsqueeze(0).to(device, dtype),
        )
        fit_level = torch.tensor([fit_level_idx], device=device)
        fit_controler.prepare(agnostic_repr, fit_level)
        fit_controler.attach(unet)

    return _hook
