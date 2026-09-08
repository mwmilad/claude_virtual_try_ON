"""Shared glue between FitControler and IDMVTONPipeline.run_tryon()'s
on_conditioning_ready extension point. Used by both inference.py (single-shot CLI) and
train.py (per-epoch sample grids) so there's one definition of how a conditioning dict
becomes a garment-agnostic representation + an attached injector. Also holds the
--existing-ckpt-dir resolution logic shared by both scripts' argument handling.
"""
from pathlib import Path

import torch
from torchvision.transforms.functional import to_tensor

from model import FitControler, build_garment_agnostic_input


def resolve_model_path_and_cache(args, idm_vton_inference, resolve_user_path):
    """Shared --model-path / --existing-ckpt-dir handling for inference.py and
    train.py. Both scripts define their own resolve_user_path() (paths relative to
    wherever the user invoked the script from, since idm-vton's own loader chdirs into
    third_party/IDM-VTON on import) -- pass it in rather than importing one copy, since
    it closes over each caller's own idm_vton_inference reference.

    Returns (model_path, cache_dir, local_files_only) ready to pass into
    IDMVTONPipeline(...).
    """
    model_path = args.model_path
    if not Path(model_path).is_absolute():
        candidate = resolve_user_path(model_path)
        if candidate.is_dir():
            model_path = str(candidate)

    cache_dir = None
    local_files_only = False
    existing_ckpt_dir = getattr(args, "existing_ckpt_dir", None)
    if existing_ckpt_dir:
        existing_dir = resolve_user_path(existing_ckpt_dir)
        if not existing_dir.is_dir():
            raise SystemExit(f"--existing-ckpt-dir {existing_dir} does not exist or isn't a directory")
        idm_vton_inference.link_existing_preprocessing_ckpt(existing_dir)
        cache_dir = str(existing_dir)
        local_files_only = True

    return model_path, cache_dir, local_files_only


EXISTING_CKPT_DIR_HELP = (
    "path to a directory you've already populated yourself (e.g. by running "
    "../idm-vton/scripts/download_checkpoints.py elsewhere, or any prior "
    "snapshot_download) containing densepose/, humanparsing/, openpose/ AND a "
    "models--yisol--IDM-VTON/ Hugging Face cache entry as siblings -- skips all "
    "downloading. See ../idm-vton/inference.py's --existing-ckpt-dir for the details; "
    "leave --model-path at its default 'yisol/IDM-VTON' repo id when using this."
)


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
