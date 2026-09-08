#!/usr/bin/env python3
"""CLI inference for FitControler-on-IDM-VTON.

**Read fitcontroler/README.md first.** This wires a from-scratch, best-effort
reconstruction of FitControler (arXiv:2512.24016 -- no official code/data released as of
when this was written) onto ../idm-vton as a plug-in, following the paper's
abstract-level description: a fit-aware layout generator + a multi-scale fit injector
(see fitcontroler/model.py).

**Without a trained --weights checkpoint, this is not useful** -- FitControler's layers
are randomly initialized, and MultiScaleFitInjector is a ControlNet-style zero-init
design, so an untrained plug-in is an exact no-op: `--fit` will silently have zero effect
and this script will just reproduce plain IDM-VTON output (with the overhead of also
running the layout generator for nothing). There is no pretrained FitControler checkpoint
to download; see fitcontroler/train.py to train one against your own fit-labeled data.

Example (requires a trained checkpoint to do anything beyond vanilla IDM-VTON):
    python inference.py \\
        --person ../idm-vton/examples/person.jpg \\
        --garment ../idm-vton/examples/garment.jpg \\
        --garment-desc "short sleeve round neck t-shirt" \\
        --fit loose \\
        --weights checkpoints/fitcontroler.pt \\
        --output out.png
"""
import argparse
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parent
IDM_VTON_DIR = REPO_ROOT.parent / "idm-vton"
if str(IDM_VTON_DIR) not in sys.path:
    sys.path.insert(0, str(IDM_VTON_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# INVOCATION_DIR is captured (and cwd changed to third_party/IDM-VTON) as a side effect
# of importing idm-vton's inference module -- see idm-vton/inference.py.
import inference as idm_vton_inference  # noqa: E402
from model import (  # noqa: E402
    FIT_LEVELS,
    FitControler,
    build_garment_agnostic_input,
    fit_level_to_index,
)


def resolve_user_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (idm_vton_inference.INVOCATION_DIR / path)


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--garment-desc", required=True)
    parser.add_argument("--fit", choices=FIT_LEVELS, default="regular", help="target fit level")
    parser.add_argument(
        "--weights", default=None,
        help="path to a trained FitControler checkpoint (see train.py). Without this, "
             "the plug-in is randomly initialized / a zero-init no-op -- output will "
             "match plain IDM-VTON regardless of --fit (a warning is printed).",
    )
    parser.add_argument("--output", default="output.png")
    parser.add_argument(
        "--model-path", default="yisol/IDM-VTON",
        help="IDM-VTON base checkpoint -- see ../idm-vton/README.md",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-type", choices=["hd", "dc"], default="hd")
    parser.add_argument("--category", choices=["upper_body", "lower_body", "dresses"], default="upper_body")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. This script targets an RTX 4090.")

    if args.weights is None:
        warnings.warn(
            "No --weights given: FitControler's layout generator is randomly "
            "initialized and its injector is zero-init, so it is an exact no-op right "
            "now. This run will produce the same output as plain idm-vton/inference.py "
            "regardless of --fit. See fitcontroler/train.py to train a real checkpoint.",
            stacklevel=1,
        )

    model_path = args.model_path
    if not Path(model_path).is_absolute():
        candidate = resolve_user_path(model_path)
        if candidate.is_dir():
            model_path = str(candidate)

    pipeline = idm_vton_inference.IDMVTONPipeline(model_path, args.device)

    if args.weights is not None:
        fit_controler = FitControler.load(resolve_user_path(args.weights), map_location=args.device)
    else:
        fit_controler = FitControler()
    fit_controler.to(args.device)
    fit_controler.eval()

    person_img = Image.open(resolve_user_path(args.person))
    garm_img = Image.open(resolve_user_path(args.garment))
    fit_level_idx = fit_level_to_index(args.fit)

    hook = make_conditioning_hook(
        fit_controler, pipeline.pipe.unet, fit_level_idx, args.device, pipeline.dtype
    )
    try:
        result, mask_gray = pipeline.run_tryon(
            person_img=person_img,
            garm_img=garm_img,
            garment_desc=args.garment_desc,
            model_type=args.model_type,
            category=args.category,
            auto_mask=True,
            user_mask=None,
            crop=args.crop,
            denoise_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=None if args.seed < 0 else args.seed,
            on_conditioning_ready=hook,
        )
    finally:
        fit_controler.detach()

    out_path = resolve_user_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
