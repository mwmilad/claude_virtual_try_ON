#!/usr/bin/env python3
"""CLI inference for FitDiT (dual SD3-transformer virtual try-on) on a single RTX 4090.

FitDiT is a genuinely two-step pipeline, unlike OmniTry's mask-free approach:
  1. generate_mask() -- runs DWpose + human parsing on the person photo to build a
     try-on-area mask (which body region gets replaced).
  2. process() -- runs the actual dual-transformer (garment branch + person branch)
     diffusion generation inside that mask.

This script is a thin CLI wrapper around upstream's own `FitDiTGenerator` class
(imported directly from third_party/FitDiT/gradio_sd3.py) rather than a reimplementation
-- the mask/pose preprocessing has enough moving parts (DWpose, human parsing, mask
offsets) that re-deriving it independently would risk silently diverging from upstream's
actual behavior.

NON-COMMERCIAL USE ONLY: FitDiT's weights are licensed CC BY-NC-SA-4.0. See upstream's
README for commercial licensing (they point to Tencent Cloud).

Upstream's own documented memory/speed modes (from the FitDiT README) -- untested
against a specific 4090 by this script's author, no GPU access in the environment that
wrote it:
  --offload none        bf16, no CPU offload: fastest, most VRAM.
  --offload model        fp16 + enable_model_cpu_offload(): moderate speed, moderate VRAM.
  --offload aggressive   fp16 + enable_sequential_cpu_offload(): slowest, least VRAM.
Start with `none`; step down to `model` then `aggressive` if you hit CUDA OOM, and please
report back what actually fits so this note can become a real, verified table.

Example:
    python inference.py \\
        --person examples/person.jpg \\
        --garment examples/garment.jpg \\
        --category "Upper-body" \\
        --output out.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
FITDIT_SRC = REPO_ROOT / "third_party" / "FitDiT"
if str(FITDIT_SRC) not in sys.path:
    sys.path.insert(0, str(FITDIT_SRC))

try:
    from gradio_sd3 import FitDiTGenerator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Couldn't import gradio_sd3.FitDiTGenerator -- run scripts/install.sh first "
        f"(expected the upstream source at {FITDIT_SRC})"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--person", required=True, help="path to the person/model photo")
    parser.add_argument("--garment", required=True, help="path to the garment photo")
    parser.add_argument(
        "--category", required=True, choices=["Upper-body", "Lower-body", "Dresses"],
        help="which body region the garment covers",
    )
    parser.add_argument("--output", default="output.png")
    parser.add_argument(
        "--model-path", default="checkpoints",
        help="path to the downloaded FitDiT checkpoints (scripts/download_checkpoints.py default)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fp16", action="store_true",
        help="load in fp16 instead of the default bf16 (upstream default is bf16)",
    )
    parser.add_argument(
        "--offload", choices=["none", "model", "aggressive"], default="model",
        help="none = fastest/most VRAM, model = moderate (default), aggressive = "
             "slowest/least VRAM. See the module docstring -- unverified against a "
             "specific 4090, start with 'none' and step down on OOM.",
    )
    parser.add_argument(
        "--resolution", choices=["768x1024", "1152x1536", "1536x2048"], default="1152x1536",
        help="try-on processing resolution (matches upstream's default)",
    )
    parser.add_argument("--steps", type=int, default=20, help="15-30 per upstream's slider range")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="1.0-5.0 per upstream's slider range")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num-images", type=int, default=1, help="images per prompt, 1-4")
    parser.add_argument("--mask-offset-top", type=int, default=0)
    parser.add_argument("--mask-offset-bottom", type=int, default=0)
    parser.add_argument("--mask-offset-left", type=int, default=0)
    parser.add_argument("--mask-offset-right", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. This script targets an RTX 4090.")

    generator = FitDiTGenerator(
        model_root=args.model_path,
        offload=(args.offload == "model"),
        aggressive_offload=(args.offload == "aggressive"),
        device=args.device,
        with_fp16=args.fp16,
    )

    pre_mask, pose_image_pil = generator.generate_mask(
        args.person,
        args.category,
        args.mask_offset_top,
        args.mask_offset_bottom,
        args.mask_offset_left,
        args.mask_offset_right,
    )
    # Upstream's Gradio app wires generate_mask's PIL output through a gr.Image
    # component with the default type="numpy" before it reaches process(), which is why
    # process() calls Image.fromarray() on it internally. Replicate that conversion here
    # since we're calling both functions directly, with no Gradio component in between.
    pose_image = np.array(pose_image_pil)

    results = generator.process(
        args.person,
        args.garment,
        pre_mask,
        pose_image,
        args.steps,
        args.guidance_scale,
        args.seed,
        args.num_images,
        args.resolution,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(results) == 1:
        results[0].save(out_path)
        print(f"saved {out_path}")
    else:
        for i, img in enumerate(results):
            p = out_path.with_stem(f"{out_path.stem}_{i}")
            img.save(p)
            print(f"saved {p}")


if __name__ == "__main__":
    main()
