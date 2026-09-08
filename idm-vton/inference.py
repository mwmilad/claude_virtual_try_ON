#!/usr/bin/env python3
"""CLI inference for IDM-VTON (SDXL-inpainting-based virtual try-on) on a single RTX 4090.

IDM-VTON is a genuinely multi-stage pipeline:
  1. OpenPose keypoints + human parsing on the person photo -> an auto-generated try-on
     mask (which pixels get replaced), via upstream's own `get_mask_location`.
  2. DensePose (via a vendored detectron2 + densepose, run through upstream's
     `apply_net.py` 'show' action) -> a pose/segmentation conditioning image.
  3. A dual-UNet SDXL-inpainting pipeline (one UNet branch for the garment, one for the
     person/scene) does the actual diffusion generation inside the mask.

NON-COMMERCIAL USE ONLY: IDM-VTON's weights are licensed CC BY-NC-SA-4.0.

Honesty note, matching this repo's convention for the other two models: the model
construction block, `pil_to_binary_mask`, `tensor_transfrom`, and the body of
`run_tryon()` below are adapted -- verbatim where possible -- from upstream's own
`start_tryon()` callback in `gradio_demo/app.py` (fetched and quoted directly from
https://github.com/yisol/IDM-VTON, commit as of 2026-09), not independently
re-derived, so the actual preprocessing/diffusion math shouldn't silently diverge
from upstream's real behavior. It is restructured into a plain CLI function instead
of importing app.py directly because app.py constructs its Gradio Blocks UI at module
level with no `if __name__ == "__main__":` guard visible in what could be fetched, so
importing it as a module risks launching a web server as a side effect.

**This has NOT been run on real 4090 hardware by whoever wrote this wrapper** (no GPU
access in the environment that authored it) -- please report back what actually works,
especially around the vendored detectron2 build (see README.md Troubleshooting).

Example:
    python inference.py \\
        --person examples/person.jpg \\
        --garment examples/garment.jpg \\
        --garment-desc "short sleeve round neck t-shirt" \\
        --output out.png
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
IDM_VTON_SRC = REPO_ROOT / "third_party" / "IDM-VTON"
GRADIO_DEMO_SRC = IDM_VTON_SRC / "gradio_demo"

for p in (IDM_VTON_SRC, GRADIO_DEMO_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Remember where the user invoked us from, so --person/--garment/--output/--mask (given
# as relative paths) still resolve against *that* directory after the chdir below.
INVOCATION_DIR = Path.cwd()

# Upstream's own relative paths (./ckpt/..., ./configs/...) are resolved against the
# process's cwd, not against this script's location -- chdir to match how
# `python gradio_demo/app.py` behaves when run from the IDM-VTON repo root.
if IDM_VTON_SRC.exists():
    os.chdir(IDM_VTON_SRC)


def resolve_user_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (INVOCATION_DIR / path)

try:
    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from transformers import (
        AutoTokenizer,
        CLIPImageProcessor,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )
    from diffusers import AutoencoderKL, DDPMScheduler

    from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
    from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
    from src.unet_hacked_tryon import UNet2DConditionModel

    from utils_mask import get_mask_location
    from preprocess.humanparsing.run_parsing import Parsing
    from preprocess.openpose.run_openpose import OpenPose
    from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
    import apply_net
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Couldn't import IDM-VTON's upstream modules -- run scripts/install.sh first "
        f"(expected the upstream source at {IDM_VTON_SRC}, including a built vendored "
        "detectron2). See README.md Troubleshooting if the detectron2 build failed."
    ) from exc


def pil_to_binary_mask(pil_image: Image.Image, threshold: int = 0) -> Image.Image:
    """Verbatim port of gradio_demo/app.py's pil_to_binary_mask."""
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = (binary_mask.astype(np.uint8) * 255)
    return Image.fromarray(mask)


class IDMVTONPipeline:
    """Loads the model once; mirrors the module-level setup block of gradio_demo/app.py."""

    def __init__(self, model_path: str, device: str, dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype)
        unet.requires_grad_(False)
        tokenizer_one = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=None, use_fast=False)
        tokenizer_two = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", revision=None, use_fast=False)
        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        text_encoder_one = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=dtype)
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder_2", torch_dtype=dtype)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", torch_dtype=dtype)
        unet_encoder = UNet2DConditionModel_ref.from_pretrained(model_path, subfolder="unet_encoder", torch_dtype=dtype)

        for m in (unet_encoder, image_encoder, vae, unet, text_encoder_one, text_encoder_two):
            m.requires_grad_(False)

        self.parsing_model = Parsing(0 if device.startswith("cuda") else -1)
        self.openpose_model = OpenPose(0 if device.startswith("cuda") else -1)

        self.tensor_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.pipe = TryonPipeline.from_pretrained(
            model_path,
            unet=unet,
            vae=vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            tokenizer=tokenizer_one,
            tokenizer_2=tokenizer_two,
            scheduler=noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=dtype,
        )
        self.pipe.unet_encoder = unet_encoder

        self.pipe.to(device)
        self.pipe.unet_encoder.to(device)
        self.openpose_model.preprocessor.body_estimation.model.to(device)

    def run_densepose(self, human_img: Image.Image) -> Image.Image:
        """Runs upstream's DensePose 'show' action (via apply_net.py) on a 768x1024
        person image, exactly as start_tryon() does. Factored out of run_tryon() so
        other callers (e.g. ../fitcontroler/train.py, which needs the same DensePose
        conditioning image when preprocessing training data) can reuse it without
        duplicating the apply_net argument construction."""
        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        densepose_args = apply_net.create_argument_parser().parse_args(
            (
                "show",
                "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                "./ckpt/densepose/model_final_162be9.pkl",
                "dp_segm",
                "-v",
                "--opts",
                "MODEL.DEVICE",
                self.device if self.device.startswith("cuda") else "cpu",
            )
        )
        pose_img = densepose_args.func(densepose_args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        return Image.fromarray(pose_img).resize((768, 1024))

    def run_tryon(
        self,
        person_img: Image.Image,
        garm_img: Image.Image,
        garment_desc: str,
        model_type: str,
        category: str,
        auto_mask: bool,
        user_mask: Image.Image | None,
        crop: bool,
        denoise_steps: int,
        guidance_scale: float,
        seed: int | None,
        on_conditioning_ready=None,
    ) -> tuple[Image.Image, Image.Image]:
        device, dtype = self.device, self.dtype

        garm_img = garm_img.convert("RGB").resize((768, 1024))
        human_img_orig = person_img.convert("RGB")

        if crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768, 1024))
        else:
            human_img = human_img_orig.resize((768, 1024))

        if auto_mask:
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location(model_type, category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
        else:
            mask = pil_to_binary_mask(user_mask.convert("RGB").resize((768, 1024)))
        mask_gray_tensor = (1 - self.tensor_transform(mask)) * self.tensor_transform(human_img)
        mask_gray = to_pil_image((mask_gray_tensor + 1.0) / 2.0)

        pose_img = self.run_densepose(human_img)

        if on_conditioning_ready is not None:
            # Extension point for plug-ins (e.g. ../fitcontroler) that need to condition
            # on the same mask/pose/parsing outputs this pipeline already computed,
            # before the diffusion pass runs. Not used by plain IDM-VTON inference.
            on_conditioning_ready(
                {
                    "human_img": human_img,
                    "mask": mask,
                    "mask_gray": mask_gray,
                    "pose_img": pose_img,
                    "model_parse": model_parse if auto_mask else None,
                    "keypoints": keypoints if auto_mask else None,
                    "tensor_transform": self.tensor_transform,
                    "device": device,
                    "dtype": dtype,
                }
            )

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            prompt = "model is wearing " + garment_desc
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.pipe.encode_prompt(
                    prompt, num_images_per_prompt=1, do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt_c = "a photo of " + garment_desc
                with torch.inference_mode():
                    prompt_embeds_c, _, _, _ = self.pipe.encode_prompt(
                        [prompt_c], num_images_per_prompt=1, do_classifier_free_guidance=False,
                        negative_prompt=[negative_prompt],
                    )

                pose_img_t = self.tensor_transform(pose_img).unsqueeze(0).to(device, dtype)
                garm_tensor = self.tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                images = self.pipe(
                    prompt_embeds=prompt_embeds.to(device, dtype),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img_t,
                    text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                    cloth=garm_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=guidance_scale,
                )[0]

        if crop:
            out_img = images[0].resize(crop_size)
            result = human_img_orig.copy()
            result.paste(out_img, (int(left), int(top)))
            return result, mask_gray
        return images[0], mask_gray


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", required=True, help="path to the person/model photo")
    parser.add_argument("--garment", required=True, help="path to the garment photo")
    parser.add_argument(
        "--garment-desc", required=True,
        help="short text description of the garment, e.g. 'short sleeve round neck t-shirt' "
             "-- fed into the prompt exactly as upstream does ('model is wearing <desc>')",
    )
    parser.add_argument("--output", default="output.png")
    parser.add_argument(
        "--model-path", default="yisol/IDM-VTON",
        help="local checkpoints dir (scripts/download_checkpoints.py's checkpoints/IDM-VTON) "
             "or a bare HF repo id to load straight from the Hub (default)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-type", choices=["hd", "dc"], default="hd",
        help="'hd' = VITON-HD-style masking (upstream's own gradio demo default), "
             "'dc' = DressCode-style (needed for --category lower_body/dresses)",
    )
    parser.add_argument("--category", choices=["upper_body", "lower_body", "dresses"], default="upper_body")
    parser.add_argument(
        "--mask", default=None,
        help="optional path to a hand-drawn binary mask image; if omitted, the mask is "
             "auto-generated from OpenPose + human parsing (upstream's default 'auto mask' path)",
    )
    parser.add_argument("--crop", action="store_true", help="auto-crop to a 3:4 region before try-on (upstream's 'crop & resize' option)")
    parser.add_argument("--steps", type=int, default=30, help="denoising steps (upstream's own inference.py CLI default)")
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--save-mask", default=None, help="optional path to also save the (grayed-out) mask preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. This script targets an RTX 4090.")

    # A relative --model-path that exists on disk (e.g. checkpoints/IDM-VTON, the
    # download script's default output) needs resolving against the invocation dir, same
    # as the image paths -- but a bare HF repo id like the default "yisol/IDM-VTON" isn't
    # a real local path, so leave anything that doesn't resolve to an existing dir as-is.
    model_path = args.model_path
    if not Path(model_path).is_absolute():
        candidate = resolve_user_path(model_path)
        if candidate.is_dir():
            model_path = str(candidate)

    pipeline = IDMVTONPipeline(model_path, args.device)

    person_img = Image.open(resolve_user_path(args.person))
    garm_img = Image.open(resolve_user_path(args.garment))
    user_mask = Image.open(resolve_user_path(args.mask)) if args.mask else None

    result, mask_gray = pipeline.run_tryon(
        person_img=person_img,
        garm_img=garm_img,
        garment_desc=args.garment_desc,
        model_type=args.model_type,
        category=args.category,
        auto_mask=user_mask is None,
        user_mask=user_mask,
        crop=args.crop,
        denoise_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=None if args.seed < 0 else args.seed,
    )

    out_path = resolve_user_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"saved {out_path}")

    if args.save_mask:
        mask_path = resolve_user_path(args.save_mask)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_gray.save(mask_path)
        print(f"saved {mask_path}")


if __name__ == "__main__":
    main()
