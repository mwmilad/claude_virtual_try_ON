#!/usr/bin/env python3
"""Training for FitControler-on-IDM-VTON. Read this whole docstring first.

Default dataset: **GarmentCodeVTON** (huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset,
from the FitVTON paper, arXiv:2606.12012) -- 78,080 synthetic try-on triplets, used as a
real, downloadable stand-in for FitControler's own (unreleased) Fit4Men dataset. Run
`python scripts/download_dataset.py` first, then `python train.py` -- see "Usage" below.
See fit_datasets.py's module docstring for exactly what's verified vs. guessed about that
dataset's schema (huggingface.co was unreachable from the environment that wrote this,
so column names are auto-detected/best-effort, not confirmed).

**Still not verified on real hardware end-to-end** -- no GPU access in the environment
that authored this. What *is* grounded, deliberately: the VTON-diffusion training step
(VAE encode -> add noise -> garment/reference UNet call -> main UNet call -> MSE loss) is
adapted from upstream IDM-VTON's own `train_xl.py` (fetched and quoted directly from
https://github.com/yisol/IDM-VTON), not guessed, so if the frozen base pipeline loads and
the dataset's columns resolve correctly, that part of the forward pass should match how
IDM-VTON actually expects to be driven. The most likely real-run failure points are (a) a
GarmentCodeVTONDataset column-name mismatch (see fit_datasets.py --inspect) and (b) shapes/
dtypes that only reveal themselves against the real IDM-VTON pipeline and real images,
neither of which were available to test against here.

Design decision (invented -- not from the paper, which doesn't specify its training
recipe at the abstract level available here): freeze the *entire* base IDM-VTON pipeline
(unet, unet_encoder, vae, text encoders, image_encoder) and train only FitControler's own
parameters (layout generator + injector). This is the standard ControlNet-style plug-in
training paradigm, and matches the paper's framing of FitControler as something that
"seamlessly integrates into modern VTON models" without retraining them -- but note it is
*different* from upstream IDM-VTON's own train_xl.py, which fine-tunes the main `unet`
itself (see its own frozen list: vae, text_encoder(_2), image_encoder, unet_encoder).

Two losses (also invented -- the paper's actual loss terms/weights aren't available):
  - `layout_loss`: BCE between the layout generator's predicted fit-adjusted mask and a
    ground-truth one. For GarmentCodeVTON, the "ground truth" is derived at batch-prep
    time by running IDM-VTON's own mask stage on the *fitted-result* image instead of
    the pre-garment avatar (see prepare_batch()'s docstring for why) -- not an official
    label, a mechanically-grounded proxy for one.
  - `diffusion_loss`: the same noise-prediction MSE IDM-VTON itself trains with, computed
    with FitControler's injector attached to the frozen main unet -- gradients flow back
    only into FitControler's parameters, everything else stays frozen.

Usage:
    python scripts/download_dataset.py                 # one-time: pulls GarmentCodeVTON
    python train.py --output-dir checkpoints            # trains against it

Fit4Men itself (if you have real access to it, or another dataset in the same shape) is
still supported via --dataset fit4men --data-root <path> -- see Fit4MenDataset below,
which remains a documented format stub (Fit4Men isn't public).

Per-epoch sample grid: unless --no-epoch-samples is passed, the end of every epoch runs
a fixed (avatar, garment) pair through the current FitControler checkpoint once per fit
level (model.FIT_LEVELS -- 5 by default: tight/fitted/regular/loose/oversized), stitches
the 5 results into one labeled image, and saves it to <output-dir>/samples/epoch_NNN.png
-- a quick visual read on whether the plug-in is learning to differentiate fits at all,
without needing an eval harness. See save_epoch_sample_grid() below. Only implemented
for --dataset garmentcode_vton (it needs a raw PIL/text sample; Fit4MenDataset's stub
returns preprocessed tensors, not that, so sampling is skipped with a note for
--dataset fit4men).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Loaded by explicit file path, not `import inference` -- see _idm_vton.py's docstring
# for the real bug (a self-import via sys.path ordering) this sidesteps.
from _idm_vton import load as _load_idm_vton_inference  # noqa: E402

idm_vton_inference = _load_idm_vton_inference()
from model import FIT_LEVELS, FitControler, FitControlerConfig, build_garment_agnostic_input  # noqa: E402
from fit_datasets import GarmentCodeVTONDataset, parse_fit_level_from_text, raw_collate  # noqa: E402
from hooks import make_conditioning_hook  # noqa: E402


@dataclass
class Fit4MenSample:
    """One (person, garment, target fit) training example. Field shapes assume a
    single item has already been collated by a DataLoader into a batch dimension."""

    image: torch.Tensor  # (B,3,H,W) target/ground-truth try-on photo, in [-1,1] -- for VAE encode
    agnostic_image: torch.Tensor  # (B,3,H,W) garment-agnostic person image, in [-1,1]
    default_mask: torch.Tensor  # (B,1,H,W) IDM-VTON's own auto-generated mask, in [0,1]
    densepose_image: torch.Tensor  # (B,3,H,W) DensePose visualization, in [-1,1]
    target_layout: torch.Tensor  # (B,1,H,W) ground-truth fit-adjusted mask, in [0,1] -- THE Fit4Men label
    fit_level: torch.Tensor  # (B,) long indices into model.FIT_LEVELS
    cloth: torch.Tensor  # (B,3,H,W) garment photo tensor, for unet_encoder
    text_embeds_cloth: torch.Tensor  # (B, seq, dim) pooled/sequence garment text embedding
    encoder_hidden_states: torch.Tensor  # (B, seq, dim) main prompt embedding
    unet_added_cond_kwargs: dict  # SDXL micro-conditioning kwargs (text_embeds, time_ids)


class Fit4MenDataset(Dataset):
    """Stub for the paper's Fit4Men dataset. There is nothing to load: the real dataset
    is not public. Point `root` at your own fit-labeled VTON data laid out however you
    like, and implement `__getitem__` to produce a `Fit4MenSample` -- the fields above
    are exactly what `training_step()` consumes."""

    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"{root} does not exist. Fit4Men is not publicly released -- point this "
                "at your own fit-labeled dataset laid out per Fit4MenSample's fields, "
                "and implement __getitem__ below."
            )

    def __len__(self) -> int:
        raise NotImplementedError("wire this up to your own dataset's index/manifest")

    def __getitem__(self, idx: int) -> Fit4MenSample:
        raise NotImplementedError(
            "Fit4Men is not publicly released as of when this was written -- implement "
            "this against your own fit-labeled data, producing the fields documented on "
            "Fit4MenSample. See fitcontroler/README.md for what each field is for."
        )


def prepare_batch(
    pipeline: "idm_vton_inference.IDMVTONPipeline",
    raw_batch: list,
) -> Fit4MenSample:
    """Turns a list of GarmentCodeVTONDataset raw samples (dicts of PIL images + text,
    see fit_datasets.py) into a Fit4MenSample batch, by running IDM-VTON's own
    preprocessing on each sample -- the same building blocks
    idm-vton/inference.py's run_tryon() uses (human parsing, OpenPose, DensePose, prompt
    encoding), just called directly here instead of through the CLI pipeline.

    `target_layout` (what FitAwareLayoutGenerator is supervised against) is derived by
    running the SAME mask stage (get_mask_location) on the *fitted result* image, not
    the pre-garment avatar: get_mask_location's output there traces how far the parsing
    model sees "upper-clothes" pixels extending on the body under that specific
    simulated fit -- wider for a loose garment, narrower for a tight one -- which is a
    mechanically-grounded proxy for a "ground-truth fit-adjusted layout." It is NOT a
    verified match for how Fit4Men itself was actually annotated (unknown, unreleased).
    """
    from utils_mask import get_mask_location  # resolved via idm-vton's sys.path setup

    device, dtype = pipeline.device, pipeline.dtype
    tt = pipeline.tensor_transform  # ToTensor + Normalize([0.5],[0.5]) -> ~[-1,1]

    agnostic_images, default_masks, densepose_images = [], [], []
    target_layouts, cloths, images, fit_levels = [], [], [], []
    prompts, prompts_cloth = [], []

    for sample in raw_batch:
        avatar = sample["avatar_image"].resize((768, 1024))
        garment = sample["garment_image"].resize((768, 1024))
        result = sample["result_image"].resize((768, 1024))

        keypoints = pipeline.openpose_model(avatar.resize((384, 512)))
        model_parse, _ = pipeline.parsing_model(avatar.resize((384, 512)))
        mask, mask_gray = get_mask_location("hd", "upper_body", model_parse, keypoints)
        mask = mask.resize((768, 1024))
        mask_gray_tensor = (1 - tt(mask)) * tt(avatar)
        mask_gray_pil = to_pil_image((mask_gray_tensor + 1.0) / 2.0)

        result_keypoints = pipeline.openpose_model(result.resize((384, 512)))
        result_parse, _ = pipeline.parsing_model(result.resize((384, 512)))
        layout_mask, _ = get_mask_location("hd", "upper_body", result_parse, result_keypoints)
        layout_mask = layout_mask.resize((768, 1024))

        pose_img = pipeline.run_densepose(avatar)

        agnostic_images.append(tt(mask_gray_pil))
        default_masks.append(to_tensor(mask))
        densepose_images.append(tt(pose_img))
        target_layouts.append(to_tensor(layout_mask))
        cloths.append(tt(garment))
        images.append(tt(result))
        fit_levels.append(parse_fit_level_from_text(sample["fit_text"]))
        prompts.append("model is wearing " + sample["fit_text"])
        prompts_cloth.append("a photo of " + sample["fit_text"])

    with torch.no_grad():
        encoder_hidden_states, _, pooled_prompt_embeds, _ = pipeline.pipe.encode_prompt(
            prompts, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
        text_embeds_cloth, _, _, _ = pipeline.pipe.encode_prompt(
            prompts_cloth, num_images_per_prompt=1, do_classifier_free_guidance=False,
        )

    # Standard SDXL micro-conditioning (original_size, crop_top_left, target_size),
    # fixed to the 768x1024 (width x height) this pipeline always resizes to.
    add_time_ids = torch.tensor([[1024, 768, 0, 0, 1024, 768]] * len(raw_batch), device=device)

    return Fit4MenSample(
        image=torch.stack(images).to(device, dtype),
        agnostic_image=torch.stack(agnostic_images).to(device, dtype),
        default_mask=torch.stack(default_masks).to(device, dtype),
        densepose_image=torch.stack(densepose_images).to(device, dtype),
        target_layout=torch.stack(target_layouts).to(device, dtype),
        fit_level=torch.tensor(fit_levels, device=device),
        cloth=torch.stack(cloths).to(device, dtype),
        text_embeds_cloth=text_embeds_cloth.to(device, dtype),
        encoder_hidden_states=encoder_hidden_states.to(device, dtype),
        unet_added_cond_kwargs={"text_embeds": pooled_prompt_embeds.to(device, dtype), "time_ids": add_time_ids},
    )


def training_step(
    pipeline: "idm_vton_inference.IDMVTONPipeline",
    fit_controler: FitControler,
    batch: Fit4MenSample,
    layout_loss_weight: float = 1.0,
    diffusion_loss_weight: float = 1.0,
) -> dict:
    """One training step: layout loss + VTON diffusion loss, gradients flowing only
    into fit_controler.parameters() (everything from `pipeline` is frozen)."""
    device, dtype = pipeline.device, pipeline.dtype
    vae, unet, unet_encoder, scheduler = (
        pipeline.pipe.vae, pipeline.pipe.unet, pipeline.pipe.unet_encoder, pipeline.pipe.scheduler
    )

    agnostic_repr = build_garment_agnostic_input(
        agnostic_image=batch.agnostic_image.to(device, dtype),
        default_mask=batch.default_mask.to(device, dtype),
        densepose_image=batch.densepose_image.to(device, dtype),
    )
    layout_logits, _ = fit_controler.prepare_for_training(
        agnostic_repr, batch.fit_level.to(device)
    )
    layout_loss = F.binary_cross_entropy_with_logits(
        layout_logits, batch.target_layout.to(device, layout_logits.dtype)
    )

    # VTON diffusion loss -- mirrors upstream train_xl.py's own training step (VAE
    # encode -> add noise -> unet_encoder reference features -> main unet noise
    # prediction -> MSE), with the frozen unet/unet_encoder/vae from `pipeline` reused
    # as-is and FitControler's injector attached only around the main unet call.
    with torch.no_grad():
        model_input = vae.encode(batch.image.to(device, vae.dtype)).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor

    noise = torch.randn_like(model_input)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (model_input.shape[0],), device=device
    ).long()
    noisy_latents = scheduler.add_noise(model_input, noise, timesteps)

    with torch.no_grad():
        _, reference_features = unet_encoder(
            batch.cloth.to(device, dtype), timesteps, batch.text_embeds_cloth.to(device, dtype),
            return_dict=False,
        )

    fit_controler.attach(unet)
    try:
        noise_pred = unet(
            noisy_latents,
            timesteps,
            batch.encoder_hidden_states.to(device, dtype),
            added_cond_kwargs=batch.unet_added_cond_kwargs,
            garment_features=reference_features,
        ).sample
    finally:
        fit_controler.detach()

    diffusion_loss = F.mse_loss(noise_pred.float(), noise.float())

    total_loss = layout_loss_weight * layout_loss + diffusion_loss_weight * diffusion_loss
    return {"loss": total_loss, "layout_loss": layout_loss.detach(), "diffusion_loss": diffusion_loss.detach()}


def freeze_base_pipeline(pipeline: "idm_vton_inference.IDMVTONPipeline") -> None:
    for module in (
        pipeline.pipe.vae,
        pipeline.pipe.unet,
        pipeline.pipe.unet_encoder,
        pipeline.pipe.text_encoder,
        pipeline.pipe.text_encoder_2,
        pipeline.pipe.image_encoder,
    ):
        module.requires_grad_(False)
        module.eval()


def save_epoch_sample_grid(
    pipeline: "idm_vton_inference.IDMVTONPipeline",
    fit_controler: FitControler,
    sample: dict,
    epoch: int,
    output_dir: Path,
    steps: int = 20,
    guidance_scale: float = 2.0,
) -> Path:
    """Runs one fixed (avatar, garment) pair through the current FitControler once per
    fit level (len(FIT_LEVELS) == 5 by default: tight/fitted/regular/loose/oversized),
    holding the prompt, seed, and everything else constant -- only the injected fit
    conditioning differs between panels -- and stitches the results into one labeled
    image. `sample` is a raw GarmentCodeVTONDataset item (avatar_image, garment_image,
    fit_text); the SAME sample should be passed every epoch (pick it once before the
    training loop) so panels are comparable across epoch_NNN.png files over time.

    Runs a full multi-step diffusion sample each call (via IDMVTONPipeline.run_tryon),
    unlike training_step()'s single noisy-timestep loss -- this is meaningfully slower
    than one training step; keep `steps` modest (default 20) since this happens once per
    epoch, not per training step.
    """
    was_training = fit_controler.training
    fit_controler.eval()

    panels = []
    try:
        with torch.no_grad():
            for fit_name in FIT_LEVELS:
                fit_idx = FIT_LEVELS.index(fit_name)
                hook = make_conditioning_hook(
                    fit_controler, pipeline.pipe.unet, fit_idx, pipeline.device, pipeline.dtype
                )
                try:
                    result, _ = pipeline.run_tryon(
                        person_img=sample["avatar_image"],
                        garm_img=sample["garment_image"],
                        garment_desc=sample["fit_text"],
                        model_type="hd",
                        category="upper_body",
                        auto_mask=True,
                        user_mask=None,
                        crop=False,
                        denoise_steps=steps,
                        guidance_scale=guidance_scale,
                        # fixed seed across panels/epochs -- isolates the effect of
                        # FitControler's fit conditioning from ordinary sampling noise
                        seed=0,
                        on_conditioning_ready=hook,
                    )
                finally:
                    fit_controler.detach()
                panels.append((fit_name, result))
    finally:
        if was_training:
            fit_controler.train()

    label_h = 28
    panel_w, panel_h = panels[0][1].size
    grid = Image.new("RGB", (panel_w * len(panels), panel_h + label_h), "white")
    draw = ImageDraw.Draw(grid)
    for i, (fit_name, img) in enumerate(panels):
        grid.paste(img, (i * panel_w, label_h))
        draw.text((i * panel_w + 8, 6), f"epoch {epoch} - {fit_name}", fill="black")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"epoch_{epoch:03d}.png"
    grid.save(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset", choices=["garmentcode_vton", "fit4men"], default="garmentcode_vton",
        help="garmentcode_vton (default): huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset, "
             "downloaded via scripts/download_dataset.py. fit4men: the paper's own dataset "
             "format -- NOT public, requires --data-root pointing at your own data and a real "
             "Fit4MenDataset implementation (see that class's docstring).",
    )
    parser.add_argument(
        "--data-dir", default="data/garmentcode_vton",
        help="[garmentcode_vton] local dir written by scripts/download_dataset.py "
             "(load_from_disk); falls back to downloading via the HF cache if missing",
    )
    parser.add_argument("--split", default="train", help="[garmentcode_vton] dataset split")
    parser.add_argument("--data-root", default=None, help="[fit4men] path to your Fit4Men-format dataset")
    parser.add_argument("--model-path", default="yisol/IDM-VTON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--layout-loss-weight", type=float, default=1.0)
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1000, help="steps between checkpoints")
    parser.add_argument(
        "--no-epoch-samples", action="store_true",
        help="disable the end-of-epoch sample grid (see save_epoch_sample_grid()) -- "
             "on by default, garmentcode_vton only",
    )
    parser.add_argument(
        "--sample-dir", default=None,
        help="where to save epoch sample grids (default: <output-dir>/samples)",
    )
    parser.add_argument("--sample-steps", type=int, default=20, help="denoising steps for epoch samples (kept low -- runs 5x per epoch)")
    parser.add_argument("--sample-guidance-scale", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "garmentcode_vton":
        dataset = GarmentCodeVTONDataset(split=args.split, data_dir=args.data_dir)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=raw_collate,
        )
        needs_prepare_batch = True
        # Fixed once, up front, so every epoch_NNN.png uses the SAME (avatar, garment)
        # pair -- otherwise panels across epochs wouldn't be comparable.
        visualization_sample = dataset[0] if not args.no_epoch_samples else None
    else:
        if not args.data_root:
            raise SystemExit("--dataset fit4men requires --data-root <path to your data>")
        dataset = Fit4MenDataset(args.data_root)  # raises FileNotFoundError until you point
        # this at real data, and __getitem__/__len__ raise NotImplementedError until you
        # implement them -- see the class docstring.
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        needs_prepare_batch = False
        if not args.no_epoch_samples:
            print(
                "note: epoch sample grids are only implemented for --dataset "
                "garmentcode_vton (needs a raw PIL/text sample) -- skipping for fit4men. "
                "Pass --no-epoch-samples to silence this."
            )
        visualization_sample = None

    pipeline = idm_vton_inference.IDMVTONPipeline(args.model_path, args.device)
    freeze_base_pipeline(pipeline)

    fit_controler = FitControler(FitControlerConfig()).to(args.device)
    fit_controler.train()
    optimizer = torch.optim.AdamW(fit_controler.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(args.sample_dir) if args.sample_dir else output_dir / "samples"

    step = 0
    for epoch in range(args.epochs):
        for raw_batch in loader:
            batch = prepare_batch(pipeline, raw_batch) if needs_prepare_batch else raw_batch

            optimizer.zero_grad()
            losses = training_step(
                pipeline, fit_controler, batch,
                layout_loss_weight=args.layout_loss_weight,
                diffusion_loss_weight=args.diffusion_loss_weight,
            )
            losses["loss"].backward()
            optimizer.step()
            step += 1

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss={losses['loss'].item():.4f} "
                    f"layout={losses['layout_loss'].item():.4f} "
                    f"diffusion={losses['diffusion_loss'].item():.4f}"
                )
            if step % args.save_every == 0:
                ckpt_path = output_dir / f"fitcontroler_step{step}.pt"
                fit_controler.save(str(ckpt_path))
                print(f"saved {ckpt_path}")

        if visualization_sample is not None:
            grid_path = save_epoch_sample_grid(
                pipeline, fit_controler, visualization_sample, epoch, sample_dir,
                steps=args.sample_steps, guidance_scale=args.sample_guidance_scale,
            )
            print(f"epoch {epoch} sample grid -> {grid_path}")

    fit_controler.save(str(output_dir / "fitcontroler_final.pt"))


if __name__ == "__main__":
    main()
