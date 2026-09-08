#!/usr/bin/env python3
"""Training SKELETON for FitControler-on-IDM-VTON. Read this whole docstring first.

**This is not a runnable training script.** Two things are missing that no amount of
code here can supply:
  1. A real dataset. The paper's own dataset, Fit4Men (~13,000 body-garment pairs of
     different fits, tops + bottoms), is not publicly released as of when this was
     written -- `Fit4MenDataset` below defines the *expected sample format* (person
     photo, garment photo + description, target fit label, ground-truth fit-adjusted
     mask) and raises NotImplementedError from `__getitem__`, rather than pretending to
     load data that doesn't exist anywhere.
  2. Verification on real hardware. This has not been run end-to-end -- no GPU access in
     the environment that authored it.

What this file *does* get right, deliberately: the VTON-diffusion training step (VAE
encode -> add noise -> garment/reference UNet call -> main UNet call -> MSE loss) is
adapted from upstream IDM-VTON's own `train_xl.py` (fetched and quoted directly from
https://github.com/yisol/IDM-VTON), not guessed, so if you do have the frozen base
pipeline loaded, that part of the forward pass should match how IDM-VTON actually
expects to be driven.

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
    ground-truth one (needs Fit4Men-style per-sample fit-adjusted mask labels).
  - `diffusion_loss`: the same noise-prediction MSE IDM-VTON itself trains with, computed
    with FitControler's injector attached to the frozen main unet -- gradients flow back
    only into FitControler's parameters, everything else stays frozen.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
IDM_VTON_DIR = REPO_ROOT.parent / "idm-vton"
for p in (IDM_VTON_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inference as idm_vton_inference  # noqa: E402
from model import FitControler, FitControlerConfig, build_garment_agnostic_input  # noqa: E402


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="path to your Fit4Men-format dataset (see Fit4MenDataset)")
    parser.add_argument("--model-path", default="yisol/IDM-VTON")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--layout-loss-weight", type=float, default=1.0)
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1000, help="steps between checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = Fit4MenDataset(args.data_root)  # raises FileNotFoundError until you point
    # this at real data, and __getitem__/__len__ raise NotImplementedError until you
    # implement them -- see the class docstring.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    pipeline = idm_vton_inference.IDMVTONPipeline(args.model_path, args.device)
    freeze_base_pipeline(pipeline)

    fit_controler = FitControler(FitControlerConfig()).to(args.device)
    fit_controler.train()
    optimizer = torch.optim.AdamW(fit_controler.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
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

    fit_controler.save(str(output_dir / "fitcontroler_final.pt"))


if __name__ == "__main__":
    main()
