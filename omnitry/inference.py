#!/usr/bin/env python3
"""CLI inference for OmniTry (FLUX.1-Fill-dev + dual LoRA) tuned for a single RTX 4090.

OmniTry's transformer is FLUX.1-Fill-dev (~11.9B params). In bf16 the weights alone are
~24GB, so the upstream project states a 28GB VRAM minimum even with whole-module CPU
offload. A 24GB 4090 has no headroom left over for activations at that point, so this
script defaults to the safest option and offers a faster one:

  --offload sequential (default)   Streams the transformer layer-by-layer to the GPU.
                                    Fits comfortably under 24GB. Slowest per step.
  --offload model                  Whole submodules moved to GPU on demand (faster).
                                    Needs ~26GB+ on this pipeline -- OOMs on a 4090.
  --quantize transformer / all     fp8-quantizes via optimum-quanto to shrink the model
                                    enough for --offload model to fit a 4090.
                                    CONFIRMED BROKEN on this pipeline, BOTH modes: fp8
                                    quantization produces NaN output (caught and reported
                                    by this script rather than saved as a black image).
                                    Root cause is unresolved -- likely some of FLUX's
                                    transformer weights (normalization/modulation layers
                                    are the usual suspects) overflow fp8's representable
                                    range during quantization itself. Do not use either
                                    mode until this is fixed upstream in this repo.

For real speed on a 4090 without quantization, see README "Getting more speed without
quantization" -- fewer --steps, --compile, and a --max-area override are the verified
levers; --offload/--quantize are not, right now.

Example (the only combination confirmed correct on a 4090):
    python inference.py \\
        --person examples/person.jpg \\
        --object examples/garment.jpg \\
        --class "top clothes" \\
        --output out.png \\
        --steps 14 --compile
"""
import argparse
import math
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import peft
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from peft import LoraConfig
from PIL import Image
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent
OMNITRY_SRC = REPO_ROOT / "third_party" / "OmniTry"
if str(OMNITRY_SRC) not in sys.path:
    sys.path.insert(0, str(OMNITRY_SRC))

try:
    from omnitry.models.transformer_flux import FluxTransformer2DModel
    from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Couldn't import omnitry.* -- run scripts/install.sh first "
        f"(expected the upstream source at {OMNITRY_SRC})"
    ) from exc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_4090_matmul_opts() -> None:
    """Ada Lovelace (4090) tuning: TF32 matmuls and PyTorch's flash/mem-efficient SDPA
    backends. These are safe defaults that cost no accuracy for this pipeline and are
    the reason no separate flash-attn build is required to get fast attention."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def check_quanto_build_toolchain() -> None:
    """On Ada+ GPUs (compute capability >= 8.9, e.g. the 4090) optimum-quanto
    auto-upgrades fp8 weights to a hardware-packed 'Marlin' format the first time a
    quantized module is moved to CUDA, by JIT-compiling a CUDA extension on the spot.
    That build needs nvcc, a C++ compiler, ninja, and the Python headers for the
    interpreter currently running -- check for them up front so a missing dependency
    fails fast with instructions instead of a traceback in the middle of a pipeline
    call."""
    import shutil
    import sysconfig

    problems = []
    py_include = sysconfig.get_path("include")
    if not os.path.exists(os.path.join(py_include, "Python.h")):
        v = sys.version_info
        problems.append(
            f"Python.h not found in {py_include} -- install your distro's dev headers "
            f"for this interpreter, e.g. `sudo apt-get install python{v.major}.{v.minor}-dev`"
        )
    if shutil.which("nvcc") is None:
        problems.append("nvcc not found on PATH -- install the CUDA toolkit matching your driver")
    if shutil.which("ninja") is None:
        problems.append("ninja not found on PATH -- pip install ninja")
    if problems:
        raise SystemExit(
            "--quantize (transformer/all) needs to JIT-build optimum-quanto's CUDA extension "
            "on this GPU (compute capability >= 8.9, e.g. RTX 4090). Missing:\n  - "
            + "\n  - ".join(problems)
        )


def quantize_fp8(module: torch.nn.Module, name: str) -> None:
    try:
        from optimum.quanto import freeze, qfloat8, quantize
    except ImportError as exc:
        raise SystemExit(
            "optimum-quanto is required for --quantize transformer/all (pip install optimum-quanto)"
        ) from exc
    print(f"[4090] quantizing {name} to fp8 ...")
    quantize(module, weights=qfloat8)
    freeze(module)


def add_omnitry_lora(transformer: torch.nn.Module, lora_rank: int, lora_alpha: int, lora_path: str) -> None:
    """Reproduces OmniTry's dual-adapter setup: a 'vtryon_lora' adapter applied to the
    person-image branch and a 'garment_lora' adapter applied to the garment/object-image
    branch of the same batch, via a monkey-patched forward on every LoRA Linear."""
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "x_embedder",
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2",
            "norm1_context.linear", "norm1.linear", "norm.linear", "proj_mlp", "proj_out",
        ],
    )
    transformer.add_adapter(lora_config, adapter_name="vtryon_lora")
    transformer.add_adapter(lora_config, adapter_name="garment_lora")

    with safe_open(lora_path, framework="pt") as f:
        lora_weights = {k: f.get_tensor(k) for k in f.keys()}
        transformer.load_state_dict(lora_weights, strict=False)

    def create_hacked_forward(module):
        def lora_forward(self, active_adapter, x, *a, **kw):
            result = self.base_layer(x, *a, **kw)
            if active_adapter is not None:
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x))) * scaling
            return result

        def hacked_lora_forward(self, x, *a, **kw):
            return torch.cat(
                (
                    lora_forward(self, "vtryon_lora", x[:1], *a, **kw),
                    lora_forward(self, "garment_lora", x[1:], *a, **kw),
                ),
                dim=0,
            )

        return hacked_lora_forward.__get__(module, type(module))

    for _, m in transformer.named_modules():
        if isinstance(m, peft.tuners.lora.layer.Linear):
            m.forward = create_hacked_forward(m)


def build_pipeline(args, cfg) -> "FluxFillPipeline":
    weight_dtype = torch.bfloat16

    transformer = (
        FluxTransformer2DModel.from_pretrained(f"{cfg.model_root}/transformer")
        .requires_grad_(False)
        .to(dtype=weight_dtype)
    )

    lora_path = args.lora_path or cfg.lora_path
    add_omnitry_lora(transformer, cfg.lora_rank, cfg.lora_alpha, lora_path)

    if args.quantize in ("transformer", "all"):
        quantize_fp8(transformer, "transformer")

    pipeline = FluxFillPipeline.from_pretrained(
        cfg.model_root, transformer=transformer.eval(), torch_dtype=weight_dtype
    )

    if args.quantize == "all":
        # Both this and transformer-only quantization are confirmed to produce NaN
        # output on real 4090 hardware (see README) -- kept opt-in for anyone debugging
        # the root cause further, not because either mode is currently safe to use.
        quantize_fp8(pipeline.text_encoder_2, "text_encoder_2 (T5)")

    # Always on: cheap, no quality cost, meaningfully lowers peak VAE memory.
    pipeline.vae.enable_tiling()
    pipeline.vae.enable_slicing()

    if args.offload == "sequential":
        pipeline.enable_sequential_cpu_offload()
    elif args.offload == "model":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda:0")

    if args.compile:
        pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")

    return pipeline


def prep_images(person: Image.Image, obj: Image.Image, weight_dtype, device, max_area: int):
    oW, oH = person.width, person.height
    ratio = min(1, math.sqrt(max_area / (oW * oH)))
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16

    person_t = T.Compose([T.Resize((tH, tW)), T.ToTensor()])(person)

    ratio = min(tW / obj.width, tH / obj.height)
    obj_resized = T.Compose(
        [T.Resize((int(obj.height * ratio), int(obj.width * ratio))), T.ToTensor()]
    )(obj)
    obj_padded = torch.ones_like(person_t)
    new_h, new_w = obj_resized.shape[1], obj_resized.shape[2]
    min_x, min_y = (tW - new_w) // 2, (tH - new_h) // 2
    obj_padded[:, min_y : min_y + new_h, min_x : min_x + new_w] = obj_resized

    img_cond = torch.stack([person_t, obj_padded]).to(dtype=weight_dtype, device=device)
    mask = torch.zeros_like(img_cond)
    return img_cond, mask, tW, tH


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--person", required=True, help="path to the person/model photo")
    parser.add_argument("--object", required=True, help="path to the garment/accessory photo")
    parser.add_argument(
        "--class", dest="object_class", required=True,
        help="one of the object_map keys in --config, e.g. 'top clothes', 'sunglasses'",
    )
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--config", default="configs/omnitry_v1_unified.yaml")
    parser.add_argument("--lora-path", default=None, help="override lora_path from the config")
    parser.add_argument("--steps", type=int, default=20, help="fewer steps = faster, some quality cost (try 12-15 for fast iteration)")
    parser.add_argument("--guidance-scale", type=float, default=30)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument(
        "--max-area", type=int, default=1024 * 1024,
        help="cap on generated width*height (default 1024*1024=1048576). Lower this "
             "(e.g. 768*768=589824) for a real, verified speedup at some quality cost -- "
             "diffusion cost scales with pixel count.",
    )
    parser.add_argument(
        "--offload", choices=["sequential", "model", "none"], default="sequential",
        help="sequential = lowest VRAM, the only mode confirmed correct on a 4090 "
             "(default, slowest); model = faster but needs ~26GB+ on this pipeline, "
             "OOMs on a 4090 (--quantize would fit it but is currently broken -- see "
             "--quantize help); none = no offload, needs a GPU with >=28GB free",
    )
    parser.add_argument(
        "--quantize", choices=["none", "transformer", "all"], default="none",
        help="CONFIRMED BROKEN on this pipeline as of now, both 'transformer' and 'all' "
             "-- fp8 quantization (via optimum-quanto) has produced NaN (black) output "
             "in real testing on a 4090 for both modes. Do not use until fixed -- see "
             "README Troubleshooting. Left in place for anyone debugging it further.",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="torch.compile the transformer (reduce-overhead). Adds warmup time on the "
             "first call; pays off on repeated runs at a fixed resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not found. This script targets an RTX 4090.")

    enable_4090_matmul_opts()

    if args.quantize != "none":
        # 4090 = Ada Lovelace = sm_89. Pinning this avoids optimum-quanto's JIT build
        # compiling for every arch it can see, which is slower and, on multi-GPU/mixed
        # driver boxes, more likely to hit an unrelated compile error.
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        check_quanto_build_toolchain()

    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    seed_everything(args.seed)

    cfg = OmegaConf.load(args.config)
    if args.object_class not in cfg.object_map:
        raise SystemExit(f"--class must be one of: {list(cfg.object_map.keys())}")

    pipeline = build_pipeline(args, cfg)

    device = torch.device("cuda:0")
    weight_dtype = torch.bfloat16
    person = Image.open(args.person).convert("RGB")
    obj = Image.open(args.object).convert("RGB")
    img_cond, mask, tW, tH = prep_images(person, obj, weight_dtype, device, args.max_area)

    prompt = cfg.object_map[args.object_class]
    # diffusers silently casts NaN pixels to garbage/black instead of erroring (it emits
    # "invalid value encountered in cast" as a plain RuntimeWarning and moves on). Catch
    # that warning here so a NaN output fails loudly with an actionable message instead
    # of quietly writing a black PNG.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            image = pipeline(
                prompt=[prompt] * 2,
                height=tH,
                width=tW,
                img_cond=img_cond,
                mask=mask,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=torch.Generator(device).manual_seed(args.seed),
            ).images[0]

    if any("invalid value encountered in cast" in str(w.message) for w in caught):
        if args.quantize in ("all", "transformer"):
            hint = (
                f" --quantize {args.quantize} is the cause -- fp8 quantization is "
                "confirmed broken on this pipeline (both modes produce NaN in testing). "
                "Retry with no --quantize flag at all."
            )
        else:
            hint = (
                " no --quantize flag was used, so this is a new failure mode -- please "
                "report it with your exact command. As a baseline, retry with "
                "--offload sequential --steps 20 --guidance-scale 30 (the values known "
                "to work) to rule out --guidance-scale, --steps, or --max-area."
            )
        raise SystemExit(
            "Generation produced NaN pixels (diffusers cast them to a black/garbage "
            "image instead of erroring)." + hint
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"saved {args.output} (seed={args.seed})")


if __name__ == "__main__":
    main()
