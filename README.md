# OmniTry on a single RTX 4090

Setup and CLI inference for [OmniTry](https://github.com/Kunbyte-AI/OmniTry) — "Virtual
Try-On Anything without Masks" — tuned to run on a single RTX 4090 (24GB).

OmniTry wraps FLUX.1-Fill-dev with two LoRA adapters (one for the person image, one for
the garment/object image) so it can try on clothes, shoes, jewelry, glasses, bags, hats,
ties and more from a single reference photo, with no mask required.

## The VRAM problem on a 4090

FLUX.1-Fill-dev's transformer has ~11.9B parameters. In bf16 that's **~24GB of weights
alone** — before activations, the T5-XXL text encoder, CLIP encoder, or VAE. The upstream
project states a **28GB minimum**, and that's *with* whole-module CPU offload already
applied. A 24GB 4090 has no headroom left at that point, so `inference.py` defaults to a
safer strategy and offers a faster opt-in one:

| `--offload`  | `--quantize` | Peak VRAM | Speed | Notes |
|---|---|---|---|---|
| `sequential` (default) | `none` | ~8-10GB | slowest | streams weights layer-by-layer, always fits |
| `model` | `fp8` | ~14-16GB | fastest that fits a 4090 | needs `optimum-quanto` |
| `model` | `none` | ~26GB+ | fast | **will OOM on a 4090**, needs a 28GB+ GPU |
| `none` | `none` | ~30GB+ | fastest | needs a 28GB+ GPU |

Start with the default (`sequential`, no quantization) to confirm everything works, then
switch to `--offload model --quantize fp8` for faster iteration once you've validated the
setup.

## 1. Install

```bash
git clone <this-repo>
cd claude_virtual_try_ON
./scripts/install.sh
```

This creates `.venv`, installs pinned dependencies (PyTorch 2.7 / CUDA 12.6, which
supports the 4090's Ada Lovelace architecture out of the box), and clones the upstream
OmniTry source into `third_party/OmniTry` (model code only, no weights).

Flash-attention is attempted but optional — PyTorch's built-in `scaled_dot_product_attention`
already dispatches to a flash-attention backend on Ada GPUs, so a failed flash-attn build
is not a blocker.

## 2. Download checkpoints

FLUX.1-Fill-dev is a **gated** model on Hugging Face:

```bash
# 1. request access at https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
# 2. authenticate:
huggingface-cli login
# 3. download FLUX.1-Fill-dev (~34GB) + the OmniTry LoRA (~350MB):
python scripts/download_checkpoints.py
```

This populates:
```
checkpoints/
├── FLUX.1-Fill-dev/
└── omnitry_v1_unified.safetensors
```

Pass `--lora-variant clothes` to instead fetch the clothing-only LoRA
(`configs/omnitry_v1_clothes.yaml`), which is smaller in scope but was trained
specifically for garments.

## 3. Run inference

```bash
python inference.py \
  --person examples/person.jpg \
  --object examples/garment.jpg \
  --class "top clothes" \
  --output out.png \
  --offload model --quantize fp8
```

Valid `--class` values (from `configs/omnitry_v1_unified.yaml`'s `object_map`):
`top clothes`, `bottom clothes`, `dress`, `shoe`, `earrings`, `bracelet`, `necklace`,
`ring`, `sunglasses`, `glasses`, `belt`, `bag`, `hat`, `tie`, `bow tie`.

Other flags:
- `--steps` (default 20), `--guidance-scale` (default 30), `--seed` (default random)
- `--compile` — `torch.compile`s the transformer (`reduce-overhead` mode). Adds warmup
  time on the first generation; only pays off across multiple runs at the same
  resolution, since a new image size triggers recompilation.
- `--lora-path` / `--config` to point at the clothes-only variant instead of unified.

## What's actually optimized for the 4090

- **TF32 matmuls + SDPA flash/mem-efficient kernels** enabled unconditionally
  (`enable_4090_matmul_opts` in `inference.py`) — free speed, no quality cost, no
  extra dependency (PyTorch ships this; no separate flash-attn build required).
- **VAE tiling + slicing** always on — shrinks the VAE decode step's peak memory with
  no visible quality impact at the resolutions this pipeline runs at.
- **fp8 weight-only quantization** (`--quantize fp8`, via `optimum-quanto`) of the
  transformer and the T5-XXL text encoder — the two largest memory consumers — cutting
  their footprint roughly in half so the faster whole-module offload path fits in 24GB.
  Quantization is applied *after* the LoRA adapters are attached, so the LoRA
  up/down-projection weights themselves stay in bf16 (a QLoRA-style split: quantized
  frozen base + full-precision adapter).
- **Sequential CPU offload by default** — the one setting guaranteed to fit a 4090
  regardless of quantization, at the cost of per-step PCIe transfer time.

## Troubleshooting

- **CUDA OOM**: make sure you're on the default `--offload sequential`, or add
  `--quantize fp8`. Also try a smaller input image (the pipeline caps resolution at
  1024×1024 already, rounded to a multiple of 16).
- **403 / gated repo error downloading FLUX.1-Fill-dev**: you must request access on the
  model page and run `huggingface-cli login` with a token that has accepted the license.
- **flash-attn fails to build**: safe to ignore; `inference.py` doesn't depend on it and
  PyTorch's SDPA backend is used automatically.
- **`ModuleNotFoundError: omnitry`**: run `./scripts/install.sh` first — it clones the
  upstream model code into `third_party/OmniTry`, which `inference.py` adds to
  `sys.path` at import time.
- **`Expected types for transformer: ... got omnitry...FluxTransformer2DModel`**: this is
  a harmless warning from `FluxFillPipeline.from_pretrained`, printed because OmniTry
  subclasses diffusers' transformer. It doesn't stop the run — the same line appears
  running upstream's own `gradio_demo.py`.
- **`--quantize fp8` crashes with `fatal error: Python.h: No such file or directory`**
  (inside a `ninja`/`nvcc` build of a `quanto_cuda` extension): on Ada GPUs (compute
  capability 8.9, i.e. the 4090) `optimum-quanto` JIT-compiles a CUDA extension the first
  time a quantized module is moved to the GPU, and that needs your Python interpreter's
  dev headers. Install them and retry:
  ```bash
  sudo apt-get install -y python3.12-dev   # match your venv's Python version
  ```
  `inference.py` now checks for `Python.h`, `nvcc`, and `ninja` up front when
  `--quantize fp8` is passed and exits with this same guidance instead of a mid-pipeline
  traceback. If you'd rather not touch system packages, drop `--quantize fp8` and use
  `--offload sequential` (the default) — slower, but needs no build toolchain.

## Credit

Model and original code: [Kunbyte-AI/OmniTry](https://github.com/Kunbyte-AI/OmniTry)
(paper: [arXiv:2508.13632](https://arxiv.org/abs/2508.13632)). This repo only adds an
install script, a checkpoint downloader, and a 4090-oriented CLI wrapper around the
upstream pipeline — the core diffusion/LoRA logic in `inference.py` mirrors upstream's
`gradio_demo.py`.
