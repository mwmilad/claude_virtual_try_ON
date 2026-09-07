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
| `model` | `transformer` | ~14-16GB | **fast, recommended** | fp8-quantizes only the transformer; T5 stays bf16 |
| `model` | `all` | ~10-12GB | fastest that fits a 4090 | **EXPERIMENTAL / KNOWN BROKEN — fp8-quantizing T5 has produced NaN/black output, see Troubleshooting** |
| `model` | `none` | ~26GB+ | fast | **will OOM on a 4090**, needs a 28GB+ GPU |
| `none` | `none` | ~30GB+ | fastest | needs a 28GB+ GPU |

If `--offload sequential` feels too slow, use **`--offload model --quantize transformer`**:
it fp8-quantizes only the transformer (the actual VRAM hog) and keeps the T5 text encoder
in bf16, so it should give you most of the speed of whole-module offload without the NaN
bug that full fp8 quantization (`--quantize all`) hit in testing — T5 is well known to be
numerically fragile at reduced precision, which is almost certainly why that combination
produced NaNs. `--quantize transformer` hasn't been run on real hardware here to confirm
it's clean either, but it targets the actual suspect; if it still produces NaN output,
`inference.py` will now catch it and tell you clearly instead of saving a black image
(see Troubleshooting) — please report back either way.

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
# safe default (confirmed correct, slower)
python inference.py \
  --person examples/person.jpg \
  --object examples/garment.jpg \
  --class "top clothes" \
  --output out.png

# faster: fp8 the transformer only, keep T5 in bf16
python inference.py \
  --person examples/person.jpg \
  --object examples/garment.jpg \
  --class "top clothes" \
  --output out.png \
  --offload model --quantize transformer
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
- **fp8 weight-only quantization** (`--quantize transformer`, via `optimum-quanto`) of
  just the transformer — the actual memory hog (~24GB → ~12GB in bf16 vs fp8) — freeing
  enough VRAM for the faster whole-module offload path (`--offload model`) to fit in
  24GB. Quantization is applied *after* the LoRA adapters are attached, so the LoRA
  up/down-projection weights themselves stay in bf16 (a QLoRA-style split: quantized
  frozen base + full-precision adapter). The T5 text encoder is deliberately left in
  bf16 — `--quantize all` additionally fp8-quantizes it and is **experimental / known
  broken**: it produced NaN/black output in testing (T5 is numerically fragile at
  reduced precision). See Troubleshooting.
- **Sequential CPU offload by default** — the one setting guaranteed to fit a 4090
  regardless of quantization, at the cost of per-step PCIe transfer time.

## Troubleshooting

- **CUDA OOM**: make sure you're on the default `--offload sequential`, or use
  `--offload model --quantize transformer`. Also try a smaller input image (the pipeline
  caps resolution at 1024×1024 already, rounded to a multiple of 16).
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
- **`--quantize transformer`/`all` crashes with `fatal error: Python.h: No such file or
  directory`** (inside a `ninja`/`nvcc` build of a `quanto_cuda` extension): on Ada GPUs
  (compute capability 8.9, i.e. the 4090) `optimum-quanto` JIT-compiles a CUDA extension
  the first time a quantized module is moved to the GPU, and that needs your Python
  interpreter's dev headers. Install them and retry:
  ```bash
  sudo apt-get install -y python3.12-dev   # match your venv's Python version
  ```
  `inference.py` checks for `Python.h`, `nvcc`, and `ninja` up front whenever
  `--quantize` isn't `none` and exits with this same guidance instead of a mid-pipeline
  traceback. If you'd rather not touch system packages, drop `--quantize` and use
  `--offload sequential` (the default) — slower, but needs no build toolchain.
- **`--quantize all` produces a fully black `output.png`** (confirmed in testing): the run
  completes with no crash, but diffusers prints `RuntimeWarning: invalid value encountered
  in cast` from `image_processor.py` right before saving — the decoded image contains
  `NaN` pixels, which diffusers silently casts to garbage/black instead of erroring.
  `inference.py` catches that exact warning and raises a clear `SystemExit` instead of
  writing the broken PNG. Root cause: fp8-quantizing the T5 text encoder produces `NaN`
  activations — T5 is well known to be numerically fragile at reduced precision, and this
  is almost certainly the same story here, possibly compounded by the custom
  monkey-patched LoRA forward (`add_omnitry_lora`'s `hacked_lora_forward`) that OmniTry's
  dual-adapter setup relies on. **Fix: use `--quantize transformer` instead of `all`**
  (keeps T5 in bf16), or drop `--quantize` entirely and use `--offload sequential`. If
  `--quantize transformer` *also* produces NaN output, that's a stronger signal the
  monkey-patched LoRA forward itself doesn't tolerate a quantized base layer — please
  report it; the interim fix is the same (drop `--quantize`).

## Credit

Model and original code: [Kunbyte-AI/OmniTry](https://github.com/Kunbyte-AI/OmniTry)
(paper: [arXiv:2508.13632](https://arxiv.org/abs/2508.13632)). This repo only adds an
install script, a checkpoint downloader, and a 4090-oriented CLI wrapper around the
upstream pipeline — the core diffusion/LoRA logic in `inference.py` mirrors upstream's
`gradio_demo.py`.
