# OmniTry on a single RTX 4090

> Part of a small collection of virtual try-on model setups in this repo — see the
> [root README](../README.md) for the other one (FitDiT).

Setup and CLI inference for [OmniTry](https://github.com/Kunbyte-AI/OmniTry) — "Virtual
Try-On Anything without Masks" — tuned to run on a single RTX 4090 (24GB).

OmniTry wraps FLUX.1-Fill-dev with two LoRA adapters (one for the person image, one for
the garment/object image) so it can try on clothes, shoes, jewelry, glasses, bags, hats,
ties and more from a single reference photo, with no mask required.

## The VRAM problem on a 4090

FLUX.1-Fill-dev's transformer has ~11.9B parameters. In bf16 that's **~24GB of weights
alone** — before activations, the T5-XXL text encoder, CLIP encoder, or VAE. The upstream
project states a **28GB minimum**, and that's *with* whole-module CPU offload already
applied. A 24GB 4090 has no headroom left at that point.

The obvious fix is fp8 quantization to shrink the transformer enough for the faster
`--offload model` path to fit — **but on this pipeline, fp8 quantization via
`optimum-quanto` is confirmed broken**, in both modes tested on real 4090 hardware:

| `--offload`  | `--quantize` | Peak VRAM | Status |
|---|---|---|---|
| `sequential` (default) | `none` | ~8-10GB | **the only combination confirmed correct** |
| `model` | `transformer` | ~14-16GB | **CONFIRMED BROKEN — produces NaN/black output** |
| `model` | `all` | ~10-12GB | **CONFIRMED BROKEN — produces NaN/black output** |
| `model` | `none` | ~26GB+ | will OOM on a 4090, needs a 28GB+ GPU |
| `none` | `none` | ~30GB+ | needs a 28GB+ GPU |

Both `--quantize transformer` (fp8 the transformer only, T5 stays bf16) and `--quantize
all` (fp8 both) have produced NaN output in real testing, so it isn't T5-specific as
first suspected — quantizing the transformer alone is enough to break it. The likely
cause: some of FLUX's transformer weights (normalization/modulation layers are the usual
suspects for this) overflow fp8's representable range during quantization itself, before
generation even starts. This is unresolved; **do not use `--quantize` right now** —
`inference.py` still exposes it for anyone who wants to debug it further (e.g. by
excluding specific layer types from quantization), but it will reliably fail on the
current code.

**If `--offload sequential` feels too slow**, see "Getting more speed without
quantization" below — there are real, verified levers that don't touch the broken code
path.

## 1. Install

```bash
git clone <this-repo>
cd claude_virtual_try_ON/omnitry
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
  --output out.png
```

Valid `--class` values (from `configs/omnitry_v1_unified.yaml`'s `object_map`):
`top clothes`, `bottom clothes`, `dress`, `shoe`, `earrings`, `bracelet`, `necklace`,
`ring`, `sunglasses`, `glasses`, `belt`, `bag`, `hat`, `tie`, `bow tie`.

Other flags:
- `--steps` (default 20), `--guidance-scale` (default 30), `--seed` (default random)
- `--max-area` (default `1024*1024`) — cap on generated width×height
- `--compile` — `torch.compile`s the transformer (`reduce-overhead` mode). Adds warmup
  time on the first generation; only pays off across multiple runs at the same
  resolution, since a new image size triggers recompilation.
- `--lora-path` / `--config` to point at the clothes-only variant instead of unified.

## Getting more speed without quantization

`--quantize` is broken right now (see above), so these are the levers that actually work
today, roughly in order of impact:

1. **Fewer steps.** `--steps 20` is the OmniTry demo default; try `--steps 12` to `--steps
   14` for iteration/preview generations — time scales close to linearly with step count,
   and quality loss is often minor for this kind of edit. Use the full 20 (or more) for a
   final render.
2. **Lower `--max-area`.** The pipeline caps generated resolution at `1024*1024` by
   default, rounded to a multiple of 16. Diffusion cost scales with pixel count, so
   `--max-area 589824` (768×768) is a real, verified speedup — try it for fast iteration,
   then re-render your final pick at the default resolution.
3. **`--compile`.** `torch.compile`s the transformer with `reduce-overhead`. The first
   generation pays a real warmup cost (can be a minute or more); every generation after
   that, at the *same* resolution, is faster. Worth it if you're generating many looks in
   one session at a fixed size; not worth it for a single one-off run.

None of these touch quantization or offload strategy, so they're safe to combine with the
default `--offload sequential`.

## What's actually optimized for the 4090

- **TF32 matmuls + SDPA flash/mem-efficient kernels** enabled unconditionally
  (`enable_4090_matmul_opts` in `inference.py`) — free speed, no quality cost, no
  extra dependency (PyTorch ships this; no separate flash-attn build required).
- **VAE tiling + slicing** always on — shrinks the VAE decode step's peak memory with
  no visible quality impact at the resolutions this pipeline runs at.
- **Sequential CPU offload by default** — the one setting confirmed to produce correct
  output on a 4090, at the cost of per-step PCIe transfer time.
- **fp8 weight-only quantization** (`--quantize transformer`/`all`, via `optimum-quanto`)
  — implemented, but **confirmed broken** on this pipeline (produces NaN output in both
  modes on real 4090 hardware). Left in the code for anyone debugging it further; see
  "The VRAM problem on a 4090" above and Troubleshooting below. Not currently a source of
  speed on this repo.

## Troubleshooting

- **CUDA OOM**: make sure you're on the default `--offload sequential` (`--offload model`
  needs ~26GB+ on this pipeline and `--quantize` is currently broken — see above). Also
  try a lower `--max-area` (e.g. `589824` for 768×768).
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
- **`--quantize transformer` or `--quantize all` produces a fully black `output.png`**
  (confirmed in testing, both modes): the run completes with no crash, but diffusers
  prints `RuntimeWarning: invalid value encountered in cast` from `image_processor.py`
  right before saving — the decoded image contains `NaN` pixels, which diffusers silently
  casts to garbage/black instead of erroring. `inference.py` catches that exact warning
  and raises a clear `SystemExit` instead of writing the broken PNG.

  This was first suspected to be T5-specific (`--quantize all` also quantizes the T5 text
  encoder, and T5 is well known to be numerically fragile at reduced precision), but
  `--quantize transformer` — which fp8-quantizes *only* the transformer and leaves T5 in
  bf16 — **also** produces NaN output. So the problem is in quantizing the transformer
  itself, not T5. The most likely cause: `optimum-quanto`'s blanket
  `quantize(transformer, weights=qfloat8)` quantizes every `nn.Linear` it finds,
  including ones that don't tolerate fp8's narrow representable range well (weights in
  normalization/modulation layers are the classic case) — those overflow to `inf`/`NaN`
  during the quantization step itself, before generation even runs. This hasn't been
  confirmed further (no GPU access in the environment that authored this fix), so treat
  it as the leading hypothesis, not a verified diagnosis.

  **Fix: don't use `--quantize` at all right now** — use `--offload sequential` (the
  default) and see "Getting more speed without quantization" above for real speedups
  (`--steps`, `--max-area`, `--compile`). If you want to chase the underlying bug
  further, the next experiment would be excluding specific layer name patterns (e.g.
  anything matching `norm`, `embedder`, `proj_out`) from `optimum-quanto`'s `quantize()`
  call in `quantize_fp8()` (`inference.py`) rather than quantizing the whole transformer
  indiscriminately — that isn't implemented here.

## Credit

Model and original code: [Kunbyte-AI/OmniTry](https://github.com/Kunbyte-AI/OmniTry)
(paper: [arXiv:2508.13632](https://arxiv.org/abs/2508.13632)). This repo only adds an
install script, a checkpoint downloader, and a 4090-oriented CLI wrapper around the
upstream pipeline — the core diffusion/LoRA logic in `inference.py` mirrors upstream's
`gradio_demo.py`.
