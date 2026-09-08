# IDM-VTON on a single RTX 4090

> Part of a small collection of virtual try-on model setups in this repo — see the
> [root README](../README.md) for the others (OmniTry, FitDiT). This one also serves as
> the base model for [`fitcontroler/`](../fitcontroler/README.md), a from-scratch
> reimplementation of the FitControler fit-control plug-in.

Setup and CLI inference for [IDM-VTON](https://github.com/yisol/IDM-VTON) — "Improving
Diffusion Models for Authentic Virtual Try-on in the Wild"
([paper](https://arxiv.org/abs/2403.05139)) — on a single RTX 4090 (24GB).

**⚠️ Non-commercial use only.** IDM-VTON's weights are licensed **CC BY-NC-SA-4.0**.

## How this differs from the other two

IDM-VTON is the heaviest, most multi-stage pipeline of the three models in this repo:

1. **OpenPose + human parsing** on the person photo → an auto-generated try-on mask.
2. **DensePose** (a vendored `detectron2` + `densepose`, invoked through upstream's own
   `apply_net.py`) → a pose/segmentation conditioning image.
3. **Dual-UNet SDXL-inpainting diffusion** — one UNet branch encodes the garment, the
   other does the actual person/scene inpainting, cross-conditioned via IP-Adapter-style
   attention — generates the final image inside the mask.

Architecturally it's SDXL-scale (two SDXL UNets + two CLIP text encoders + a CLIP vision
encoder + VAE), and unlike FitDiT/OmniTry it pulls in `detectron2`/`densepose` as a real
dependency, which means compiling CUDA ops at install time — the single most likely
source of friction in this setup (see Troubleshooting).

## Honesty check: what's verified here

The model-loading block, `pil_to_binary_mask`, `tensor_transform`, and the body of
`IDMVTONPipeline.run_tryon()` in `inference.py` are adapted — verbatim where possible —
from upstream's own `start_tryon()` callback in `gradio_demo/app.py` (fetched and quoted
directly from the upstream GitHub repo), not independently re-derived, so the actual
preprocessing/diffusion math shouldn't silently diverge from upstream's real behavior.
It's restructured into a plain CLI function rather than importing `app.py` directly,
because `app.py` builds its Gradio UI at module level with no `if __name__ ==
"__main__":` guard visible in what could be fetched — importing it as a module risks
launching a web server as a side effect.

**What is NOT verified: none of this has been run on real 4090 hardware by whoever wrote
this setup** (no GPU access in the environment that authored it, and upstream's arXiv
page and Hugging Face were both unreachable from that environment's network policy, so
even the paper itself could only be cross-checked via GitHub source, not read directly).
The vendored `detectron2` build in particular is a well-known friction point for this
upstream repo in general — please report back what actually works.

## 1. Install

```bash
git clone <this-repo>
cd claude_virtual_try_ON/idm-vton
./scripts/install.sh
```

This creates `.venv`, installs pinned dependencies (`torch==2.0.1`/cu118 — upstream's own
pin, and cu118 already supports the 4090's Ada Lovelace architecture, no bump needed),
clones upstream IDM-VTON into `third_party/IDM-VTON`, and builds the vendored
`detectron2` package at `third_party/IDM-VTON/gradio_demo/detectron2` (needed for
DensePose) via `pip install -e`.

## 2. Download checkpoints

IDM-VTON needs weights from **two separate Hugging Face repos** — see
`scripts/download_checkpoints.py`'s docstring for exactly why:

```bash
python scripts/download_checkpoints.py
```

This populates:
```
checkpoints/IDM-VTON/                        # the diffusers-format model itself
third_party/IDM-VTON/ckpt/densepose/          # DensePose weights
third_party/IDM-VTON/ckpt/humanparsing/       # SCHP human-parsing ONNX models
third_party/IDM-VTON/ckpt/openpose/ckpts/     # OpenPose body-pose weights
```

The last three live *inside* the cloned upstream source, not under `checkpoints/`,
because upstream's own preprocessing code hardcodes those paths relative to the IDM-VTON
repo root (`inference.py` `chdir()`s there before running, to match).

Neither repo appeared gated as of when this was written — no `huggingface-cli login`
should be required, but if you hit a 403, request access on the relevant repo page and
authenticate first.

## 3. Run inference

```bash
python inference.py \
  --person examples/person.jpg \
  --garment examples/garment.jpg \
  --garment-desc "short sleeve round neck t-shirt" \
  --output out.png
```

`--garment-desc` is a short text description fed straight into the prompt exactly as
upstream does (`"model is wearing " + garment_desc`) — it meaningfully affects output
quality, so describe the garment the way upstream's own examples do (fit, sleeve length,
neckline, etc.), not just "shirt".

Other flags:
- `--category {upper_body,lower_body,dresses}` (default `upper_body`) — which region the
  garment covers.
- `--model-type {hd,dc}` (default `hd`) — `hd` matches upstream's own Gradio demo default
  (VITON-HD-style masking); `dc` (DressCode-style) is what you likely want for
  `--category lower_body`/`dresses`, though that combination isn't verified here.
- `--mask path/to/mask.png` — use a hand-drawn binary mask instead of the default
  auto-generated one (OpenPose + human parsing).
- `--crop` — auto-crop to a 3:4 region before try-on (upstream's "crop & resize" option).
- `--steps` (default 30), `--guidance-scale` (default 2.0), `--seed` (default random) —
  match upstream's own `inference.py` CLI defaults / `gradio_demo/app.py`'s hardcoded
  values.
- `--model-path` — defaults to the bare HF repo id `yisol/IDM-VTON` (loads straight from
  the Hub); point it at `checkpoints/IDM-VTON` to use the local download instead.
- `--save-mask path/to/mask_preview.png` — also save the grayed-out mask preview.

## Troubleshooting

- **`detectron2` build fails during `install.sh`** (missing `nvcc`, or a CUDA
  version mismatch): the vendored `detectron2`/`densepose` packages compile CUDA
  extensions against the installed `torch==2.0.1`+cu118 build. Make sure a matching CUDA
  toolkit is on `PATH` (e.g. `apt-get install nvidia-cuda-toolkit`, or set `CUDA_HOME` to
  point at one) and retry `pip install -e third_party/IDM-VTON/gradio_demo/detectron2`.
  This is a well-known friction point for this upstream repo generally, not something
  specific to this wrapper.
- **`ModuleNotFoundError: src` / `No module named 'preprocess'` / `No module named
  'apply_net'`**: run `./scripts/install.sh` first — it clones upstream into
  `third_party/IDM-VTON`, which `inference.py` adds to `sys.path` at import time (both
  the repo root and `gradio_demo/`, matching how upstream's own `app.py` resolves its
  imports).
- **`FileNotFoundError` around `ckpt/densepose/model_final_162be9.pkl` or
  `ckpt/humanparsing/*.onnx`**: run `scripts/download_checkpoints.py` — those paths are
  hardcoded by upstream relative to the IDM-VTON repo root, not configurable via a flag.
- **CUDA OOM**: this is an SDXL-scale pipeline (two UNets); there's no offload flag
  wired up here yet (unlike omnitry/fitdit) since upstream's own demo doesn't expose one
  — if this becomes a real problem on a 4090, the next lever would be adding
  `enable_model_cpu_offload()`/`enable_sequential_cpu_offload()` calls onto `self.pipe`
  in `IDMVTONPipeline.__init__`, not implemented here.
- **403 / gated repo error downloading checkpoints**: request access on the relevant
  Hugging Face repo page (model repo `yisol/IDM-VTON`, or the Space of the same name) and
  run `huggingface-cli login`.

## Credit

Model and original code: [yisol/IDM-VTON](https://github.com/yisol/IDM-VTON) (paper:
[arXiv:2403.05139](https://arxiv.org/abs/2403.05139)). This repo only adds an install
script, a checkpoint downloader, and a CLI wrapper (`inference.py`) around upstream's own
pipeline classes and preprocessing modules — no reimplementation of the
mask/pose/DensePose/diffusion logic itself.
