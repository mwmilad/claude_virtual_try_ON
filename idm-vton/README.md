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
dependency. `gradio_demo/detectron2` ships as a **prebuilt binary**
(`_C.cpython-39-x86_64-linux-gnu.so`) rather than source to compile — there's no build
step, but it only imports under **Python 3.9 exactly**, since that's a real ABI
constraint baked into the filename, not a suggestion. That Python-version requirement is
the single most likely source of friction in this setup (see Troubleshooting).

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
This showed up concretely once real hardware was involved: `install.sh` originally tried
`pip install -e` on the vendored `detectron2` (assuming it was source to compile), which
fails immediately — it's actually a prebuilt `.so`, not a package with a `setup.py`. Fixed
once, but take it as a sign the rest of this setup is still under-tested too — please
report back what actually works (or doesn't).

## 1. Install

```bash
git clone <this-repo>
cd claude_virtual_try_ON/idm-vton
./scripts/install.sh
```

This creates `.venv` **using `python3.9` specifically** (required — see above), installs
pinned dependencies (`torch==2.0.1`/cu118 — upstream's own pin, and cu118 already
supports the 4090's Ada Lovelace architecture, no bump needed), clones upstream IDM-VTON
into `third_party/IDM-VTON`, and checks that the vendored `detectron2` at
`third_party/IDM-VTON/gradio_demo/detectron2` — a prebuilt binary, not something this
script builds or installs — actually imports under this venv's Python.

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

### Already have these checkpoints somewhere?

If you've already downloaded IDM-VTON on this machine (e.g. via this same
`download_checkpoints.py` run elsewhere, or any earlier `snapshot_download`/
`from_pretrained` call), you don't need to fetch it again. `--existing-ckpt-dir` expects
one directory containing, as direct siblings:

```
<existing-ckpt-dir>/
├── densepose/          # -> symlinked to third_party/IDM-VTON/ckpt/densepose
├── humanparsing/        # -> symlinked to third_party/IDM-VTON/ckpt/humanparsing
├── openpose/             # -> symlinked to third_party/IDM-VTON/ckpt/openpose
└── models--yisol--IDM-VTON/    # a standard Hugging Face cache entry (from
                                  # snapshot_download/from_pretrained caching to this dir)
```

```bash
python inference.py \
  --existing-ckpt-dir /path/to/existing/ckpt \
  --person examples/person.jpg \
  --garment examples/garment.jpg \
  --garment-desc "short sleeve round neck t-shirt" \
  --output out.png
```

This symlinks `third_party/IDM-VTON/ckpt` to the `densepose`/`humanparsing`/`openpose`
subfolders (skipping `scripts/download_checkpoints.py` entirely for those), and passes
your directory as `cache_dir` with `local_files_only=True` to every `from_pretrained`
call for the model weights — so it resolves `models--yisol--IDM-VTON` from your local
cache and never touches the network, even to check for updates. Leave `--model-path` at
its default `yisol/IDM-VTON` repo id when using this (that's the id `from_pretrained`
looks up inside the cache — it isn't a literal filesystem path here). If your directory
is missing any of the four expected files, `inference.py` prints which ones before
proceeding (not a hard failure — your layout might just differ slightly).

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
- `--existing-ckpt-dir` — skip downloading entirely by pointing at checkpoints you
  already have (see "Already have these checkpoints somewhere?" above).
- `--save-mask path/to/mask_preview.png` — also save the grayed-out mask preview.

## Troubleshooting

- **`ERROR: ... does not appear to be a Python project: neither 'setup.py' nor
  'pyproject.toml' found`, pointing at `gradio_demo/detectron2`**: this was a real bug in
  an earlier version of `install.sh`, which wrongly tried `pip install -e` on it.
  `gradio_demo/detectron2` isn't installable source — it's a **prebuilt binary**
  (`_C.cpython-39-x86_64-linux-gnu.so`) meant to be imported straight off `sys.path`
  (`inference.py` already adds `gradio_demo/` to `sys.path` for exactly this). If you're
  seeing this error, pull the latest `install.sh` — it no longer attempts to install it,
  just checks that it imports.
- **`detectron2` fails to import** (`ImportError`, `undefined symbol`, or a segfault),
  even after pulling the fix above: the prebuilt `.so` is filename-tagged
  `cpython-39` — it needs the venv's Python to be **exactly 3.9**, and it likely also
  needs the installed `torch`/CUDA build to match whatever it was originally linked
  against (unverified from this environment — no GPU to test). `install.sh` now creates
  `.venv` with `python3.9` specifically and fails fast with install guidance if
  `python3.9` isn't on `PATH` — if you already have a `.venv` built with a different
  Python, delete it and rerun `install.sh` (`rm -rf .venv && ./scripts/install.sh`). If
  it still won't import even under 3.9, the fallback is installing the *official*
  `detectron2` (not the vendored copy) from source, matched to your torch/CUDA build —
  `pip install 'git+https://github.com/facebookresearch/detectron2.git'` — not wired up
  or tested here; you'd also need `densepose` importable (upstream vendors that
  separately at `gradio_demo/densepose/`, pure Python, should be unaffected by this).
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
