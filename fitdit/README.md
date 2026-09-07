# FitDiT on a single RTX 4090

> Part of a small collection of virtual try-on model setups in this repo — see the
> [root README](../README.md) for the other one (OmniTry).

Setup and CLI inference for [FitDiT](https://github.com/BoyuanJiang/FitDiT) — "Advancing
the Authentic Garment Details for High-fidelity Virtual Try-on"
([paper](https://arxiv.org/abs/2411.10499)) — on a single RTX 4090 (24GB).

**⚠️ Non-commercial use only.** FitDiT's weights are licensed **CC BY-NC-SA-4.0**. For
commercial use, upstream points to
[Tencent Cloud](https://cloud.tencent.com/document/product/1668/108532). Make sure that's
compatible with what you're building before you rely on this.

## How this differs from OmniTry

OmniTry (the other model in this repo) is mask-free — you hand it a person photo and a
garment photo and it figures out the rest with one FLUX + LoRA pass. FitDiT is a genuine
**two-step** pipeline:

1. **Mask generation** — runs pose detection (DWpose) and human parsing on the person
   photo to figure out which region of the image should be replaced (with adjustable
   offsets if the auto-generated mask doesn't cover what you want).
2. **Try-on generation** — a dual diffusion-transformer setup (one SD3 transformer branch
   for the garment, one for the person) generates the final image inside that mask.

Architecturally it's also a different scale of model than OmniTry: two SD3-based
transformers plus two CLIP image encoders plus a pose-guider network, rather than one
FLUX.1-Fill-dev (~12B params) transformer. Upstream doesn't publish exact VRAM numbers,
but does document four speed/memory modes (see below) — meaningfully more configurable
than OmniTry's offload knob, and this one hasn't hit the numerical instability that
fp8 quantization did on the OmniTry side.

## Honesty check: what's verified here

Everything in `inference.py` is a direct, faithful port of upstream's own
`FitDiTGenerator` class from `gradio_sd3.py` — same calls, same argument order, same
one non-obvious detail worth calling out: `generate_mask()` returns the pose image as a
PIL Image, but `process()` expects a numpy array (in the real Gradio app, this
conversion happens implicitly via a `gr.Image` component's default `type="numpy"`);
`inference.py` replicates that conversion explicitly.

**What is NOT verified: none of this has been run on real 4090 hardware by whoever wrote
this setup** (no GPU access in that environment). The code faithfully mirrors upstream,
but the memory-mode recommendation below is upstream's own documented behavior, not a
measurement. Please report back what actually works so this section can be tightened
into real numbers, the way `omnitry/README.md`'s VRAM table was corrected after testing.

## 1. Install

```bash
git clone <this-repo>
cd claude_virtual_try_ON/fitdit
./scripts/install.sh
```

This creates `.venv`, installs pinned dependencies (`torch==2.4.0` — standard PyPI
wheels already support the 4090's Ada Lovelace architecture, no special index needed),
and clones the upstream FitDiT source into `third_party/FitDiT` (model code only, no
weights).

## 2. Download checkpoints

FitDiT is a **gated** model on Hugging Face:

```bash
# 1. request access at https://huggingface.co/BoyuanJiang/FitDiT
# 2. authenticate:
huggingface-cli login
# 3. download the full model snapshot:
python scripts/download_checkpoints.py
```

This does a full snapshot download into `checkpoints/` rather than picking individual
files — `FitDiTGenerator` loads several components straight out of that directory by
subfolder name (`transformer_garm/`, `transformer_vton/`, `pose_guider/`, plus whatever
the bundled pose/parsing preprocessors need), and there's no shorter list of "the files
that matter" without risking missing one.

Two CLIP image encoders (`openai/clip-vit-large-patch14` and
`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`) are downloaded separately and automatically
by `transformers` the first time you run inference — they're public, not gated, but you
do need internet access (or a warm HF cache) at inference time, not just at setup time.

## 3. Run inference

```bash
python inference.py \
  --person examples/person.jpg \
  --garment examples/garment.jpg \
  --category "Upper-body" \
  --output out.png
```

`--category` must be one of `Upper-body`, `Lower-body`, `Dresses` — it tells the mask
step which region of the person photo to replace.

If the auto-generated mask doesn't cover the right area (e.g. sleeves getting cut off),
adjust it with `--mask-offset-top` / `--mask-offset-bottom` / `--mask-offset-left` /
`--mask-offset-right` (pixels, upstream's slider range is -200 to 200) and re-run —
there's no interactive brush tool here like the Gradio demo has, so offsets are your
only lever from the CLI.

Other flags:
- `--resolution` — `768x1024` (fastest/least VRAM), `1152x1536` (default, matches
  upstream's own default), or `1536x2048` (slowest/most VRAM/detail)
- `--steps` (default 20, upstream's range is 15-30), `--guidance-scale` (default 2.0,
  upstream's range is 1.0-5.0), `--seed` (default random), `--num-images` (1-4 per run)
- `--fp16` — load in fp16 instead of the default bf16
- `--offload {none,model,aggressive}` (default `model`) — see the memory modes above

## Troubleshooting

- **CUDA OOM**: step down through `--offload none` → `model` (default) → `aggressive`,
  and/or drop `--resolution` to `768x1024`.
- **403 / gated repo error downloading the model**: request access on the
  [model page](https://huggingface.co/BoyuanJiang/FitDiT) and run `huggingface-cli
  login` with a token that has accepted it.
- **`ModuleNotFoundError: gradio_sd3` / `No module named 'src'` / `No module named
  'preprocess'`**: run `./scripts/install.sh` first — it clones the upstream source into
  `third_party/FitDiT`, which `inference.py` adds to `sys.path` at import time.
- **Slow first run / network calls to huggingface.co**: expected — the two CLIP image
  encoders download on first use unless already cached (see "Download checkpoints"
  above).

## Credit

Model and original code: [BoyuanJiang/FitDiT](https://github.com/BoyuanJiang/FitDiT)
(paper: [arXiv:2411.10499](https://arxiv.org/abs/2411.10499)). This repo only adds an
install script, a checkpoint downloader, and a CLI wrapper (`inference.py`) around
upstream's own `FitDiTGenerator` class from `gradio_sd3.py` — no reimplementation of the
mask/pose/generation logic.
