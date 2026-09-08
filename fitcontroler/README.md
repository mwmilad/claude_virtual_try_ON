# FitControler (on IDM-VTON) — best-effort reimplementation

> Part of the [virtual try-on collection](../README.md) in this repo. This one is
> different in kind from [`omnitry/`](../omnitry/README.md) and
> [`fitdit/`](../fitdit/README.md): those two are thin CLI wrappers around **verbatim
> upstream code and pretrained weights**. This directory is a **from-scratch
> reimplementation of a research paper with no public code, no public data, and no
> pretrained weights** — read this whole README before using it, especially the "What
> you actually get" section.

Paper: **"FitControler: Toward Fit-Aware Virtual Try-On"**, Lu Yang, Yicheng Liu, Yanan
Li, Xiang Bai, Hao Lu — [arXiv:2512.24016](https://arxiv.org/abs/2512.24016), submitted
Dec 30, 2025 (Huazhong University of Science and Technology / Wuhan Institute of
Technology).

## Why this is labeled "best-effort," not "a port"

Unlike `fitdit/` and `omnitry/`, where this repo could clone upstream's actual source and
call its actual classes, **there is no upstream source for FitControler to clone**: as of
when this was written, its code and data ("Fit4Men," the paper's own dataset) had not
been released, and this environment's network policy blocked direct access to both arXiv
and Hugging Face — so even the paper's full text, equations, and figures could not be
fetched directly. What informed this implementation is limited to abstract-level
descriptions surfaced through web search (search-engine summaries and a ResearchGate
listing), cross-checked against upstream **IDM-VTON**'s actual GitHub source (which
*was* fetchable) for how the base VTON model this plugs into actually works.

Concretely, here's what the search-derived material said, verbatim as summarized:

> FitControler is a learnable plug-in that can seamlessly integrate into modern VTON
> models to enable customized fit control... features a fit-aware layout generator to
> redraw the body-garment layout conditioned on a set of delicately processed
> garment-agnostic representations, and a multi-scale fit injector is then used to
> deliver layout cues to enable layout-driven VTON... built a fit-aware VTON dataset
> termed Fit4Men, including 13,000 body-garment pairs of different fits, covering both
> tops and bottoms... Two fit consistency metrics are also introduced to assess the
> fitness of generations.

That is the entirety of the technical grounding available. Everything below this line —
exact architecture, channel counts, conditioning mechanism, loss functions, the fit
vocabulary, injection points, training recipe — is this repo's own invention, built to be
*structurally consistent* with those two named components, not a verified reproduction
of the paper's actual design or numbers.

### What's faithful vs. invented

| Aspect | Status |
|---|---|
| Two-component design (layout generator + multi-scale injector) | Faithful to the abstract |
| "Plug-in" that attaches to an existing VTON model without retraining it | Faithful to the abstract |
| Conditions on a garment-agnostic representation + fit signal | Faithful in spirit |
| Base VTON model = IDM-VTON specifically | Not from the paper (it says "modern VTON models," plural) — chosen here because IDM-VTON is the other SDXL-class model already in this repo's collection and its source was fetchable to integrate against |
| Exact layout-generator architecture (FiLM-conditioned conv U-Net) | **Invented** |
| Exact injector mechanism (ControlNet-style zero-init 1x1 convs on UNet down_blocks) | **Invented** (a standard, defensible choice for "plug-in without retraining," not from the paper) |
| Fit vocabulary (`tight/fitted/regular/loose/oversized`) | **Invented** — the paper doesn't specify its fit representation at the abstract level |
| Garment-agnostic representation contents (agnostic image + default mask + DensePose viz) | **Invented**, reusing pieces IDM-VTON's own mask stage already computes |
| Loss functions (layout BCE + VTON diffusion MSE) | **Invented** |
| Fit4Men dataset | **Not available anywhere** — `train.py`'s `Fit4MenDataset` is a documented format stub, not real data |
| Two fit-consistency evaluation metrics the paper introduces | **Not implemented** — their definitions weren't in the available material |
| Pretrained weights | **Do not exist.** `model.py`'s networks are randomly initialized every time |

## What you actually get

- `model.py` — `FitAwareLayoutGenerator`, `MultiScaleFitInjector`, and the top-level
  `FitControler` wrapper that attaches to a diffusers `UNet2DConditionModel` via forward
  hooks. Sanity-tested (shapes, the zero-init-injector-is-a-no-op property, a training
  backward pass, and a save/load round-trip) with a synthetic tensor, not against real
  images or a real UNet.
- `inference.py` — wires `FitControler` onto `../idm-vton`'s pipeline via a small
  extension point added to `idm-vton/inference.py`
  (`IDMVTONPipeline.run_tryon(..., on_conditioning_ready=...)`), so the plug-in reuses
  IDM-VTON's own mask/pose/DensePose outputs instead of recomputing them.
- `train.py` — a training-step **skeleton**, structurally grounded in upstream
  IDM-VTON's actual `train_xl.py` training step (VAE encode → add noise →
  `unet_encoder` reference features → main `unet` call → MSE loss — fetched and quoted
  from upstream, not guessed), adapted to freeze the *entire* base pipeline and train
  only `FitControler`'s own parameters. This is a deliberate, documented departure from
  upstream's own recipe (which fine-tunes the main `unet` itself) — the plug-in framing
  in the abstract implied a frozen-backbone approach, ControlNet-style.

**Without a trained `--weights` checkpoint, `inference.py` is an expensive no-op.**
`MultiScaleFitInjector` is zero-initialized by construction (see `model.py`), so an
untrained plug-in adds an exact zero residual to the frozen IDM-VTON UNet — `--fit`
changes nothing, and you'll just get plain IDM-VTON output, slower, with a warning
printed. There is nothing to download to fix this; `train.py` needs real fit-labeled
data this repo does not have and cannot supply.

## Usage

```bash
# 1. set up the base model first
cd ../idm-vton && ./scripts/install.sh && python scripts/download_checkpoints.py && cd ../fitcontroler

# 2. run (only meaningfully different from plain IDM-VTON once you have a trained
#    checkpoint from train.py -- see the warning above)
python inference.py \
  --person ../idm-vton/examples/person.jpg \
  --garment ../idm-vton/examples/garment.jpg \
  --garment-desc "short sleeve round neck t-shirt" \
  --fit loose \
  --weights checkpoints/fitcontroler.pt \
  --output out.png
```

`--fit` accepts `tight`, `fitted`, `regular`, `loose`, `oversized` (see caveat above:
this vocabulary is invented, not from the paper).

## Training

```bash
python train.py --data-root /path/to/your/fit-labeled/data --output-dir checkpoints
```

This will immediately raise `NotImplementedError`/`FileNotFoundError` until you (a) have
your own fit-labeled dataset and (b) implement `Fit4MenDataset.__getitem__` in
`train.py` against it — see that class's docstring for the exact fields
`training_step()` expects (`Fit4MenSample`). There is no shortcut here: Fit4Men itself is
not public.

## License

IDM-VTON's weights are **CC BY-NC-SA-4.0, non-commercial only** — anything produced by
running `fitcontroler/inference.py` inherits that restriction, since it's IDM-VTON's own
frozen UNet doing the actual image generation. FitControler paper's own license (for
Fit4Men, if/when released) is unknown.

## Credit

Paper: Lu Yang, Yicheng Liu, Yanan Li, Xiang Bai, Hao Lu,
["FitControler: Toward Fit-Aware Virtual Try-On"](https://arxiv.org/abs/2512.24016),
arXiv:2512.24016. No official code or data release to credit as of when this was
written — everything under this directory is this repo's own (best-effort,
uncertified) implementation, built on top of [yisol/IDM-VTON](../idm-vton/README.md).
