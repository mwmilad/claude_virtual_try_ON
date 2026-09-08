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
| Fit4Men dataset | **Not available anywhere** — `train.py`'s `Fit4MenDataset` remains a documented format stub for it, not real data. `train.py` trains against a real stand-in dataset instead by default — see "Dataset" below |
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
- `fit_datasets.py` — loads a real, downloadable dataset (**GarmentCodeVTON**, see
  "Dataset" below) as a stand-in for the paper's own unreleased Fit4Men.
- `scripts/download_dataset.py` — downloads it (`huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset`,
  plus an optional real-world eval set) into `data/`.
- `train.py` — an actual training loop (not just a skeleton) against that dataset,
  structurally grounded in upstream IDM-VTON's real `train_xl.py` training step (VAE
  encode → add noise → `unet_encoder` reference features → main `unet` call → MSE loss —
  fetched and quoted from upstream, not guessed), adapted to freeze the *entire* base
  pipeline and train only `FitControler`'s own parameters. This is a deliberate,
  documented departure from upstream's own recipe (which fine-tunes the main `unet`
  itself) — the plug-in framing in the abstract implied a frozen-backbone approach,
  ControlNet-style. Fit4Men itself is still supported as a documented format stub via
  `--dataset fit4men` if you ever get real access to it (or an equivalent).

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
pip install -r requirements.txt   # adds `datasets` for fit_datasets.py/train.py

# 2. run (only meaningfully different from plain IDM-VTON once you have a trained
#    checkpoint from train.py -- see the warning above and "Dataset"/"Training" below)
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

## Dataset

Fit4Men (the paper's own dataset) isn't public, so training here defaults to
**[GarmentCodeVTON](https://huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset)** —
from the FitVTON paper ([arXiv:2606.12012](https://arxiv.org/abs/2606.12012)), chosen
because it's the closest thing that's actually downloadable today:

- **78,080 synthetic try-on triplets** — 19 garment references × 16 body shapes
  (male/female) × 10 poses, generated by draping parametrized garments (via GarmentCode)
  onto varied bodies and physically simulating the drape, so garment **fit varies by
  construction**, not by hand-labeling.
- Fit/size is encoded as a **structured text description** per sample (per the FitVTON
  paper, e.g. *"long-length upper garment"* on a *"slim, medium-tall body"*), not a
  discrete tight/loose column — `fit_datasets.py`'s `parse_fit_level_from_text()` maps
  that text to this repo's invented `tight/fitted/regular/loose/oversized` vocabulary via
  keyword matching, as a bridge, not an official mapping.
- Companion real-world eval set: **[FittingEffect3K](https://huggingface.co/datasets/ZenoNing/FittingEffectDataset)**
  — 3,350 real triplets, 14 garments, 10 models, 5 poses, with body measurements and a
  VLM-based tightness/looseness scoring protocol. Downloadable via `--include-eval`
  below; `train.py` doesn't wire up an eval loop against it (not implemented here).

**A bigger, more explicit alternative worth checking**:
**[FIT](https://johannakarras.github.io/FIT/)** (Google Research + UW,
[arXiv:2604.08526](https://arxiv.org/abs/2604.08526), SIGGRAPH 2026) — 1,064,824
train / 5,000 test triplets, 168 body shapes across **sizes XS–3XL**, explicitly spanning
"loose to tight" fits with ground-truth measurements. Its paper says data/code "will be
made publicly available," but this environment couldn't reach either arXiv or that
project page to confirm current release status or license — check yourself before
switching to it (would need a new `Dataset` class in `fit_datasets.py` alongside
`GarmentCodeVTONDataset`, not written here).

**Schema caveat**: this environment's network policy blocked `huggingface.co` outright
(confirmed by testing `datasets.load_dataset(...)` directly, not just guessing from the
block on browsing) — so `fit_datasets.py`'s column-name mapping (`FIELD_CANDIDATES`) and
the fit-keyword vocabulary (`FIT_KEYWORDS`) are **best-effort guesses**, not verified
against the dataset's real schema. `GarmentCodeVTONDataset.__init__` auto-detects columns
from a candidate list and raises a clear `KeyError` naming the real columns if none
match. Before training for real, run this from a machine with actual Hugging Face access:

```bash
python fit_datasets.py --inspect
```

and fix `FIELD_CANDIDATES`/`FIT_KEYWORDS` in `fit_datasets.py` if it doesn't match.

### Download

```bash
pip install -r requirements.txt   # after ../idm-vton/requirements.txt is already installed
python scripts/download_dataset.py                 # -> data/garmentcode_vton/
python scripts/download_dataset.py --include-eval   # also -> data/fitting_effect_3k/
```

This downloads via the `datasets` library's own caching
(`load_dataset(...).save_to_disk(...)`), so `train.py` reloads instantly from
`data/garmentcode_vton/` on every subsequent run instead of re-hitting the Hub.

## Training

```bash
python train.py --output-dir checkpoints
```

Defaults to `--dataset garmentcode_vton --data-dir data/garmentcode_vton` (the directory
`scripts/download_dataset.py` just populated; falls back to downloading straight from the
Hub if that directory doesn't exist yet). Each step: `train.py`'s `prepare_batch()` runs
IDM-VTON's *own* preprocessing (human parsing, OpenPose, DensePose, prompt encoding) on
the raw triplet to build a `Fit4MenSample` — including deriving `target_layout` (what
`FitAwareLayoutGenerator` is supervised against) by running IDM-VTON's mask stage on the
**fitted-result** image rather than the pre-garment avatar, so it traces the garment's
actual visible extent under that sample's specific simulated fit. That derivation is a
mechanically-grounded proxy, not a verified match for how Fit4Men itself would have been
annotated. See `prepare_batch()`'s docstring in `train.py` for the full reasoning.

**Still not run end-to-end on real hardware** (no GPU access in the environment that
wrote this) — the two most likely failure points on a first real run are (a) a
`GarmentCodeVTONDataset` column mismatch (see the schema caveat above) and (b)
shapes/dtypes that only reveal themselves against the real IDM-VTON pipeline and real
images. Useful flags: `--batch-size`, `--num-workers`, `--lr`, `--epochs`,
`--layout-loss-weight`, `--diffusion-loss-weight`, `--save-every`.

To use Fit4Men itself (or an equivalent you have real access to) instead:
```bash
python train.py --dataset fit4men --data-root /path/to/your/fit-labeled/data
```
This still raises `NotImplementedError`/`FileNotFoundError` until you implement
`Fit4MenDataset.__getitem__` against your data — see that class's docstring for the exact
fields `training_step()` expects (`Fit4MenSample`).

## License

IDM-VTON's weights are **CC BY-NC-SA-4.0, non-commercial only** — anything produced by
running `fitcontroler/inference.py` inherits that restriction, since it's IDM-VTON's own
frozen UNet doing the actual image generation. FitControler paper's own license (for
Fit4Men, if/when released) is unknown. GarmentCodeVTON/FittingEffect3K's license
couldn't be confirmed either (the dataset card wasn't reachable) — check it on the HF
page yourself before training a checkpoint you intend to use commercially.

## Credit

Paper this reimplements: Lu Yang, Yicheng Liu, Yanan Li, Xiang Bai, Hao Lu,
["FitControler: Toward Fit-Aware Virtual Try-On"](https://arxiv.org/abs/2512.24016),
arXiv:2512.24016. No official code or data release to credit as of when this was
written — everything under this directory is this repo's own (best-effort,
uncertified) implementation, built on top of [yisol/IDM-VTON](../idm-vton/README.md).

Dataset: Zeno Ning et al., ["FitVTON: Fit-aware Virtual Try-On via Body-Garment Size
Control"](https://arxiv.org/abs/2606.12012), arXiv:2606.12012 —
[GarmentCodeVTONDataset](https://huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset)
and [FittingEffectDataset](https://huggingface.co/datasets/ZenoNing/FittingEffectDataset)
are their work, used here only as training data, not part of the FitControler
reimplementation itself.
