# Virtual Try-On on a single RTX 4090

Setup and CLI inference for a small collection of open-source virtual try-on models,
each tuned to run on a single RTX 4090 (24GB). Each model lives in its own subdirectory
with its own install script, checkpoint downloader, and `inference.py`.

## Models

| | Approach | License | Status |
|---|---|---|---|
| [`omnitry/`](omnitry/README.md) | Mask-free — FLUX.1-Fill-dev + dual LoRA, one pass | — | Working (`--offload sequential`); fp8 quantization confirmed broken, see its README |
| [`fitdit/`](fitdit/README.md) | Two-step: pose/parsing mask, then dual SD3-transformer generation | **CC BY-NC-SA-4.0 — non-commercial only** | Code is a faithful port of upstream; not yet verified on real 4090 hardware |
| [`idm-vton/`](idm-vton/README.md) | Multi-step: OpenPose + human parsing + DensePose, then dual-UNet SDXL-inpainting diffusion | **CC BY-NC-SA-4.0 — non-commercial only** | CLI is adapted from upstream's own `gradio_demo/app.py`; not yet verified on real 4090 hardware |
| [`fitcontroler/`](fitcontroler/README.md) | Fit-control plug-in (layout generator + multi-scale injector) on top of `idm-vton/` | inherits IDM-VTON's CC BY-NC-SA-4.0 | **From-scratch reimplementation, no official code/data/weights exist** — read its README before using; untrained, it's a no-op |

Each subdirectory is fully self-contained — its own `requirements.txt`, `.venv`, and
`third_party/<upstream>` clone — so they don't share or conflict with each other's
dependencies (several pin different, incompatible `torch` versions). `fitcontroler/` is
the exception: it's a plug-in that imports `idm-vton/`'s own `inference.py` directly, so
set that one up first.

## Quick start

```bash
git clone <this-repo>
cd claude_virtual_try_ON/<omnitry|fitdit|idm-vton>
./scripts/install.sh
# then follow that directory's README for checkpoint download + usage
```

(`fitcontroler/` has no install script of its own — it's a plug-in on top of
`idm-vton/`, see its README.)

See each subdirectory's README for the details: VRAM/speed tradeoffs, licensing,
troubleshooting, and what's actually been confirmed working versus what's a documented-
but-untested upstream claim.
