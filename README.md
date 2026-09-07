# Virtual Try-On on a single RTX 4090

Setup and CLI inference for a small collection of open-source virtual try-on models,
each tuned to run on a single RTX 4090 (24GB). Each model lives in its own subdirectory
with its own install script, checkpoint downloader, and `inference.py`.

## Models

| | Approach | License | Status |
|---|---|---|---|
| [`omnitry/`](omnitry/README.md) | Mask-free — FLUX.1-Fill-dev + dual LoRA, one pass | — | Working (`--offload sequential`); fp8 quantization confirmed broken, see its README |
| [`fitdit/`](fitdit/README.md) | Two-step: pose/parsing mask, then dual SD3-transformer generation | **CC BY-NC-SA-4.0 — non-commercial only** | Code is a faithful port of upstream; not yet verified on real 4090 hardware |

Each subdirectory is fully self-contained — its own `requirements.txt`, `.venv`, and
`third_party/<upstream>` clone — so the two don't share or conflict with each other's
dependencies (they pin different, incompatible `torch` versions).

## Quick start

```bash
git clone <this-repo>
cd claude_virtual_try_ON/<omnitry|fitdit>
./scripts/install.sh
# then follow that directory's README for checkpoint download + usage
```

See each subdirectory's README for the details: VRAM/speed tradeoffs, licensing,
troubleshooting, and what's actually been confirmed working versus what's a documented-
but-untested upstream claim.
