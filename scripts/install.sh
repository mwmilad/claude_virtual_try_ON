#!/usr/bin/env bash
# Installs OmniTry (FLUX.1-Fill-dev + dual-LoRA virtual try-on) for a single RTX 4090.
#
# Usage: ./scripts/install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== checking GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found -- install the NVIDIA driver (>= 550.x recommended for a 4090) first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM_MB}" -lt 20000 ]; then
  echo "warning: detected GPU has < 20GB VRAM. OmniTry needs a 24GB-class card plus the" >&2
  echo "         offload/quantization flags documented in README.md even to fit a 4090." >&2
fi

echo "== creating virtualenv (.venv) =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "== installing project requirements (PyTorch cu126 build, Ada/RTX 4090 compatible) =="
pip install -r requirements.txt

echo "== cloning upstream OmniTry source (model code, not weights) =="
mkdir -p third_party
if [ ! -d third_party/OmniTry/.git ]; then
  git clone https://github.com/Kunbyte-AI/OmniTry.git third_party/OmniTry
else
  echo "third_party/OmniTry already present, skipping clone"
fi

echo "== optional: flash-attn (alternate attention kernel; SDPA/flash-sdp is used by default without it) =="
pip install flash-attn==2.6.3 --no-build-isolation || echo "flash-attn build skipped/failed -- fine, falling back to PyTorch SDPA"

cat <<'EOF'

Install complete. Next steps:
  1. Request access to the gated FLUX.1-Fill-dev model:
       https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
     then authenticate:
       huggingface-cli login
  2. Download checkpoints (~35GB total):
       python scripts/download_checkpoints.py
  3. Run inference (see README.md for --offload / --quantize choices on a 4090):
       python inference.py \
         --person examples/person.jpg \
         --object examples/garment.jpg \
         --class "top clothes" \
         --output out.png
EOF
