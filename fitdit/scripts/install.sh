#!/usr/bin/env bash
# Installs FitDiT (dual SD3-transformer virtual try-on) for a single RTX 4090.
#
# Usage: ./scripts/install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== checking GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found -- install the NVIDIA driver first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== creating virtualenv (.venv) =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

echo "== installing project requirements (pinned torch 2.4.0 -- CUDA wheels support the 4090 out of the box) =="
pip install -r requirements.txt

echo "== cloning upstream FitDiT source (model code, not weights) =="
mkdir -p third_party
if [ ! -d third_party/FitDiT/.git ]; then
  git clone https://github.com/BoyuanJiang/FitDiT.git third_party/FitDiT
else
  echo "third_party/FitDiT already present, skipping clone"
fi

cat <<'EOF'

Install complete. Next steps:
  1. Request access to the gated FitDiT model weights:
       https://huggingface.co/BoyuanJiang/FitDiT
     then authenticate:
       huggingface-cli login
  2. Download checkpoints:
       python scripts/download_checkpoints.py
  3. Run inference (see README.md -- this is a two-step mask-then-generate flow, and
     the model is NON-COMMERCIAL USE ONLY, CC BY-NC-SA-4.0):
       python inference.py \
         --person examples/person.jpg \
         --garment examples/garment.jpg \
         --category "Upper-body" \
         --output out.png
EOF
