#!/usr/bin/env bash
# Installs IDM-VTON (SDXL-inpainting-based virtual try-on) for a single RTX 4090.
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

echo "== installing project requirements (pinned torch 2.0.1/cu118 -- supports the 4090 out of the box) =="
pip install -r requirements.txt

echo "== cloning upstream IDM-VTON source (model code, not weights) =="
mkdir -p third_party
if [ ! -d third_party/IDM-VTON/.git ]; then
  git clone https://github.com/yisol/IDM-VTON.git third_party/IDM-VTON
else
  echo "third_party/IDM-VTON already present, skipping clone"
fi

echo "== linking vendored detectron2 into idm-vton/ (no build step needed) =="
echo "   third_party/IDM-VTON/gradio_demo/detectron2 is modular (__init__.py"
echo "   everywhere) and has no setup.py -- there's nothing to 'pip install -e' here."
echo "   It imports fine once it's somewhere Python looks: symlinked here into this"
echo "   script's own directory (which Python automatically puts on sys.path when"
echo "   running inference.py/train.py from it) -- confirmed working this way."
if [ -e detectron2 ]; then
  echo "detectron2/ already exists here, leaving it as-is"
else
  ln -s third_party/IDM-VTON/gradio_demo/detectron2 detectron2
  echo "linked ./detectron2 -> third_party/IDM-VTON/gradio_demo/detectron2"
fi

if python -c "import detectron2; print('OK:', detectron2.__file__)"; then
  :
else
  echo "detectron2 failed to import -- see idm-vton/README.md's Troubleshooting section" >&2
  exit 1
fi

cat <<'EOF'

Install complete. Next steps:
  1. Download checkpoints (two separate sources -- see download_checkpoints.py's docstring):
       python scripts/download_checkpoints.py
  2. Run inference (this is a genuine multi-stage pipeline -- pose estimation, human
     parsing, DensePose, then SDXL-inpainting diffusion -- and the model is
     NON-COMMERCIAL USE ONLY, CC BY-NC-SA-4.0):
       python inference.py \
         --person examples/person.jpg \
         --garment examples/garment.jpg \
         --garment-desc "short sleeve round neck t-shirt" \
         --output out.png

If the detectron2 import check above failed, see idm-vton/README.md's Troubleshooting
section -- this is a well-known friction point for this upstream repo in general, not
something specific to this wrapper.
EOF
