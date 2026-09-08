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
# Must be exactly Python 3.9: the vendored detectron2 (see below) ships as a PREBUILT
# binary, third_party/IDM-VTON/gradio_demo/detectron2/_C.cpython-39-*.so -- that
# filename's cpython-39 tag is a real ABI constraint, not a suggestion. Under any other
# Python version `import detectron2` fails (ImportError / undefined symbol), not just
# "might be slower."
if command -v python3.9 >/dev/null 2>&1; then
  PYTHON_BIN=python3.9
else
  echo "python3.9 not found on PATH -- required for the vendored detectron2's prebuilt" >&2
  echo "_C.cpython-39-*.so to import at all. Install it first, e.g.:" >&2
  echo "  sudo apt-get install python3.9 python3.9-venv   # Debian/Ubuntu" >&2
  echo "  # or via pyenv: pyenv install 3.9.18" >&2
  exit 1
fi
"$PYTHON_BIN" -m venv .venv
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

echo "== checking the vendored detectron2 imports (prebuilt binary, no build step) =="
echo "   third_party/IDM-VTON/gradio_demo/detectron2 ships a prebuilt _C.cpython-39-*.so"
echo "   -- there is no setup.py, nothing to 'pip install -e' here. It works by adding"
echo "   gradio_demo/ to sys.path (inference.py already does this), so this just"
echo "   verifies the .so actually loads under this venv's Python."
if python -c "
import sys
sys.path.insert(0, 'third_party/IDM-VTON/gradio_demo')
import detectron2
print('OK:', detectron2.__file__)
"; then
  :
else
  echo "detectron2 failed to import -- see idm-vton/README.md's Troubleshooting section" >&2
  echo "(most likely cause: this venv isn't exactly Python 3.9, or torch/CUDA here" >&2
  echo "doesn't match whatever the prebuilt .so was linked against)." >&2
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
