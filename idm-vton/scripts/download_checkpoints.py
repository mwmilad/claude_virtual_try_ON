#!/usr/bin/env python3
"""Downloads IDM-VTON's weights from two separate Hugging Face repos.

IDM-VTON needs checkpoints from two *different* repos, which upstream's own gradio_demo
keeps in two different places:

1. The diffusers-format SDXL-inpainting-based model itself (unet/, unet_encoder/, vae/,
   tokenizer/, tokenizer_2/, text_encoder/, text_encoder_2/, image_encoder/, scheduler/)
   -- hosted as the *model* repo "yisol/IDM-VTON", loaded via `from_pretrained(base_path=
   "yisol/IDM-VTON", subfolder=...)` in upstream's gradio_demo/app.py. This script
   downloads it into ./checkpoints/IDM-VTON, and inference.py points `--model-path` at
   that directory (or you can pass the bare repo id "yisol/IDM-VTON" directly to load
   straight from the Hub without a local copy).

2. Preprocessing checkpoints for DensePose, human parsing (SCHP), and OpenPose --
   published as files inside the *Space* "yisol/IDM-VTON" (repo_type="space"), not the
   model repo, at https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt :
       ckpt/densepose/model_final_162be9.pkl
       ckpt/humanparsing/parsing_atr.onnx
       ckpt/humanparsing/parsing_lip.onnx
       ckpt/openpose/ckpts/body_pose_model.pth
   Upstream's preprocessing code hardcodes these as paths *relative to the IDM-VTON repo
   root* (e.g. preprocess/humanparsing/run_parsing.py resolves
   `Path(__file__).parents[2] / "ckpt/humanparsing/parsing_atr.onnx"`, and
   gradio_demo/app.py's DensePose call passes the literal relative path
   './ckpt/densepose/model_final_162be9.pkl'). So this script downloads them straight
   into third_party/IDM-VTON/ckpt/ -- NOT into ./checkpoints -- to match where upstream's
   own code expects to find them. inference.py chdir()s into third_party/IDM-VTON before
   running for the same reason.

Neither repo appeared gated as of when this was written -- no huggingface-cli login
should be required, but if you hit a 403, request access on the relevant repo page and
authenticate first.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "IDM-VTON"
IDM_VTON_SRC = REPO_ROOT / "third_party" / "IDM-VTON"
PREPROC_CKPT_DIR = IDM_VTON_SRC / "ckpt"


def main() -> None:
    if not IDM_VTON_SRC.exists():
        raise SystemExit(
            f"{IDM_VTON_SRC} not found -- run scripts/install.sh first "
            "(it clones upstream IDM-VTON there)."
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading yisol/IDM-VTON (model repo: unet, vae, text/image encoders, ...) ...")
    snapshot_download(repo_id="yisol/IDM-VTON", local_dir=CHECKPOINT_DIR)

    PREPROC_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading yisol/IDM-VTON (Space: DensePose / human-parsing / OpenPose ckpts) ...")
    snapshot_download(
        repo_id="yisol/IDM-VTON",
        repo_type="space",
        allow_patterns=["ckpt/*"],
        local_dir=IDM_VTON_SRC,
    )

    print("Done.")
    print("  model weights:        ", CHECKPOINT_DIR)
    print("  preprocessing weights:", PREPROC_CKPT_DIR)


if __name__ == "__main__":
    main()
