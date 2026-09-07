#!/usr/bin/env python3
"""Downloads the FitDiT model weights into ./checkpoints.

FitDiT is a gated model on Hugging Face: request access at
https://huggingface.co/BoyuanJiang/FitDiT and run `huggingface-cli login`
before running this script.

This does a full snapshot download rather than picking individual files, because
FitDiTGenerator (upstream's gradio_sd3.py) loads several components straight out of
model_root by subfolder name (transformer_garm/, transformer_vton/, pose_guider/, plus
whatever the bundled DWpose/human-parsing preprocessors need) -- there's no shorter list
of "the files that matter" without risking missing one.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading BoyuanJiang/FitDiT (full snapshot) ...")
    snapshot_download(repo_id="BoyuanJiang/FitDiT", local_dir=CHECKPOINT_DIR)
    print("Done. Checkpoints in:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
