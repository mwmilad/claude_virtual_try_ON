#!/usr/bin/env python3
"""Downloads FLUX.1-Fill-dev and the OmniTry LoRA weights into ./checkpoints.

FLUX.1-Fill-dev is a gated model on Hugging Face: request access at
https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev and run
`huggingface-cli login` before running this script.
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora-variant",
        choices=["unified", "clothes"],
        default="unified",
        help="unified = all object classes (default), clothes = clothing-only LoRA",
    )
    parser.add_argument(
        "--skip-base-model",
        action="store_true",
        help="skip the ~34GB FLUX.1-Fill-dev download (use if already present)",
    )
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_base_model:
        print("Downloading black-forest-labs/FLUX.1-Fill-dev ...")
        snapshot_download(
            repo_id="black-forest-labs/FLUX.1-Fill-dev",
            local_dir=CHECKPOINT_DIR / "FLUX.1-Fill-dev",
        )

    lora_filename = f"omnitry_v1_{args.lora_variant}.safetensors"
    print(f"Downloading Kunbyte/OmniTry:{lora_filename} ...")
    hf_hub_download(
        repo_id="Kunbyte/OmniTry",
        filename=lora_filename,
        local_dir=CHECKPOINT_DIR,
    )

    print("Done. Checkpoints in:", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
