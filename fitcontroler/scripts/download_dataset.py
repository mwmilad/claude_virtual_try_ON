#!/usr/bin/env python3
"""Downloads the training/eval datasets for fitcontroler/train.py.

Default: **GarmentCodeVTON** (huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset,
from the FitVTON paper, arXiv:2606.12012) -- 78,080 synthetic try-on triplets used as
the (unreleased Fit4Men's) stand-in dataset. See ../fit_datasets.py's module docstring and
../README.md for why this one and not the paper's own dataset.

Also available with --include-eval: **FittingEffect3K**
(huggingface.co/datasets/ZenoNing/FittingEffectDataset) -- 3,350 real-world evaluation
triplets from the same paper, useful as a held-out sanity check even though train.py
doesn't wire up an eval loop against it yet.

Downloads via the `datasets` library's own caching (load_dataset(...).save_to_disk(...)),
not huggingface_hub.snapshot_download, since GarmentCodeVTONDataset is published as a
`datasets`-format dataset (Arrow/Parquet shards), not a loose file tree -- save_to_disk
gives train.py a local Arrow directory it can reload instantly on every run without
re-hitting the Hub.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --include-eval
    python scripts/download_dataset.py --split train --output-dir /data/garmentcode_vton
"""
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "garmentcode_vton"
DEFAULT_EVAL_OUTPUT_DIR = REPO_ROOT / "data" / "fitting_effect_3k"


def download(dataset_name: str, split: str, output_dir: Path) -> None:
    from datasets import load_dataset

    print(f"Downloading {dataset_name} [{split}] ...")
    ds = load_dataset(dataset_name, split=split)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"  {len(ds)} rows -> {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="ZenoNing/GarmentCodeVTONDataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--include-eval", action="store_true",
        help="also download ZenoNing/FittingEffectDataset (FittingEffect3K, real-world eval triplets)",
    )
    parser.add_argument("--eval-output-dir", default=str(DEFAULT_EVAL_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pip install datasets (Hugging Face's `datasets` library, listed in "
            "requirements.txt) first"
        ) from exc

    download(args.dataset, args.split, Path(args.output_dir))

    if args.include_eval:
        download("ZenoNing/FittingEffectDataset", "train", Path(args.eval_output_dir))

    print("Done. Point train.py at these with --data-dir (see train.py --help).")
    print(
        "If loading fails with a KeyError about missing columns, this dataset's schema "
        "wasn't reachable from the environment that wrote fitcontroler/fit_datasets.py -- "
        "run `python ../fit_datasets.py --inspect` and fix FIELD_CANDIDATES there."
    )


if __name__ == "__main__":
    main()
