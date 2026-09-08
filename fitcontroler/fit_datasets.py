"""Dataset loading for FitControler training.

Primary dataset: **GarmentCodeVTON** (from the FitVTON paper,
[arXiv:2606.12012](https://arxiv.org/abs/2606.12012)) --
https://huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset -- 78,080 synthetic
try-on triplets (19 garment references x 16 body shapes x 10 poses), each a
(body/avatar render, garment image, fitted-result render) triplet plus a text
description of the garment's size/fit on that body. Chosen over the paper's own
(unreleased) Fit4Men dataset because this one is actually downloadable today. See
../README.md's dataset comparison and fitcontroler/README.md for the fuller picture,
including the (still-unreleased-as-of-writing) alternative "FIT" dataset
(arXiv:2604.08526, Google Research + UW, 1.06M triplets, sizes XS-3XL).

**Schema caveat, read before debugging a KeyError**: this environment's network policy
blocks huggingface.co, so the exact column names in FIELD_CANDIDATES below are a
best-effort guess from paper/search-summary descriptions of the dataset, not verified
against its real schema. `__init__` auto-detects columns from a candidate list per
field and raises a clear error naming the columns it actually found if none of the
candidates match -- run `python fit_datasets.py --inspect` first (from an environment with
real Hugging Face access) to confirm or correct FIELD_CANDIDATES, and to correct
FIT_KEYWORDS below against the dataset's actual text vocabulary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from model import FIT_LEVELS

# Best-effort guesses at this dataset's column names, most-likely-first. See the module
# docstring: unverified, auto-detected at load time, correctable via `column_map=` or by
# editing this dict after running `python fit_datasets.py --inspect`.
FIELD_CANDIDATES = {
    "avatar_image": ["avatar_image", "body_image", "person_image", "model_image", "source_image", "agnostic_image", "image_body"],
    "garment_image": ["garment_image", "cloth_image", "cloth", "garment"],
    "result_image": ["result_image", "output_image", "tryon_image", "target_image", "fitted_image", "gt_image", "image"],
    "fit_text": ["fit_text", "size_text", "prompt", "text", "caption", "size_description", "garment_description", "fit_description"],
}

# GarmentCodeVTON encodes fit via structured text prompts (per the FitVTON paper, e.g.
# "long-length upper garment" on a "slim, medium-tall body"), not a discrete label
# column -- this keyword heuristic bridges that free text to model.FIT_LEVELS. Also
# unverified against the dataset's real vocabulary; adjust after inspecting real samples.
FIT_KEYWORDS = {
    "skin-tight": "tight",
    "tight": "tight",
    "slim": "fitted",
    "fitted": "fitted",
    "regular": "regular",
    "standard": "regular",
    "relaxed": "loose",
    "loose": "loose",
    "baggy": "oversized",
    "oversized": "oversized",
}


def parse_fit_level_from_text(text: Optional[str]) -> int:
    """Heuristic keyword match; falls back to 'regular' if nothing matches."""
    text_lower = (text or "").lower()
    for keyword, level in FIT_KEYWORDS.items():
        if keyword in text_lower:
            return FIT_LEVELS.index(level)
    return FIT_LEVELS.index("regular")


def _resolve_column(available: list, candidates: list, field_name: str) -> str:
    for c in candidates:
        if c in available:
            return c
    raise KeyError(
        f"couldn't find a column for '{field_name}' among the dataset's real columns "
        f"{available} -- none of the guessed candidates {candidates} matched. Pass an "
        f"explicit column_map=... to GarmentCodeVTONDataset, or fix FIELD_CANDIDATES in "
        f"fitcontroler/fit_datasets.py after running `python fit_datasets.py --inspect`."
    )


class GarmentCodeVTONDataset(Dataset):
    """Wraps huggingface.co/datasets/ZenoNing/GarmentCodeVTONDataset via the `datasets`
    library. Returns RAW samples (PIL images + text), not preprocessed tensors: turning
    a (avatar, garment, fitted-result) triplet into IDM-VTON's actual training inputs
    requires running IDM-VTON's own preprocessing (human parsing, OpenPose, DensePose,
    prompt encoding) on the avatar image, which needs the loaded IDMVTONPipeline -- that
    happens once per training step in train.py's `prepare_batch()`, not here, so this
    class stays independent of any loaded model.
    """

    def __init__(
        self,
        split: str = "train",
        column_map: Optional[dict] = None,
        data_dir: Optional[str] = None,
        dataset_name: str = "ZenoNing/GarmentCodeVTONDataset",
    ):
        """
        data_dir: a local directory previously written by
            scripts/download_dataset.py (via `Dataset.save_to_disk`) -- loaded with
            `load_from_disk` instead of re-downloading from the Hub. Falls back to
            `load_dataset(dataset_name, split=split)` (which uses/populates the normal
            Hugging Face cache) if data_dir is None or doesn't exist yet.
        """
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError as exc:
            raise ImportError(
                "pip install datasets (Hugging Face's `datasets` library, listed in "
                "fitcontroler/requirements.txt) to use GarmentCodeVTONDataset"
            ) from exc

        if data_dir is not None and Path(data_dir).exists():
            self.ds = load_from_disk(str(data_dir))
        else:
            self.ds = load_dataset(dataset_name, split=split)
        available = list(self.ds.column_names)

        resolved = dict(column_map or {})
        for field, candidates in FIELD_CANDIDATES.items():
            resolved.setdefault(field, _resolve_column(available, candidates, field))
        self.column_map = resolved

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        cm = self.column_map

        def as_pil(value) -> Image.Image:
            return (value if isinstance(value, Image.Image) else Image.open(value)).convert("RGB")

        return {
            "avatar_image": as_pil(row[cm["avatar_image"]]),
            "garment_image": as_pil(row[cm["garment_image"]]),
            "result_image": as_pil(row[cm["result_image"]]),
            "fit_text": str(row[cm["fit_text"]]),
        }


def raw_collate(batch: list) -> list:
    """DataLoader collate_fn for GarmentCodeVTONDataset -- keeps samples as a plain
    list of dicts (PIL images can't be torch-stacked directly), letting
    train.py's prepare_batch() do the real preprocessing + batching."""
    return batch


def inspect_dataset(dataset_name: str = "ZenoNing/GarmentCodeVTONDataset", split: str = "train") -> None:
    """Prints the real column names + one sample's field types. Run this FIRST, on a
    machine with real Hugging Face access, to confirm or correct FIELD_CANDIDATES and
    FIT_KEYWORDS above against the dataset's actual schema and text vocabulary."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    print(f"{dataset_name} [{split}]: {len(ds)} rows")
    print("columns:", ds.column_names)
    sample = ds[0]
    for key, value in sample.items():
        extra = f" size={value.size}" if hasattr(value, "size") and isinstance(value, Image.Image) else f" value={value!r:.200}"
        print(f"  {key}: {type(value).__name__}{extra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inspect", action="store_true", help="print real column names + a sample row, then exit")
    parser.add_argument("--dataset", default="ZenoNing/GarmentCodeVTONDataset")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    if args.inspect:
        inspect_dataset(args.dataset, args.split)
    else:
        parser.print_help()
