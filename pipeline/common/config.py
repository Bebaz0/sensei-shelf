"""Shared paths and constants. Every module imports from here; nobody hardcodes paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
PLANOGRAM = DATA / "planogram.json"
# <camera_id>/<unix_ts>.jpg — train and test are mixed, there are no split folders.
IMAGES_DIR = DATA / "Frutas e Legumes (train & test)"

OUT = ROOT / "out"
CROPS_DIR = OUT / "crops"
CROPS_INDEX = OUT / "crops_index.json"
FEATURES = OUT / "features.json"
VLM = OUT / "vlm.json"
RESULTS = OUT / "results.json"
SUBMISSION = OUT / "submission.json"

LABELS = ROOT / "labels" / "dev.json"

STOCK_VALUES = {"empty": 0.0, "half": 0.5, "full": 1.0}

# TODO: tuning knobs (thresholds, crop size) go here as modules need them.

if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Print resolved paths.").parse_args()
    for k, v in dict(globals()).items():
        if k.isupper():
            print(f"{k:12} {v}")
