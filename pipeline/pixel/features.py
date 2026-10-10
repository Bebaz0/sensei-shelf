"""Owner B. Cheap pixel features per crop (fill ratio, colour stats, ...). Reads crops_index.json, writes features.json."""
import argparse
from pathlib import Path

import numpy as np

from pipeline.common.config import CROPS_INDEX, FEATURES


def area_features(crop: np.ndarray) -> dict:
    """TODO: return feature dict for one crop, e.g. fill_ratio, dominant hue, edge density."""
    raise NotImplementedError


def run(index_path: Path, out_path: Path) -> None:
    """TODO: compute area_features for every crop in the index, write features.json."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    ap.add_argument("--out", type=Path, default=FEATURES)
    args = ap.parse_args()
    run(args.index, args.out)
