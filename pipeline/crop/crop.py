"""Owner A. Warp every planogram area of every image into a flat crop. Writes out/crops/ + crops_index.json (see CONTRACTS.md)."""
import argparse
from pathlib import Path

import numpy as np

from pipeline.common.config import CROPS_DIR, CROPS_INDEX, IMAGES_DIR, PLANOGRAM


def crop_area(img: np.ndarray, polygon: list[tuple[int, int]]) -> np.ndarray:
    """TODO: perspective-warp the 4-point area to a rectangle. Handle corners outside the image."""
    raise NotImplementedError


def run(images_dir: Path, planogram: Path, out_dir: Path, index_path: Path) -> None:
    """TODO: for each (camera, image, area): crop, save PNG, append entry to crops_index.json."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=IMAGES_DIR)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--out-dir", type=Path, default=CROPS_DIR)
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    args = ap.parse_args()
    run(args.images, args.planogram, args.out_dir, args.index)
