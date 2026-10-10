"""Find and load camera images."""
import argparse
from pathlib import Path

import numpy as np

from pipeline.common.config import IMAGES_DIR


def list_images(images_dir: Path = IMAGES_DIR) -> list[tuple[str, str]]:
    """TODO: return sorted [(camera_id, filename), ...]. Filename is the unix capture timestamp, e.g. 1751052241.jpg."""
    raise NotImplementedError


def load_image(camera_id: str, filename: str, images_dir: Path = IMAGES_DIR) -> np.ndarray:
    """TODO: return BGR image (cv2.imread)."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="List images per camera.")
    ap.add_argument("--images", type=Path, default=IMAGES_DIR)
    args = ap.parse_args()
    # TODO: print image count per camera.
    raise NotImplementedError
