"""Owner C. Hand-label a dev set into labels/dev.json, same format as the hidden annotations.json."""
import argparse
from pathlib import Path

from pipeline.common.config import IMAGES_DIR, LABELS, PLANOGRAM


def label_image(camera_id: str, filename: str, areas: list[dict]) -> dict:
    """TODO: show image with area polygons, collect {area_id: {planogram_compliant, stock_level, stock_value}}."""
    raise NotImplementedError


def run(images_dir: Path, planogram: Path, out_path: Path, per_camera: int) -> None:
    """TODO: pick `per_camera` images per camera, label them, merge into out_path (don't overwrite existing labels)."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=IMAGES_DIR)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--out", type=Path, default=LABELS)
    ap.add_argument("--per-camera", type=int, default=2)
    args = ap.parse_args()
    run(args.images, args.planogram, args.out, args.per_camera)
