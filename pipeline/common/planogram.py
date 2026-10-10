"""Load planogram.json: camera_id -> list of areas (area_id, product_id, product_name, image_coordinates)."""
import argparse
from pathlib import Path

from pipeline.common.config import PLANOGRAM


def load_planogram(path: Path = PLANOGRAM) -> dict[str, list[dict]]:
    """TODO: return {camera_id: [area, ...]} from data["Frutas e Legumes"]["cameras"][cam]["products"]."""
    raise NotImplementedError


def area_polygon(area: dict) -> list[tuple[int, int]]:
    """TODO: corners in order TL, TR, BR, BL. Coordinates may be negative (area partly off-image)."""
    raise NotImplementedError


def is_packaged(product_name: str) -> bool:
    """TODO: names ending in KG are loose (unit = one fruit); names with a weight (500 G, EMB. 3KG) are packaged."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect the planogram.")
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    args = ap.parse_args()
    # TODO: print area count per camera (expect 14 cameras, 152 areas).
    raise NotImplementedError
