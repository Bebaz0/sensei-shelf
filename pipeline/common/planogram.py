"""Load planogram.json: camera_id -> list of areas (area_id, product_id, product_name, image_coordinates)."""
import argparse
import json
import re
from pathlib import Path

from pipeline.common.config import PLANOGRAM

SECTION = "Frutas e Legumes"
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")

# A pack weight is a number followed by a unit ("500 G", "250GR", "EMB. 3KG", "1.5 KG").
# The lookbehind rejects calibre ranges ("73/85 KG") and codes ("R0 KG"), which are loose items.
PACK_WEIGHT = re.compile(r"(?<![\w/.,])\d+(?:[.,]\d+)?\s*(?:KG|GR|G)\b")


def load_planogram(path: Path = PLANOGRAM) -> dict[str, list[dict]]:
    """Return {camera_id: [area, ...]}, areas exactly as in the file."""
    cameras = json.loads(Path(path).read_text(encoding="utf-8"))[SECTION]["cameras"]
    return {camera_id: camera["products"] for camera_id, camera in cameras.items()}


def area_polygon(area: dict) -> list[tuple[int, int]]:
    """Corners in order TL, TR, BR, BL. Coordinates may be negative (area partly off-image)."""
    coords = area["image_coordinates"]
    return [(int(coords[c][0]), int(coords[c][1])) for c in CORNERS]


def is_packaged(product_name: str) -> bool:
    """True when the unit is one package (name carries a pack weight), False for loose items sold by KG."""
    return PACK_WEIGHT.search(product_name.upper()) is not None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect the planogram.")
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--products", action="store_true", help="also list every product with its loose/package unit")
    args = ap.parse_args()

    planogram = load_planogram(args.planogram)
    for camera_id, areas in sorted(planogram.items()):
        off_image = sum(any(x < 0 or y < 0 for x, y in area_polygon(a)) for a in areas)
        print(f"{camera_id}  {len(areas):3} areas  {off_image} partly off-image")
    all_areas = [a for areas in planogram.values() for a in areas]
    products = {a["product_id"]: a["product_name"] for a in all_areas}
    print(f"{len(planogram)} cameras, {len(all_areas)} areas, {len(products)} products")
    if args.products:
        for product_id, name in sorted(products.items(), key=lambda kv: kv[1]):
            print(f"{product_id:>8}  {'package' if is_packaged(name) else 'loose  '}  {name}")
