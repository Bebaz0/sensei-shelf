"""Owner D (B wrote this minimal version to unblock the calibration labelling, see docs/DECISIONS.md).

Interactive labelling tool. Opens a cv2 window per area with the full frame (quad highlighted),
the context crop, and the earliest-frame crop of the same area for reference. Keys:
  1/2/3  stock empty/half/full (writes the label and advances)
  X      wrong product -> picker over ALL products: type to filter (English or Portuguese name),
         digit = pick, , and . = page, Backspace = erase, ESC = cancel; then 1/2/3 for stock
  U      unsure, skip without writing
  n      next area (no write)
  p      previous area (no write)
  q      save and quit
Labels are written incrementally to labels/<name>.json so a crash does not lose work.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.common.config import (
    CALIBRATION, CROPS_INDEX, IMAGES_DIR, PLANOGRAM, PRODUCTS_EN, ROOT, RUN_FRAMES, STOCK_VALUES,
)
from pipeline.common.planogram import area_polygon, load_planogram
from pipeline.crop.qa import frame_time

STOCK_KEY = {ord("1"): "empty", ord("2"): "half", ord("3"): "full"}
WINDOW = "label_tool"


def load_worklist(
    from_file: Path | None, run_frames_path: Path, planogram: dict, frames: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """Return list of (cam, filename, area_id) to iterate.

    from_file set: parse `<cam> <file> <area_id>` lines (comments after # ignored).
    from_file None: full mode, every area of every frame in `frames` (all train frames of the crops
    index when given), else of every frame in run_frames.json.
    """
    if from_file:
        items = []
        for line in from_file.read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            items.append((parts[0], parts[1], parts[2]))
        return items
    if frames is None:
        frames = json.loads(run_frames_path.read_text())
    out = []
    for cam, filename in frames:
        for area in planogram.get(cam, []):
            out.append((cam, filename, area["area_id"]))
    return out


def already_labelled(labels: dict, cam: str, filename: str, area_id: str) -> bool:
    return area_id in labels.get(cam, {}).get(filename, {})


def write_label(path: Path, labels: dict, cam: str, filename: str, area_id: str, entry: dict) -> None:
    labels.setdefault(cam, {}).setdefault(filename, {})[area_id] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2))


def _unique_products(areas: list[dict]) -> list[dict]:
    seen, out = set(), []
    for a in areas:
        if a["product_id"] not in seen:
            seen.add(a["product_id"])
            out.append({"product_id": a["product_id"], "product_name": a["product_name"]})
    return out


def all_products(planogram: dict, cam: str, names_en: dict) -> list[dict]:
    """Every product of every camera, this camera's first. Each has an English name for display and search."""
    here = _unique_products(planogram.get(cam, []))
    seen = {p["product_id"] for p in here}
    rest = [p for c, areas in sorted(planogram.items()) if c != cam for p in _unique_products(areas)
            if p["product_id"] not in seen and not seen.add(p["product_id"])]
    products = here + sorted(rest, key=lambda p: p["product_name"])
    return [{**p, "name_en": names_en.get(p["product_id"], {}).get("name_en", "")} for p in products]


def filter_products(products: list[dict], query: str) -> list[dict]:
    """Case-insensitive match of every word of the query against the Portuguese and English name."""
    words = query.lower().split()
    return [p for p in products if all(w in f"{p['product_name']} {p['name_en']}".lower() for w in words)]


def _load_frame(cam: str, filename: str) -> np.ndarray | None:
    img = cv2.imread(str(IMAGES_DIR / cam / filename))
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != (1280, 720):
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    return img


def _draw_frame_with_quads(frame: np.ndarray, areas: list[dict], current_area_id: str) -> np.ndarray:
    out = frame.copy()
    for area in areas:
        polygon = np.int32(area_polygon(area))
        is_cur = area["area_id"] == current_area_id
        colour = (0, 255, 255) if is_cur else (120, 120, 120)
        thickness = 3 if is_cur else 1
        cv2.polylines(out, [polygon], isClosed=True, color=colour, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def _load_ctx(index: dict, cam: str, filename: str, area_id: str) -> np.ndarray | None:
    meta = index.get(cam, {}).get(filename, {}).get(area_id)
    return cv2.imread(str(ROOT / meta["ctx"])) if meta else None


def _earliest_ctx(index: dict, cam: str, area_id: str, exclude: str) -> np.ndarray | None:
    for f in sorted(index.get(cam, {})):
        if f == exclude:
            continue
        meta = index[cam][f].get(area_id)
        if meta:
            img = cv2.imread(str(ROOT / meta["ctx"]))
            if img is not None:
                return img
    return None


def _fit(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5, colour=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def build_ui(
    frame: np.ndarray,
    cur_ctx: np.ndarray | None,
    earliest_ctx: np.ndarray | None,
    existing_label: dict | None,
    header: str,
    status: str,
    flagged_wrong: bool,
) -> np.ndarray:
    left = _fit(frame, 900, 506)
    panel_w = 360
    h = left.shape[0]
    right = np.full((h, panel_w, 3), 30, np.uint8)
    _text(right, "current", (10, 20), 0.5)
    if cur_ctx is not None:
        cur = _fit(cur_ctx, panel_w - 20, (h // 2) - 40)
        rh, rw = cur.shape[:2]
        right[30:30 + rh, 10:10 + rw] = cur
    mid = h // 2
    _text(right, "earliest frame", (10, mid + 20), 0.5)
    if earliest_ctx is not None:
        ear = _fit(earliest_ctx, panel_w - 20, (h // 2) - 40)
        rh, rw = ear.shape[:2]
        right[mid + 30:mid + 30 + rh, 10:10 + rw] = ear

    main = np.hstack([left, right])
    header_strip = np.full((44, main.shape[1], 3), 30, np.uint8)
    status_strip = np.full((28, main.shape[1], 3), 30, np.uint8)
    _text(header_strip, header, (10, 20), 0.55)
    if existing_label:
        tag = f"already labelled: {existing_label.get('stock_level', '?')}"
        if existing_label.get("planogram_compliant") is False:
            tag += f" + WRONG PRODUCT {existing_label.get('observed_product_id', '?')}"
        _text(header_strip, tag, (10, 38), 0.45, (120, 220, 120))
    _text(status_strip, status, (10, 20), 0.44)
    screen = np.vstack([header_strip, main, status_strip])
    if flagged_wrong:
        cv2.rectangle(screen, (0, 0), (screen.shape[1] - 1, screen.shape[0] - 1), (0, 0, 255), 4)
    return screen


PAGE = 9


def product_picker(products: list[dict], screen: np.ndarray) -> str | None:
    """Overlay a searchable, paged list of all products; return the picked product_id, or None if cancelled."""
    query, page = "", 0
    while True:
        shown = filter_products(products, query)
        pages = max(1, -(-len(shown) // PAGE))
        page = min(page, pages - 1)
        rows = shown[page * PAGE:(page + 1) * PAGE]
        overlay = screen.copy()
        cv2.rectangle(overlay, (40, 60), (screen.shape[1] - 40, screen.shape[0] - 60), (0, 0, 0), -1)
        _text(overlay, "wrong product - type to search, digit = pick, , . = page, ESC = cancel", (60, 90), 0.55)
        _text(overlay, f"search: {query}_   ({len(shown)} products, page {page + 1}/{pages})", (60, 122), 0.6, (120, 220, 255))
        for i, p in enumerate(rows):
            en = f"  [{p['name_en']}]" if p["name_en"] else ""
            _text(overlay, f"{i + 1}) {p['product_name']}{en}"[:95], (60, 160 + i * 32), 0.55)
        cv2.imshow(WINDOW, overlay)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            return None
        if ord("1") <= k <= ord("9"):
            if k - ord("1") < len(rows):
                return rows[k - ord("1")]["product_id"]
        elif k == ord("."):
            page = min(page + 1, pages - 1)
        elif k == ord(","):
            page = max(page - 1, 0)
        elif k in (8, 127):
            query, page = query[:-1], 0
        elif 32 <= k < 127:
            query, page = query + chr(k), 0


def run(name: str, from_file: Path | None, out_path: Path | None) -> None:
    planogram = load_planogram(PLANOGRAM)
    index = json.loads(CROPS_INDEX.read_text()) if CROPS_INDEX.exists() else {}
    # Full mode covers every train frame of the crops index, not whatever the last run left in run_frames.json.
    frames = [(cam, f) for cam in sorted(index) for f in sorted(index[cam])] if index else None
    worklist = load_worklist(from_file, RUN_FRAMES, planogram, frames)
    names_en = json.loads(PRODUCTS_EN.read_text()) if PRODUCTS_EN.exists() else {}
    out_path = out_path or (ROOT / "labels" / f"{name}.json")
    labels = json.loads(out_path.read_text()) if out_path.exists() else {}
    done = sum(1 for (c, f, a) in worklist if already_labelled(labels, c, f, a))
    print(f"{len(worklist)} areas on worklist; {done} already labelled")
    if not worklist:
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    i, flagged_wrong, observed_pid = 0, False, None
    while 0 <= i < len(worklist):
        cam, filename, area_id = worklist[i]
        areas = planogram.get(cam, [])
        area = next((a for a in areas if a["area_id"] == area_id), None)
        if area is None:
            print(f"skip: {area_id} not in planogram for {cam}")
            i += 1
            continue
        frame = _load_frame(cam, filename)
        if frame is None:
            print(f"skip: cannot load {cam}/{filename}")
            i += 1
            continue
        done = sum(1 for (c, f, a) in worklist if already_labelled(labels, c, f, a))
        header = (
            f"{i + 1}/{len(worklist)}  done:{done}  {cam}/{filename} {frame_time(filename)}"
            f"  {area_id}  expected: {area['product_name']}"
        )
        existing = labels.get(cam, {}).get(filename, {}).get(area_id)
        status = "1/2/3=stock  X=wrong product  U=unsure  n/p=next/prev  q=quit"
        if flagged_wrong:
            name = next((p["product_name"] for p in all_products(planogram, cam, names_en) if p["product_id"] == observed_pid), observed_pid)
            status = f"WRONG PRODUCT = {name}: press 1/2/3 to SAVE with stock (X=cancel)"
        screen = build_ui(
            _draw_frame_with_quads(frame, areas, area_id),
            _load_ctx(index, cam, filename, area_id),
            _earliest_ctx(index, cam, area_id, exclude=filename),
            existing,
            header,
            status,
            flagged_wrong,
        )
        cv2.imshow(WINDOW, screen)
        k = cv2.waitKey(0) & 0xFF

        if k == ord("q"):
            break
        if flagged_wrong and k not in STOCK_KEY and k != ord("x"):
            # The wrong product is only saved together with a stock level: do not drop it silently.
            print("wrong product picked but not saved yet: press 1/2/3 for the stock level (X cancels)")
            continue
        if k == ord("n"):
            i += 1
            continue
        if k == ord("p"):
            i = max(0, i - 1)
            flagged_wrong, observed_pid = False, None
            continue
        if k == ord("u"):
            i += 1
            flagged_wrong, observed_pid = False, None
            continue
        if k == ord("x"):
            if flagged_wrong:
                flagged_wrong, observed_pid = False, None
            else:
                picked = product_picker(all_products(planogram, cam, names_en), screen)
                if picked:
                    flagged_wrong, observed_pid = True, picked
            continue
        if k in STOCK_KEY:
            level = STOCK_KEY[k]
            entry = {
                "planogram_compliant": not flagged_wrong,
                "stock_level": level,
                "stock_value": STOCK_VALUES[level],
            }
            if flagged_wrong and observed_pid:
                entry["observed_product_id"] = observed_pid
            write_label(out_path, labels, cam, filename, area_id, entry)
            flagged_wrong, observed_pid = False, None
            i += 1
            continue

    cv2.destroyAllWindows()
    total = sum(len(a) for c in labels.values() for a in c.values())
    print(f"saved {total} labels to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="your name; writes to labels/<name>.json")
    ap.add_argument("--from", dest="from_file", type=Path, default=None,
                    help=f"iterate only areas in this file (e.g. {CALIBRATION.relative_to(ROOT)})")
    ap.add_argument("--out", type=Path, default=None, help="override output path")
    args = ap.parse_args()
    run(args.name, args.from_file, args.out)
