"""Owner A. Cut every planogram area of every image into two PNG crops. Writes out/crops/, crops_index.json and run_frames.json (see CONTRACTS.md).

Per area: a warped crop (the quad flattened to a square, BGRA, alpha 0 where the quad is off-image)
and a context crop (axis-aligned, padded, with the quad outlined). A train run (no --append) also
writes QA overlays and contact sheets to out/qa/.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.common.config import (
    CROPS_DIR, CROPS_INDEX, CTX_FILL, CTX_OUTLINE, CTX_PAD, FRAME_SIZE, IMAGES_DIR, PLANOGRAM, QA_DIR, ROOT, RUN_FRAMES,
    WARP_SIZE,
)
from pipeline.common.images import list_images, load_image
from pipeline.common.planogram import area_polygon, load_planogram
from pipeline.crop import qa


def warp_area(img: np.ndarray, polygon: list[tuple[int, int]], size: int = WARP_SIZE) -> tuple[np.ndarray, float]:
    """Flatten the quad (TL, TR, BR, BL) of a BGR image to a size x size BGRA crop.

    Alpha is 255 where the pixel comes from the image and 0 where the quad lies outside it (those
    pixels are black). Also returns the visible fraction: the share of the crop that is inside the image.
    """
    square = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    matrix = cv2.getPerspectiveTransform(np.float32(polygon), square)
    bgr = cv2.warpPerspective(img, matrix, (size, size), flags=cv2.INTER_LINEAR)
    inside = np.full(img.shape[:2], 255, np.uint8)
    alpha = cv2.warpPerspective(inside, matrix, (size, size), flags=cv2.INTER_NEAREST)
    bgr[alpha == 0] = 0  # interpolation leaves a blended row at the image edge
    return np.dstack([bgr, alpha]), float((alpha > 0).mean())


def context_crop(img: np.ndarray, polygon: list[tuple[int, int]], pad: float = CTX_PAD) -> np.ndarray:
    """Axis-aligned BGR crop around the quad, padded by `pad` of its bounding box per side, quad outlined.

    Keeps the camera resolution. Pixels outside the image are filled with CTX_FILL.
    """
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    pad_x, pad_y = round((max(xs) - min(xs)) * pad), round((max(ys) - min(ys)) * pad)
    x0, y0, x1, y1 = min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y

    ctx = np.full((y1 - y0, x1 - x0, 3), CTX_FILL, np.uint8)
    height, width = img.shape[:2]
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if sx0 < sx1 and sy0 < sy1:
        ctx[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    outline = np.int32([[x - x0, y - y0] for x, y in polygon])
    cv2.polylines(ctx, [outline], isClosed=True, color=CTX_OUTLINE, thickness=1, lineType=cv2.LINE_AA)
    return ctx


def crop_frame(img: np.ndarray, camera_id: str, filename: str, areas: list[dict], out_dir: Path) -> dict:
    """Write both crops of every area of one frame; return its crops_index entries {area_id: {...}}."""
    stem = Path(filename).stem
    (out_dir / camera_id).mkdir(parents=True, exist_ok=True)
    entries = {}
    for area in areas:
        polygon = area_polygon(area)
        warp, visible_fraction = warp_area(img, polygon)
        paths = {kind: out_dir / camera_id / f"{stem}_{area['area_id']}_{kind}.png" for kind in ("warp", "ctx")}
        cv2.imwrite(str(paths["warp"]), warp)
        cv2.imwrite(str(paths["ctx"]), context_crop(img, polygon))
        entries[area["area_id"]] = {
            "product_id": area["product_id"],
            "product_name": area["product_name"],
            "quad": [list(p) for p in polygon],
            "visible_fraction": round(visible_fraction, 4),
            "warp": _index_path(paths["warp"]),
            "ctx": _index_path(paths["ctx"]),
        }
    return entries


def _index_path(path: Path) -> str:
    """Path as stored in the index: relative to the repo root when inside it (out/crops/...)."""
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def run(
    images_dir: Path, planogram_path: Path, out_dir: Path, index_path: Path, run_frames_path: Path,
    append: bool = False, qa_dir: Path | None = None,
) -> None:
    """Crop every area of every frame under images_dir.

    append=True merges the new frames into the existing index (train crops stay available during a
    validation run); otherwise the index is rebuilt. run_frames.json always lists exactly the frames
    found in images_dir, including any that failed to crop, so `decide` still outputs them.
    """
    planogram = load_planogram(planogram_path)
    frames = list_images(images_dir)
    index = json.loads(index_path.read_text()) if append and index_path.exists() else {}

    done: dict[str, list[str]] = {}
    failed = 0
    for camera_id, filename in frames:
        try:
            img = load_image(camera_id, filename, images_dir)
            if (img.shape[1], img.shape[0]) != FRAME_SIZE:  # planogram coordinates assume FRAME_SIZE
                print(f"WARNING {camera_id}/{filename} is {img.shape[1]}x{img.shape[0]}, resized to {FRAME_SIZE}")
                img = cv2.resize(img, FRAME_SIZE, interpolation=cv2.INTER_AREA)
            entries = crop_frame(img, camera_id, filename, planogram.get(camera_id, []), out_dir)
        except Exception as error:  # one bad frame must not stop the run
            failed += 1
            print(f"WARNING crop failed for {camera_id}/{filename}: {error!r}")
            continue
        index.setdefault(camera_id, {})[filename] = entries
        done.setdefault(camera_id, []).append(filename)
        if qa_dir is not None:
            qa.write_overlay(img, planogram.get(camera_id, []), qa_dir / "overlay" / camera_id / f"{Path(filename).stem}.jpg")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index))
    run_frames_path.parent.mkdir(parents=True, exist_ok=True)
    run_frames_path.write_text(json.dumps([list(frame) for frame in frames]))
    if qa_dir is not None:
        for camera_id, filenames in done.items():
            qa.write_contact_sheet(index[camera_id], filenames, qa_dir / "sheets" / f"{camera_id}.jpg")

    areas = sum(len(index[c][f]) for c, fs in done.items() for f in fs)
    print(f"cropped {areas} areas from {len(frames) - failed}/{len(frames)} frames -> {out_dir}")
    print(f"index: {sum(len(fs) for fs in index.values())} frames in {index_path}; run frames: {run_frames_path}")
    if qa_dir is not None:
        print(f"QA overlays and contact sheets: {qa_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=IMAGES_DIR)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--out-dir", type=Path, default=CROPS_DIR)
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    ap.add_argument("--run-frames", type=Path, default=RUN_FRAMES)
    ap.add_argument("--qa-dir", type=Path, default=QA_DIR)
    ap.add_argument("--append", action="store_true", help="validation run: merge into the existing index, no QA images")
    args = ap.parse_args()
    # No QA images on an append run: validation frames must never be looked at (CLAUDE.md hard rule 1).
    run(
        args.images, args.planogram, args.out_dir, args.index, args.run_frames,
        append=args.append, qa_dir=None if args.append else args.qa_dir,
    )
