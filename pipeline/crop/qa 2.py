"""Owner A. QA images for eyeballing the crop stage: quad overlays per frame and a contact sheet per camera."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from pipeline.common.config import ROOT
from pipeline.common.planogram import area_polygon

FONT = cv2.FONT_HERSHEY_SIMPLEX
STORE_TZ = ZoneInfo("Europe/Lisbon")
CELL = 96  # contact sheet thumbnail side, px
LABEL_WIDTH = 230  # contact sheet left column, px
HEADER_HEIGHT = 24  # contact sheet time row, px
# BGR, cycled per area so that neighbouring quads differ.
PALETTE = [(60, 60, 255), (60, 200, 60), (255, 140, 40), (40, 200, 255), (255, 60, 220), (220, 220, 40)]


def area_number(area_id: str) -> int:
    """6215_2 -> 2. Used for labels and for sorting areas in natural order."""
    return int(area_id.rsplit("_", 1)[1])


def frame_time(filename: str) -> str:
    """Store-local HH:MM for a unix-timestamp filename, else the start of the name."""
    stem = Path(filename).stem
    return datetime.fromtimestamp(int(stem), STORE_TZ).strftime("%H:%M") if stem.isdigit() else stem[:8]


def _text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, colour: tuple[int, int, int]) -> None:
    """Coloured text with a black halo, readable on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, colour, 1, cv2.LINE_AA)


def write_overlay(img: np.ndarray, areas: list[dict], path: Path) -> None:
    """Save the full BGR frame with every quad drawn and labelled (area number + start of the product name)."""
    out = img.copy()
    height, width = out.shape[:2]
    for i, area in enumerate(areas):
        polygon = np.int32(area_polygon(area))
        colour = PALETTE[i % len(PALETTE)]
        cv2.polylines(out, [polygon], isClosed=True, color=colour, thickness=2, lineType=cv2.LINE_AA)
        # Label at the centre of the visible part of the quad.
        cx, cy = np.clip(polygon, [0, 0], [width - 1, height - 1]).mean(axis=0).astype(int)
        name = " ".join(area["product_name"].split()[:2])
        _text(out, str(area_number(area["area_id"])), (cx - 8, cy), 0.6, colour)
        _text(out, name, (cx - 40, cy + 14), 0.35, (255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_contact_sheet(camera_index: dict, filenames: list[str], path: Path) -> None:
    """Save one grid per camera: a row per area, a column per frame in time order, cells are the warped crops.

    camera_index is crops_index[camera_id]: {filename: {area_id: entry}}.
    """
    filenames = sorted(filenames)
    area_ids = sorted({a for f in filenames for a in camera_index[f]}, key=area_number)
    sheet = np.full((HEADER_HEIGHT + CELL * len(area_ids), LABEL_WIDTH + CELL * len(filenames), 3), 30, np.uint8)

    for col, filename in enumerate(filenames):
        _text(sheet, frame_time(filename), (LABEL_WIDTH + col * CELL + 26, 17), 0.45, (255, 255, 255))
    for row, area_id in enumerate(area_ids):
        y = HEADER_HEIGHT + row * CELL
        name = next(camera_index[f][area_id]["product_name"] for f in filenames if area_id in camera_index[f])
        _text(sheet, area_id, (6, y + 40), 0.5, (255, 255, 255))
        _text(sheet, name[:30], (6, y + 60), 0.35, (200, 200, 200))
        for col, filename in enumerate(filenames):
            entry = camera_index[filename].get(area_id)
            warp = cv2.imread(str(ROOT / entry["warp"])) if entry else None  # 3 channels: off-image pixels are black
            if warp is not None:
                x = LABEL_WIDTH + col * CELL
                sheet[y:y + CELL, x:x + CELL] = cv2.resize(warp, (CELL, CELL), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
