"""Owner B. Cheap pixel features per crop. Reads crops_index.json, writes features.json."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.common.config import CROPS_INDEX, FEATURES, OUT

_RUN_FRAMES = OUT / "run_frames.json"
_HSV_BINS = 32


def _load_image(path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _hsv_hist(img_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = cv2.calcHist([hsv], [0], None, [_HSV_BINS], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [_HSV_BINS], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [_HSV_BINS], [0, 256]).flatten()
    hist = np.concatenate([h, s, v])
    total = hist.sum()
    return hist / total if total > 0 else hist


def _chi2_dist(a: np.ndarray, b: np.ndarray) -> float:
    denom = a + b
    mask = denom > 0
    return float(0.5 * np.sum((a[mask] - b[mask]) ** 2 / denom[mask]))


def area_features(
    crop: np.ndarray,
    visible_fraction: float = 1.0,
    reference: np.ndarray | None = None,
) -> dict:
    """Return feature dict for one 256x256 RGB warp crop.

    fill_ratio: fraction of pixels that differ from the border background estimate.
    hist_dist_ref: chi-squared HSV histogram distance to reference crop; 0 if no reference.
    edge_density: fraction of pixels that are Canny edges.
    visible_fraction: passed through from the crops index.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Background estimated from the 8-pixel border; shelves are low-S, mid-V.
    border_mask = np.zeros(hsv.shape[:2], dtype=bool)
    border_mask[:8, :] = True
    border_mask[-8:, :] = True
    border_mask[:, :8] = True
    border_mask[:, -8:] = True
    border_pixels = hsv[border_mask]
    bg_s = float(np.median(border_pixels[:, 1]))
    bg_v = float(np.median(border_pixels[:, 2]))

    s_diff = np.abs(hsv[:, :, 1].astype(float) - bg_s)
    v_diff = np.abs(hsv[:, :, 2].astype(float) - bg_v)
    fill_ratio = float(((s_diff > 30) | (v_diff > 40)).mean())

    hist_dist_ref = 0.0
    if reference is not None:
        hist_dist_ref = _chi2_dist(_hsv_hist(crop), _hsv_hist(reference))

    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float((edges > 0).mean())

    return {
        "fill_ratio": round(fill_ratio, 4),
        "hist_dist_ref": round(hist_dist_ref, 4),
        "edge_density": round(edge_density, 4),
        "visible_fraction": round(visible_fraction, 4),
    }


def run(index_path: Path, out_path: Path, run_frames: list[tuple[str, str]] | None = None) -> None:
    """Compute area_features for every crop in the index, write features.json.

    run_frames: list of (cam_id, filename) to process; None means process all.
    The earliest frame of each (cam, area) is used as the hist_dist_ref reference
    (proxy for a stocked/full state until labelled reference crops are available).
    """
    if not index_path.exists():
        print(f"crops_index not found: {index_path}")
        return

    index: dict = json.loads(index_path.read_text())
    frame_set: set[tuple[str, str]] | None = (
        {(cam, fn) for cam, fn in run_frames} if run_frames is not None else None
    )

    # Pre-build reference map: (cam, area_id) -> warp path of the earliest frame.
    reference_path: dict[tuple[str, str], str] = {}
    for cam_id, frames in index.items():
        for filename in sorted(frames):
            for area_id, meta in frames[filename].items():
                key = (cam_id, area_id)
                if key not in reference_path:
                    reference_path[key] = meta.get("warp", "")

    out: dict = {}
    for cam_id, frames in index.items():
        for filename, areas in frames.items():
            if frame_set is not None and (cam_id, filename) not in frame_set:
                continue
            for area_id, meta in areas.items():
                crop = _load_image(meta.get("warp", ""))
                if crop is None:
                    continue

                ref_warp = reference_path.get((cam_id, area_id), "")
                ref = _load_image(ref_warp) if ref_warp and ref_warp != meta.get("warp") else None

                feats = area_features(
                    crop,
                    visible_fraction=meta.get("visible_fraction", 1.0),
                    reference=ref,
                )
                out.setdefault(cam_id, {}).setdefault(filename, {})[area_id] = feats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    total = sum(len(a) for c in out.values() for a in c.values())
    print(f"Features: {total} areas -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    ap.add_argument("--out", type=Path, default=FEATURES)
    ap.add_argument("--all", dest="all_frames", action="store_true",
                    help="Process all frames in the index, not just run_frames.json")
    args = ap.parse_args()

    run_frames = None
    if not args.all_frames and _RUN_FRAMES.exists():
        run_frames = json.loads(_RUN_FRAMES.read_text())

    run(args.index, args.out, run_frames=run_frames)
