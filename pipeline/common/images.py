"""Find and load camera images, whatever the folder layout.

A frame is identified everywhere by (camera_id, filename). The camera comes from a parent folder named
after a planogram camera; an image outside such a folder (flat or renamed layout) is matched to the
camera whose train frames it looks most like.
"""
import argparse
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pipeline.common.config import IMAGES_DIR, THUMB_MIN_CORR, THUMB_SIZE
from pipeline.common.planogram import load_planogram

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _image_paths(images_dir: Path) -> list[Path]:
    """Every image file under images_dir, skipping macOS zip debris (__MACOSX folders, ._ files)."""
    return sorted(
        p for p in images_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
        and p.is_file()
        and not p.name.startswith("._")
        and "__MACOSX" not in p.parts
    )


def _camera_from_folder(path: Path, images_dir: Path, cameras: set[str]) -> str | None:
    """Nearest parent folder (below images_dir) named after a known camera, else None."""
    for part in reversed(path.relative_to(images_dir).parts[:-1]):
        if part in cameras:
            return part
    return None


def _thumbnail(img: np.ndarray) -> np.ndarray:
    """Small edge-strength view of a BGR image, zero mean and unit norm so that a dot product is a correlation.

    Edges (shelf rails, fridge frames) stay put when the lighting, the stock or a night blind changes;
    plain grayscale mixed up the frames where a blind covers the shelves.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    thumb = cv2.resize(edges, THUMB_SIZE, interpolation=cv2.INTER_AREA).ravel()
    thumb -= thumb.mean()
    return thumb / (np.linalg.norm(thumb) + 1e-6)


@lru_cache(maxsize=None)
def camera_thumbnails(reference_dir: Path = IMAGES_DIR) -> dict[str, np.ndarray]:
    """One mean thumbnail per camera, from the frames of reference_dir that sit in a camera folder."""
    cameras = set(load_planogram())
    thumbs: dict[str, list[np.ndarray]] = {}
    for path in _image_paths(reference_dir):
        camera_id = _camera_from_folder(path, reference_dir, cameras)
        img = cv2.imread(str(path)) if camera_id else None
        if img is not None:
            thumbs.setdefault(camera_id, []).append(_thumbnail(img))
    means = {camera_id: np.mean(ts, axis=0) for camera_id, ts in thumbs.items()}
    return {camera_id: mean / (np.linalg.norm(mean) + 1e-6) for camera_id, mean in means.items()}


def nearest_camera(img: np.ndarray, references: dict[str, np.ndarray]) -> tuple[str, float]:
    """Camera whose reference thumbnail correlates best with the BGR image, and that correlation."""
    thumb = _thumbnail(img)
    camera_id = max(references, key=lambda c: float(references[c] @ thumb))
    return camera_id, float(references[camera_id] @ thumb)


@lru_cache(maxsize=None)
def find_images(images_dir: Path = IMAGES_DIR) -> dict[tuple[str, str], Path]:
    """Return {(camera_id, filename): path} for every image under images_dir (searched recursively)."""
    images_dir = Path(images_dir)
    cameras = set(load_planogram())
    found: dict[tuple[str, str], Path] = {}
    for path in _image_paths(images_dir):
        camera_id = _camera_from_folder(path, images_dir, cameras)
        if camera_id is None:
            img = cv2.imread(str(path))
            if img is None:
                print(f"WARNING unreadable image, skipped: {path}")
                continue
            references = camera_thumbnails()
            if not references:
                raise FileNotFoundError(
                    f"{path} is not inside a camera folder and there are no train frames in {IMAGES_DIR} to match it against"
                )
            camera_id, corr = nearest_camera(img, references)
            flag = "  WARNING weak match" if corr < THUMB_MIN_CORR else ""
            print(f"camera by thumbnail: {path.name} -> {camera_id} (corr {corr:.2f}){flag}")
        key = (camera_id, path.name)
        if key in found:
            print(f"WARNING duplicate frame {key}, keeping {found[key]}, ignoring {path}")
            continue
        found[key] = path
    return found


def list_images(images_dir: Path = IMAGES_DIR) -> list[tuple[str, str]]:
    """Sorted [(camera_id, filename), ...]. Train filenames are the unix capture timestamp, e.g. 1751052241.jpg."""
    return sorted(find_images(Path(images_dir)))


def load_image(camera_id: str, filename: str, images_dir: Path = IMAGES_DIR) -> np.ndarray:
    """BGR image (cv2.imread) of one frame listed by list_images. Raises FileNotFoundError if it cannot be read."""
    images_dir = Path(images_dir)
    path = images_dir / camera_id / filename
    if not path.is_file():
        path = find_images(images_dir).get((camera_id, filename), path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="List images per camera.")
    ap.add_argument("--images", type=Path, default=IMAGES_DIR)
    args = ap.parse_args()

    counts = Counter(camera_id for camera_id, _ in list_images(args.images))
    for camera_id, n in sorted(counts.items()):
        print(f"{camera_id}  {n:3} images")
    print(f"{len(counts)} cameras, {sum(counts.values())} images")
