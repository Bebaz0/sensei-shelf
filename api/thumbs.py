"""Pillow thumbnail helper with on-disk cache. Mirrors scripts/sync_dashboard.py:save_jpg."""
from pathlib import Path

from PIL import Image


def thumbnail_cached(src: Path, cache: Path, max_side: int | None, flatten_alpha: bool = False) -> Path:
    """Return a path to a JPG at most `max_side` px on the longer side.

    If max_side is None and the source needs no flattening, return `src` unchanged.
    Otherwise write to `cache` (reusing it when newer than src) and return `cache`.
    """
    needs_work = flatten_alpha or max_side is not None or src.suffix.lower() != ".jpg"
    if not needs_work:
        return src
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    if flatten_alpha and img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (114, 114, 114))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    if max_side is not None:
        img.thumbnail((max_side, max_side))
    img.save(cache, "JPEG", quality=80)
    return cache
