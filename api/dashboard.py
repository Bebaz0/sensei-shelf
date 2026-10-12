"""Dashboard-facing endpoints under /data. Replace scripts/sync_dashboard.sh at dev time."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from pipeline.common.config import CROPS_DIR, OUT, RESULTS, TRAIN_IMAGE_DIRS

from api.cache import load_json_cached
from api.thumbs import thumbnail_cached

router = APIRouter(prefix="/data")

CACHE_DIR = OUT / "api_cache"


def _rewrite_crops(results: dict) -> dict:
    """Replace each `crop` path (out/crops/{cam}/{stem}.png) with /data/crops/{cam}/{stem}.jpg."""
    out: dict = {}
    for cam, frames in results.items():
        out[cam] = {}
        for fname, areas in frames.items():
            out[cam][fname] = {}
            for area_id, row in areas.items():
                new_row = dict(row)
                crop = row.get("crop")
                if crop:
                    stem = Path(crop).stem
                    new_row["crop"] = f"/data/crops/{cam}/{stem}.jpg"
                out[cam][fname][area_id] = new_row
    return out


@router.get("/results.json")
def results() -> dict:
    data = load_json_cached(RESULTS)
    if not data:
        raise HTTPException(status_code=404, detail=f"results not found at {RESULTS}")
    return _rewrite_crops(data)


@router.get("/frames/{cam}/{filename}")
def frame(cam: str, filename: str, max: int | None = Query(default=None, ge=16, le=4096)) -> FileResponse:
    src = next((d / cam / filename for d in TRAIN_IMAGE_DIRS if (d / cam / filename).is_file()), TRAIN_IMAGE_DIRS[0] / cam / filename)
    if not src.is_file():
        raise HTTPException(status_code=404, detail=f"frame not found: {cam}/{filename}")
    if max is None:
        return FileResponse(src, media_type="image/jpeg")
    cache = CACHE_DIR / "frames" / cam / f"{filename}_{max}.jpg"
    served = thumbnail_cached(src, cache, max)
    return FileResponse(served, media_type="image/jpeg")


@router.get("/crops/{cam}/{stem}.jpg")
def crop(cam: str, stem: str, max: int | None = Query(default=None, ge=16, le=2048)) -> FileResponse:
    src = next((p for p in [CROPS_DIR / cam / f"{stem}.png", CROPS_DIR / cam / f"{stem}.jpg"] if p.is_file()), None)
    if src is None:
        raise HTTPException(status_code=404, detail=f"crop not found: {cam}/{stem}")
    cache = CACHE_DIR / "crops" / cam / f"{stem}_{max or 'full'}.jpg"
    served = thumbnail_cached(src, cache, max, flatten_alpha=True)
    return FileResponse(served, media_type="image/jpeg")
