"""Read-only inspection endpoints over the artefacts in out/ and labels/.

Thin wrapper: every route is a lookup in cached JSON plus a call to an existing pipeline helper.
"""
from fastapi import APIRouter, HTTPException

from pipeline.common.config import FEATURES, LABELS, PLANOGRAM, RESULTS, SUBMISSION
from pipeline.common.planogram import load_planogram
from pipeline.eval.score import score as score_fn

from api.cache import load_json_cached

router = APIRouter(prefix="/api")


@router.get("/cameras")
def cameras() -> list[dict]:
    """{camera_id, products: [{area_id, product_id, product_name}]} for every camera in the planogram."""
    planogram = load_planogram(PLANOGRAM)
    return [
        {
            "camera_id": cam,
            "products": [
                {"area_id": a["area_id"], "product_id": a["product_id"], "product_name": a["product_name"]}
                for a in areas
            ],
        }
        for cam, areas in sorted(planogram.items())
    ]


@router.get("/score")
def score() -> dict:
    """Score out/submission.json against labels/dev.json. 404 if either file is missing or empty."""
    pred = load_json_cached(SUBMISSION)
    gt = load_json_cached(LABELS)
    if not pred:
        raise HTTPException(status_code=404, detail=f"submission not found at {SUBMISSION}")
    if not gt:
        raise HTTPException(status_code=404, detail=f"labels not found at {LABELS}")
    return score_fn(pred, gt)


@router.get("/stats")
def stats() -> dict:
    """Aggregate counts over out/results.json: stock levels, compliance, low confidence, fallbacks."""
    results = load_json_cached(RESULTS)
    if not results:
        raise HTTPException(status_code=404, detail=f"results not found at {RESULTS}")
    stock = {"empty": 0, "half": 0, "full": 0}
    compliance = {"true": 0, "false": 0}
    n_frames = 0
    n_areas = 0
    n_low = 0
    n_fallback = 0
    for frames in results.values():
        for areas in frames.values():
            n_frames += 1
            for row in areas.values():
                n_areas += 1
                level = row.get("pred", {}).get("stock_level")
                if level in stock:
                    stock[level] += 1
                compliant = row.get("pred", {}).get("planogram_compliant")
                compliance["true" if compliant else "false"] += 1
                if row.get("low_confidence"):
                    n_low += 1
                reason = row.get("reason", "")
                if "fallback" in reason or "no sources" in reason:
                    n_fallback += 1
    return {
        "n_frames": n_frames,
        "n_areas": n_areas,
        "stock": stock,
        "compliance": compliance,
        "n_low_confidence": n_low,
        "n_fallback": n_fallback,
    }


@router.get("/area/{cam}/{filename}/{area_id}")
def area(cam: str, filename: str, area_id: str) -> dict:
    """Everything we know about one area: planogram meta, decide row, features, label (if any)."""
    planogram = load_planogram(PLANOGRAM)
    product = next((a for a in planogram.get(cam, []) if a["area_id"] == area_id), None)
    if product is None:
        raise HTTPException(status_code=404, detail=f"area {cam}/{area_id} not in planogram")
    results = load_json_cached(RESULTS) or {}
    features = load_json_cached(FEATURES) or {}
    labels = load_json_cached(LABELS) or {}
    return {
        "camera_id": cam,
        "filename": filename,
        "area_id": area_id,
        "product_id": product["product_id"],
        "product_name": product["product_name"],
        "result": results.get(cam, {}).get(filename, {}).get(area_id),
        "features": features.get(cam, {}).get(filename, {}).get(area_id),
        "label": labels.get(cam, {}).get(filename, {}).get(area_id),
    }
