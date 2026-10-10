"""Owner C. Combine pixel features + VLM answers into per-area verdicts. Writes results.json (dashboard) and submission.json (scored).

Every area of every run frame is output: areas come from planogram.json, frames from run_frames.json
(else the --images folder), and any area whose inputs are missing or broken falls back to FALLBACK_P.
"""
import argparse
import json
import math
from pathlib import Path

from pipeline.common.config import (
    CROPS_INDEX, EMPTY_IS_COMPLIANT, FEATURES, IMAGES_DIR, PLANOGRAM, RESULTS, RUN_FRAMES, STOCK_BIAS,
    STOCK_VALUES, SUBMISSION, VLM,
)
from pipeline.common.images import list_images
from pipeline.common.planogram import load_planogram

LEVELS = ("empty", "half", "full")
PARAMS_PATH = Path(__file__).with_name("params.json")

# Stock prior when no source has an answer. (0.1, 0.4, 0.5) decides `half` under the EV rule.
# Replace with the label distribution once labels exist (params.json "fallback_p").
FALLBACK_P = {"empty": 0.1, "half": 0.4, "full": 0.5}

# Defaults until lofo.py --fit writes params.json. Weights are per stock source; a missing source is skipped.
DEFAULT_PARAMS = {
    "fallback_p": FALLBACK_P,
    "weights": {"gemini": 1.0, "claude": 1.0, "knn": 1.0, "probe": 1.0},
    # p_wrong = sigmoid(a + b * x), x in [-1, 1] = mean over VLMs of (+conf if mismatch else -conf).
    # Conservative: a mismatch needs confidence > 0.6 to be called non-compliant.
    "platt": {"a": -1.5, "b": 2.5},
    "knn_wrong_weight": 0.5,  # share of the same-area kNN p_wrong in the final p_wrong
    "fill_sigma": 10.0,  # VLM fill_pct uncertainty, in percentage points
    "stock_bias": STOCK_BIAS,
    # Confidence gate: an area below min_top_p, with p_wrong inside wrong_band or with no source is low_confidence
    # and goes to the VLM (decide --queue). Thresholds are hand-set until queued areas have VLM answers to tune on.
    "gate": {"min_top_p": 0.6, "wrong_band": [0.3, 0.7]},
}

# Mentor thresholds on covered share: empty 0-10%, half 10-50%, full above 50%.
EMPTY_MAX_PCT, HALF_MAX_PCT = 10.0, 50.0


def load_params(path: Path = PARAMS_PATH) -> dict:
    """DEFAULT_PARAMS overridden by the committed params.json, if present."""
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    if path.is_file():
        params.update(json.loads(path.read_text()))
    return params


def _normalise(p: dict) -> dict:
    total = sum(max(p.get(k, 0.0), 0.0) for k in LEVELS)
    if total <= 0:
        raise ValueError(f"no probability mass: {p}")
    return {k: max(p.get(k, 0.0), 0.0) / total for k in LEVELS}


def stock_level(p: dict, bias: float = STOCK_BIAS) -> str:
    """Expected-value argmax under the scoring (+1 correct, -2 lower than truth, -3 higher).
    bias > 0 subtracts bias * rank from each level's EV, leaning the decision lower."""
    p_e, p_h, p_f = (p[k] for k in LEVELS)
    ev = {
        "empty": p_e - 2 * (p_h + p_f),
        "half": p_h - 3 * p_e - 2 * p_f - bias,
        "full": p_f - 3 * (p_e + p_h) - 2 * bias,
    }
    return max(LEVELS, key=lambda k: (ev[k], -LEVELS.index(k)))  # ties go to the lower level


def is_compliant(p_wrong: float) -> bool:
    """Symmetric scoring (+1/-2 both ways): non-compliant only when p_wrong > 0.5."""
    return p_wrong <= 0.5


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def vlm_stock_probs(answer: dict, sigma: float) -> dict | None:
    """Soft stock distribution from one VLM answer: fill_pct through the mentor thresholds,
    mixed half and half with the stated level smoothed by confidence."""
    parts = []
    fill = answer.get("fill_pct")
    if isinstance(fill, (int, float)):
        p_e = _phi((EMPTY_MAX_PCT - fill) / sigma)
        p_f = 1 - _phi((HALF_MAX_PCT - fill) / sigma)
        parts.append({"empty": p_e, "half": max(1 - p_e - p_f, 0.0), "full": p_f})
    if answer.get("stock") in LEVELS:
        conf = min(max(float(answer.get("confidence", 0.5)), 0.0), 1.0)
        parts.append({k: conf * (k == answer["stock"]) + (1 - conf) / 3 for k in LEVELS})
    if not parts:
        return None
    return _normalise({k: sum(part[k] for part in parts) for k in LEVELS})


def stock_sources(features: dict | None, vlm: dict | None, sigma: float) -> dict[str, dict]:
    """{source_name: {empty, half, full}} for every source that answered for this area."""
    sources = {}
    for provider in ("gemini", "claude"):
        answer = (vlm or {}).get(provider)
        if answer and answer.get("observed") != "occluded":
            p = vlm_stock_probs(answer, sigma)
            if p:
                sources[provider] = p
    for name in ("knn", "probe"):
        p = (features or {}).get(name)
        if p and any(p.get(k) for k in LEVELS):
            sources[name] = _normalise(p)
    return sources


def stock_probs(sources: dict[str, dict], params: dict) -> dict:
    """v1: weighted average of the sources that answered, else the fallback prior."""
    weights = {name: params["weights"].get(name, 0.0) for name in sources}
    total = sum(weights.values())
    if total <= 0:
        return _normalise(params["fallback_p"])
    return _normalise({k: sum(weights[n] * sources[n][k] for n in sources) / total for k in LEVELS})


def vlm_mismatch(vlm: dict | None) -> float | None:
    """Platt input x in [-1, 1]: mean over VLMs of +conf (mismatch) or -conf (match). An empty or occluded
    area is never a mismatch (empty is compliant). None when no VLM answered."""
    xs = []
    for provider in ("gemini", "claude"):
        answer = (vlm or {}).get(provider)
        if not answer or answer.get("observed") == "occluded" or "matches_expected" not in answer:
            continue
        conf = min(max(float(answer.get("confidence", 0.5)), 0.0), 1.0)
        mismatch = not answer["matches_expected"] and answer.get("observed") != "empty"
        xs.append(conf if mismatch else -conf)
    return sum(xs) / len(xs) if xs else None


def knn_wrong(features: dict | None) -> float | None:
    knn = (features or {}).get("knn")
    return float(knn["p_wrong"]) if isinstance(knn, dict) and knn.get("p_wrong") is not None else None


def p_wrong(features: dict | None, vlm: dict | None, params: dict) -> float:
    """Calibrated P(wrong product): Platt scaling of the VLM matches_expected answers, blended with the
    same-area kNN p_wrong (compliance is stable per area)."""
    x, knn = vlm_mismatch(vlm), knn_wrong(features)
    if x is None:
        return 0.0 if knn is None else knn
    a, b = params["platt"]["a"], params["platt"]["b"]
    p_vlm = 1 / (1 + math.exp(-(a + b * x)))
    if knn is None:
        return p_vlm
    w = params["knn_wrong_weight"]
    return (1 - w) * p_vlm + w * knn


def _observed(vlm: dict | None) -> str | None:
    for provider in ("gemini", "claude"):
        answer = (vlm or {}).get(provider)
        if answer and answer.get("observed") not in (None, "empty", "occluded"):
            return answer["observed"]
    return None


def decide_area(area: dict, features: dict | None, vlm: dict | None, params: dict) -> dict:
    """One results.json row. Never raises: broken inputs fall back to the prior and compliant."""
    try:
        sources = stock_sources(features, vlm, params["fill_sigma"])
        p_stock = stock_probs(sources, params)
        level = stock_level(p_stock, params["stock_bias"])
        pw = p_wrong(features, vlm, params)
        compliant = is_compliant(pw) or (level == "empty" and EMPTY_IS_COMPLIANT)
        votes = ", ".join(f"{n}: {max(p, key=p.get)}" for n, p in sources.items()) or "no sources, prior"
        gate = params["gate"]
        low = not sources or max(p_stock.values()) < gate["min_top_p"] or gate["wrong_band"][0] < pw < gate["wrong_band"][1]
        observed = _observed(vlm)
        reason = f"{votes}; p_wrong {pw:.2f}"
    except Exception as e:  # noqa: BLE001  a crash here must never drop the area
        p_stock = _normalise(params["fallback_p"])
        level, pw, compliant, observed = stock_level(p_stock, params["stock_bias"]), 0.0, True, None
        reason, low = f"fallback ({type(e).__name__}: {e})", True
    return {
        "pred": {"planogram_compliant": compliant, "stock_level": level, "stock_value": STOCK_VALUES[level]},
        "p_stock": {k: round(v, 4) for k, v in p_stock.items()},
        "p_wrong": round(pw, 4),
        "observed_product_id": observed if not compliant else None,
        "low_confidence": low,
        "reason": reason,
    }


def run_frames(run_frames_path: Path, images_dir: Path) -> list[tuple[str, str]]:
    """Frames of the current run: run_frames.json, else every image of images_dir."""
    if run_frames_path.is_file():
        return [tuple(f) for f in json.loads(run_frames_path.read_text())]
    print(f"{run_frames_path} missing, listing {images_dir}")
    return list_images(images_dir)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"ignoring unreadable {path}: {e}")
        return {}


def decide_frames(
    frames: list[tuple[str, str]], planogram: dict, features: dict, vlm: dict, params: dict, crops: dict | None = None,
) -> dict:
    """results.json content for the given frames: every planogram area of each frame's camera."""
    results: dict = {}
    for camera_id, filename in frames:
        for area in planogram.get(camera_id, []):
            area_id = area["area_id"]
            row = decide_area(
                area,
                features.get(camera_id, {}).get(filename, {}).get(area_id),
                vlm.get(camera_id, {}).get(filename, {}).get(area_id),
                params,
            )
            row["crop"] = ((crops or {}).get(camera_id, {}).get(filename, {}).get(area_id) or {}).get("ctx")
            results.setdefault(camera_id, {}).setdefault(filename, {})[area_id] = row
    return results


def write_queue(results: dict, path: Path) -> int:
    """Low-confidence areas as `camera_id filename area_id` lines (format read by vlm.ask --areas). Returns the count."""
    lines = [f"{c} {f} {a}" for c, fs in results.items() for f, areas in fs.items()
             for a, row in areas.items() if row["low_confidence"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines))
    return len(lines)


def to_submission(results: dict) -> dict:
    """Official format: only planogram_compliant, stock_level, stock_value per area."""
    return {
        cam: {fn: {a: dict(row["pred"]) for a, row in areas.items()} for fn, areas in frames.items()}
        for cam, frames in results.items()
    }


def run(
    planogram_path: Path, features_path: Path, vlm_path: Path, results_path: Path, submission_path: Path,
    run_frames_path: Path = RUN_FRAMES, images_dir: Path = IMAGES_DIR, crops_index_path: Path = CROPS_INDEX,
    queue_path: Path | None = None,
) -> None:
    planogram = load_planogram(planogram_path)
    frames = run_frames(run_frames_path, images_dir)
    unknown = sorted({cam for cam, _ in frames} - set(planogram))
    if unknown:
        print(f"WARNING: cameras not in the planogram, no areas to output: {unknown}")

    results = decide_frames(
        frames, planogram, _load_json(features_path), _load_json(vlm_path), load_params(), _load_json(crops_index_path),
    )
    submission = to_submission(results)

    expected = sum(len(planogram.get(cam, [])) for cam, _ in frames)
    written = sum(len(areas) for fs in submission.values() for areas in fs.values())
    assert written == expected, f"wrote {written} areas, expected {expected}"

    for path, data in ((results_path, results), (submission_path, submission)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
    n_fallback = sum("fallback" in r["reason"] or "no sources" in r["reason"]
                     for fs in results.values() for areas in fs.values() for r in areas.values())
    print(f"decide: {len(frames)} frames, {written} areas ({n_fallback} on the prior) -> {submission_path}")
    if queue_path:
        print(f"decide: {write_queue(results, queue_path)}/{written} low-confidence areas -> {queue_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--vlm", type=Path, default=VLM)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--submission", type=Path, default=SUBMISSION)
    ap.add_argument("--run-frames", type=Path, default=RUN_FRAMES)
    ap.add_argument("--images", type=Path, default=IMAGES_DIR, help="frames to decide when run_frames.json is missing")
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    ap.add_argument("--queue", type=Path, help="also write the low-confidence areas here, for vlm.ask --areas")
    args = ap.parse_args()
    run(args.planogram, args.features, args.vlm, args.results, args.submission, args.run_frames, args.images, args.index,
        args.queue)
