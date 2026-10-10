"""Owner C. Leave-one-frame-out (LOFO) evaluation of decide, and the fit that writes params.json.

For every labelled frame: fit the decide parameters on the OTHER labelled frames, predict the held-out
frame with decide.decide_frames, then score all held-out predictions together. Fitted parameters:
  - fallback_p: label distribution (add-one smoothed)
  - weights per stock source and stock_bias: grid search on the training score
  - platt a, b: one logistic regression of "wrong product" on the VLM mismatch signal
  - knn_wrong_weight: grid search on the training compliance score

  python -m pipeline.decide.lofo                 # LOFO report
  python -m pipeline.decide.lofo --ablation      # one row per source set (pitch table)
  python -m pipeline.decide.lofo --fit           # fit on every label, write pipeline/decide/params.json
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline.common.config import EMPTY_IS_COMPLIANT, FEATURES, LABELS, PLANOGRAM, STOCK_VALUES, VLM
from pipeline.common.planogram import load_planogram
from pipeline.decide.decide import (
    DEFAULT_PARAMS, LEVELS, PARAMS_PATH, decide_frames, knn_wrong, stock_sources, to_submission, vlm_mismatch,
)

SOURCES = ("gemini", "claude", "knn", "probe")
WEIGHT_GRID = (0.0, 1.0, 2.0)
BIAS_GRID = (0.0, 0.1, 0.2, 0.4)
KNN_WRONG_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_WRONG_FOR_PLATT = 5
# GAIN[truth, pred]: +1 correct, -2 predicted lower, -3 predicted higher (official rules).
GAIN = np.array([[1, -3, -3], [-2, 1, -3], [-2, -2, 1]], dtype=float)
LEVEL_OF_VALUE = {v: k for k, v in STOCK_VALUES.items()}


def build_table(labels: dict, features: dict, vlm: dict, sigma: float) -> dict:
    """Arrays over every labelled area (frame, area id): source probs S (n, src, 3) with mask M, truths, Platt input X, kNN p_wrong K."""
    rows = []
    for cam, frames in labels.items():
        for fn, areas in frames.items():
            for area_id, gt in areas.items():
                if "stock_value" not in gt or gt.get("unsure"):
                    continue
                f = features.get(cam, {}).get(fn, {}).get(area_id)
                v = vlm.get(cam, {}).get(fn, {}).get(area_id)
                try:
                    sources = stock_sources(f, v, sigma)
                except Exception:  # noqa: BLE001  same per-area fallback as decide
                    sources = {}
                rows.append((cam, fn, area_id, sources, vlm_mismatch(v), knn_wrong(f), LEVELS.index(LEVEL_OF_VALUE[float(gt["stock_value"])]),
                             bool(gt.get("planogram_compliant", True))))
    n = len(rows)
    S, M = np.zeros((n, len(SOURCES), 3)), np.zeros((n, len(SOURCES)))
    for i, (_, _, _, sources, *_rest) in enumerate(rows):
        for j, name in enumerate(SOURCES):
            if name in sources:
                S[i, j] = [sources[name][lv] for lv in LEVELS]
                M[i, j] = 1
    return {
        "frame": [(r[0], r[1]) for r in rows], "area": [r[2] for r in rows], "S": S, "M": M,
        "X": np.array([np.nan if r[4] is None else r[4] for r in rows]),
        "K": np.array([np.nan if r[5] is None else float(r[5]) for r in rows]),
        "T": np.array([r[6] for r in rows], dtype=int),
        "C": np.array([r[7] for r in rows], dtype=bool),
    }


def stock_pred(t: dict, idx: np.ndarray, w: np.ndarray, bias: float, fallback: np.ndarray) -> np.ndarray:
    """Vectorised decide.stock_probs + decide.stock_level."""
    den = t["M"][idx] @ w
    num = np.einsum("nsk,s,ns->nk", t["S"][idx], w, t["M"][idx])
    p = np.where(den[:, None] > 0, num / np.maximum(den, 1e-12)[:, None], fallback)
    ev = p @ GAIN - bias * np.arange(3)  # EV[pred] = sum_t p_t GAIN[t, pred]
    return ev.argmax(axis=1)  # first max = lower level on ties, as in decide


def p_wrong_vec(t: dict, idx: np.ndarray, a: float, b: float, wk: float) -> np.ndarray:
    """Vectorised decide.p_wrong."""
    x, k = t["X"][idx], t["K"][idx]
    p_vlm = 1 / (1 + np.exp(-(a + b * np.nan_to_num(x))))
    blend = (1 - wk) * p_vlm + wk * np.nan_to_num(k)
    return np.where(np.isnan(x), np.where(np.isnan(k), 0.0, np.nan_to_num(k)), np.where(np.isnan(k), p_vlm, blend))


def compliance_gain(t: dict, idx: np.ndarray, pw: np.ndarray, level: np.ndarray) -> float:
    compliant = (pw <= 0.5) | ((level == 0) & EMPTY_IS_COMPLIANT)
    return float(np.where(compliant == t["C"][idx], 1, -2).sum())


def fit(t: dict, idx: np.ndarray, allowed: tuple[str, ...] = SOURCES) -> dict:
    """decide parameters fitted on the labelled areas idx (never on the frame being evaluated)."""
    params = json.loads(json.dumps(DEFAULT_PARAMS))
    counts = np.bincount(t["T"][idx], minlength=3) + 1.0
    fallback = counts / counts.sum()
    params["fallback_p"] = {lv: round(float(p), 4) for lv, p in zip(LEVELS, fallback)}

    present = [j for j, s in enumerate(SOURCES) if s in allowed and t["M"][idx, j].any()]
    best = (-np.inf, None, None)
    for ws in itertools.product(WEIGHT_GRID, repeat=len(present)):
        w = np.zeros(len(SOURCES))
        w[present] = ws
        for bias in BIAS_GRID:
            level = stock_pred(t, idx, w, bias, fallback)
            g = GAIN[t["T"][idx], level].sum()
            if g > best[0]:
                best = (g, w, bias)
    _, w, bias = best
    params["weights"] = {s: float(w[j]) for j, s in enumerate(SOURCES)}
    params["stock_bias"] = float(bias)
    level = stock_pred(t, idx, w, bias, fallback)

    has_x = idx[~np.isnan(t["X"][idx])]
    wrong = ~t["C"][has_x]
    if wrong.sum() >= MIN_WRONG_FOR_PLATT and (~wrong).sum() >= MIN_WRONG_FOR_PLATT:
        lr = LogisticRegression().fit(t["X"][has_x, None], wrong)
        params["platt"] = {"a": float(lr.intercept_[0]), "b": float(lr.coef_[0, 0])}
    a, b = params["platt"]["a"], params["platt"]["b"]
    params["knn_wrong_weight"] = max(
        KNN_WRONG_GRID, key=lambda wk: compliance_gain(t, idx, p_wrong_vec(t, idx, a, b, wk), level))
    return params


def restrict(features: dict, vlm: dict, allowed: tuple[str, ...]) -> tuple[dict, dict]:
    """Drop every disallowed source from the inputs, so ablations change stock AND p_wrong (VLM, kNN)."""
    def keep(area: dict | None, drop: tuple[str, ...]) -> dict | None:
        return {k: v for k, v in area.items() if k not in drop} if isinstance(area, dict) else area
    nest = lambda d, drop: {c: {f: {a: keep(v, drop) for a, v in areas.items()} for f, areas in fs.items()}
                            for c, fs in d.items()}
    return nest(features, tuple(s for s in ("knn", "probe") if s not in allowed)), \
        nest(vlm, tuple(s for s in ("gemini", "claude") if s not in allowed))


def lofo_predictions(
    labels: dict, features: dict, vlm: dict, planogram: dict, allowed: tuple[str, ...] = SOURCES,
    probs: dict | None = None,
) -> dict:
    """Submission-format predictions for every labelled frame, each made with parameters fitted without it.
    probs, if given, is filled with {(cam, filename, area_id): p_stock} for log-loss."""
    features, vlm = restrict(features, vlm, allowed)
    t = build_table(labels, features, vlm, DEFAULT_PARAMS["fill_sigma"])
    frames = sorted(set(t["frame"]))
    if len(frames) < 2:
        raise SystemExit(f"LOFO needs at least 2 labelled frames, got {len(frames)}")
    frame_of = np.array([frames.index(f) for f in t["frame"]])
    pred: dict = {}
    for i, frame in enumerate(frames):
        params = fit(t, np.flatnonzero(frame_of != i), allowed)
        rows = decide_frames([frame], planogram, features, vlm, params)
        if probs is not None:
            probs.update({(c, f, a): r["p_stock"] for c, fs in rows.items() for f, areas in fs.items() for a, r in areas.items()})
        sub = to_submission(rows)
        for cam, fs in sub.items():
            pred.setdefault(cam, {}).update(fs)
    return pred


def score_pred(pred: dict, gt: dict) -> dict:
    """Official score via B's scorer; split into compliance and stock points."""
    from pipeline.eval.score import score
    res = score(pred, gt)
    pts = [a for fs in res["breakdown"].values() for areas in fs.values() for a in areas.values()]
    return {"total": res["total"], "max": res["max"], "areas": len(pts),
            "compliance": sum(a["planogram"] for a in pts), "stock": sum(a["stock"] for a in pts)}


def late_subset(gt: dict, n: int = 2) -> dict:
    """The last n labelled frames of each camera (proxy for the evening validation frames)."""
    return {cam: {fn: fs[fn] for fn in sorted(fs)[-n:]} for cam, fs in gt.items()}


def _clean_labels(labels: dict) -> dict:
    return {cam: {fn: {a: g for a, g in areas.items() if "stock_value" in g and not g.get("unsure")}
                  for fn, areas in fs.items()} for cam, fs in labels.items()}


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def log_loss(probs: dict, gt: dict) -> float:
    """Mean -log p(true level) over labelled areas: calibration of the stock probabilities, which the EV rule relies on."""
    ll = [-math.log(max(probs[(c, f, a)][LEVEL_OF_VALUE[float(g["stock_value"])]], 1e-6))
          for c, fs in gt.items() for f, areas in fs.items() for a, g in areas.items() if (c, f, a) in probs]
    return sum(ll) / len(ll) if ll else float("nan")


def _row(name: str, pred: dict, gt: dict, probs: dict | None = None) -> str:
    s, late = score_pred(pred, gt), score_pred(pred, late_subset(gt))
    ll = f" logloss {log_loss(probs, gt):.3f}" if probs else ""
    return (f"{name:16} total {s['total']:6} / {s['max']:<5} compliance {s['compliance']:5} stock {s['stock']:6}{ll}"
            f"  | late frames {late['total']:5} / {late['max']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--vlm", type=Path, default=VLM)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--ablation", action="store_true", help="one LOFO row per source set")
    ap.add_argument("--fit", action="store_true", help="fit on every label and write params.json")
    ap.add_argument("--params", type=Path, default=PARAMS_PATH)
    args = ap.parse_args()

    labels, features, vlm = _clean_labels(_load(args.labels)), _load(args.features), _load(args.vlm)
    planogram = load_planogram(args.planogram)
    n_areas = sum(len(a) for fs in labels.values() for a in fs.values())
    print(f"{n_areas} labelled areas in {sum(len(fs) for fs in labels.values())} frames")

    if args.fit:
        t = build_table(labels, features, vlm, DEFAULT_PARAMS["fill_sigma"])
        params = fit(t, np.arange(len(t["T"])))
        args.params.write_text(json.dumps(params, indent=1) + "\n")
        print(f"wrote {args.params}: {json.dumps(params)}")
    else:
        sets = {"v1 (all)": SOURCES}
        if args.ablation:
            sets = {"prior only": (), **{f"{s} only": (s,) for s in SOURCES}, "vlm only": ("gemini", "claude"),
                    "no vlm": ("knn", "probe"), **sets}
        for name, allowed in sets.items():
            probs: dict = {}
            print(_row(name, lofo_predictions(labels, features, vlm, planogram, allowed, probs), labels, probs))
