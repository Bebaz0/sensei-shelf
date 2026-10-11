"""Owner C. Leave-one-frame-out (LOFO) evaluation of decide, and the fit that writes params.json.

For every labelled frame: fit the decide parameters on the OTHER labelled frames, predict the held-out
frame with decide.decide_frames, then score all held-out predictions together. Fitted parameters:
  - fallback_p: label distribution (add-one smoothed)
  - weights per stock source and stock_bias: grid search on the training score
  - platt a, b: one logistic regression of "wrong product" on the VLM mismatch signal
  - knn_wrong_weight: grid search on the training compliance score

  python -m pipeline.decide.lofo                 # LOFO report
  python -m pipeline.decide.lofo --ablation      # one row per source set (pitch table)
  python -m pipeline.decide.lofo --gate          # no-VLM LOFO split into confident vs queued-for-VLM areas
  python -m pipeline.decide.lofo --forward       # forward-in-time: kNN and temporal use only earlier frames
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

SOURCES = ("gemini", "claude", "knn", "probe", "gb", "temporal")
WEIGHT_GRID = (0.0, 1.0, 2.0)
TEMPORAL_BELOW_GRID = (0.6, 0.7, 0.8, 0.9, 1.01)  # 1.01 = temporal always joins (old behaviour)
TEMPORAL_GRID = (0.0, 0.25, 0.5, 1.0)  # tie-breaker: small weights only
NO_VLM = ("knn", "probe", "gb", "temporal")
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


def stock_pred(t: dict, idx: np.ndarray, w: np.ndarray, bias: float, fallback: np.ndarray, below: float = 1.01) -> np.ndarray:
    """Vectorised decide.stock_probs + decide.stock_level. below: temporal joins only where the visual top p is under it."""
    M = t["M"][idx].copy()
    jt = SOURCES.index("temporal")
    wv = w.copy()
    wv[jt] = 0
    dv = M @ wv
    pv = np.einsum("nsk,s,ns->nk", t["S"][idx], wv, M) / np.maximum(dv, 1e-12)[:, None]
    M[:, jt] *= ~((dv > 0) & (pv.max(axis=1) >= below))
    den = M @ w
    num = np.einsum("nsk,s,ns->nk", t["S"][idx], w, M)
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
    best = (-np.inf, None, None, None)
    grids = [TEMPORAL_GRID if SOURCES[j] == "temporal" else WEIGHT_GRID for j in present]
    for ws in itertools.product(*grids):
        w = np.zeros(len(SOURCES))
        w[present] = ws
        for bias, below in itertools.product(BIAS_GRID, TEMPORAL_BELOW_GRID):
            level = stock_pred(t, idx, w, bias, fallback, below)
            g = GAIN[t["T"][idx], level].sum()
            if g > best[0]:
                best = (g, w, bias, below)
    _, w, bias, below = best
    params["temporal_below"] = float(below)
    params["weights"] = {s: float(w[j]) for j, s in enumerate(SOURCES)}
    params["stock_bias"] = float(bias)
    level = stock_pred(t, idx, w, bias, fallback, below)

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
    return nest(features, tuple(s for s in ("knn", "probe", "gb", "temporal") if s not in allowed)), \
        nest(vlm, tuple(s for s in ("gemini", "claude") if s not in allowed))


def lofo_predictions(
    labels: dict, features: dict, vlm: dict, planogram: dict, allowed: tuple[str, ...] = SOURCES,
    probs: dict | None = None, low: dict | None = None,
) -> dict:
    """Submission-format predictions for every labelled frame, each made with parameters fitted without it.
    probs, if given, is filled with {(cam, filename, area_id): p_stock} for log-loss; low with the low_confidence flag."""
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
        if low is not None:
            low.update({(c, f, a): r["low_confidence"] for c, fs in rows.items() for f, areas in fs.items() for a, r in areas.items()})
        sub = to_submission(rows)
        for cam, fs in sub.items():
            pred.setdefault(cam, {}).update(fs)
    return pred


def gate_report(pred: dict, gt: dict, low: dict) -> str:
    """Points per area of the no-VLM predictions on confident vs queued areas: the gate is good when queued is clearly lower."""
    lines = []
    for name, want in (("confident", False), ("queued", True)):
        sub = {c: {f: {a: g for a, g in areas.items() if low.get((c, f, a)) == want} for f, areas in fs.items()} for c, fs in gt.items()}
        s = score_pred(pred, sub)
        per = s["total"] / s["areas"] if s["areas"] else float("nan")
        lines.append(f"{name:10} {s['areas']:5} areas  {per:+.2f} pts/area  compliance {s['compliance']:5} stock {s['stock']:6}")
    return "\n".join(lines)


def score_pred(pred: dict, gt: dict) -> dict:
    """Official score via B's scorer; split into compliance and stock points."""
    from pipeline.eval.score import score
    res = score(pred, gt)
    pts = [a for fs in res["breakdown"].values() for areas in fs.values() for a in areas.values()]
    return {"total": res["total"], "max": res["max"], "areas": len(pts),
            "compliance": sum(a["planogram"] for a in pts), "stock": sum(a["stock"] for a in pts)}


def rebuilt_features(features: dict, labels: dict, causal: bool) -> dict:
    """features with knn and temporal rebuilt from the given labels; causal = earlier frames only (the deployment case:
    validation is later in the evening). Rebuilding both ways keeps a stale features.json out of the comparison."""
    from pipeline.common.config import CROPS_INDEX
    from pipeline.pixel.neighbours import add_neighbours
    from pipeline.pixel.stockmodel import add_gb
    out = json.loads(json.dumps(features))
    for areas in (a for fs in out.values() for f in fs.values() for a in f.values()):
        areas.pop("knn", None)
        areas.pop("temporal", None)
        areas.pop("gb", None)
    index = _load(CROPS_INDEX)
    add_neighbours(out, index, labels, causal=causal)
    add_gb(out, index, labels, causal=causal)
    return out


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
    ap.add_argument("--forward", action="store_true", help="LOFO with kNN and temporal from earlier frames only")
    ap.add_argument("--gate", action="store_true", help="no-VLM LOFO split into confident vs queued areas")
    ap.add_argument("--params", type=Path, default=PARAMS_PATH)
    args = ap.parse_args()

    labels, features, vlm = _clean_labels(_load(args.labels)), _load(args.features), _load(args.vlm)
    planogram = load_planogram(args.planogram)
    n_areas = sum(len(a) for fs in labels.values() for a in fs.values())
    print(f"{n_areas} labelled areas in {sum(len(fs) for fs in labels.values())} frames")

    if args.forward:
        for name, causal in (("LOFO (both sides)", False), ("forward in time", True)):
            feats = rebuilt_features(features, labels, causal)
            print(f"== {name}")
            for src, allowed in (("prior only", ()), ("kNN only", ("knn",)), ("temporal only", ("temporal",)), ("kNN + temporal", ("knn", "temporal")), ("gb only", ("gb",)), ("kNN + gb + temporal", NO_VLM)):
                probs: dict = {}
                print(_row(src, lofo_predictions(labels, feats, vlm, planogram, allowed, probs), labels, probs))
            low: dict = {}
            print(gate_report(lofo_predictions(labels, feats, vlm, planogram, NO_VLM, low=low), labels, low))
    elif args.fit:
        t = build_table(labels, features, vlm, DEFAULT_PARAMS["fill_sigma"])
        params = fit(t, np.arange(len(t["T"])))
        args.params.write_text(json.dumps(params, indent=1) + "\n")
        print(f"wrote {args.params}: {json.dumps(params)}")
    elif args.gate:
        low: dict = {}
        pred = lofo_predictions(labels, features, vlm, planogram, NO_VLM, low=low)
        print(gate_report(pred, labels, low))
    else:
        sets = {"v1 (all)": SOURCES}
        if args.ablation:
            sets = {"prior only": (), **{f"{s} only": (s,) for s in SOURCES}, "vlm only": ("gemini", "claude"),
                    "no vlm": NO_VLM, "no vlm, no temporal": ("knn", "probe"), **sets}
        for name, allowed in sets.items():
            probs: dict = {}
            print(_row(name, lofo_predictions(labels, features, vlm, planogram, allowed, probs), labels, probs))
