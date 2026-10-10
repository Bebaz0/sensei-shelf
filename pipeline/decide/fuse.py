"""Owner C. Experiment: learned stock stacker (multinomial logistic regression), scored with the same LOFO as v1.

Adopted only with a clear LOFO gain over v1 (log both rows in docs/DECISIONS.md). Compliance stays v1 (Platt).
Stock = EV rule on the stacker's probabilities, with the stock_bias fitted by v1 on the same training frames.

  python -m pipeline.decide.fuse      # prints v1 and stacker LOFO rows side by side
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from pipeline.common.config import EMPTY_IS_COMPLIANT, FEATURES, LABELS, PLANOGRAM, STOCK_VALUES, VLM
from pipeline.common.planogram import load_planogram
from pipeline.decide.decide import DEFAULT_PARAMS, LEVELS, decide_frames, stock_level, stock_sources, to_submission
from pipeline.decide.lofo import SOURCES, _clean_labels, _load, _row, build_table, fit, lofo_predictions

PIXEL_KEYS = ("fill_ratio", "hist_dist_ref", "edge_density", "visible_fraction")


def area_vector(features: dict | None, vlm: dict | None) -> list[float]:
    """Few, bounded inputs: each source's 3 probs + a present flag, pixel features, previous-frame stock."""
    features = features or {}
    try:
        sources = stock_sources(features, vlm, DEFAULT_PARAMS["fill_sigma"])
    except Exception:  # noqa: BLE001
        sources = {}
    x = []
    for s in SOURCES:
        x += [sources[s][k] for k in LEVELS] + [1.0] if s in sources else [0.0, 0.0, 0.0, 0.0]
    for k in PIXEL_KEYS:
        v = features.get(k)
        x += [float(v), 1.0] if isinstance(v, (int, float)) else [0.0, 0.0]
    prev = (features.get("temporal") or {}).get("prev_stock")
    x += [float(prev == k) for k in LEVELS]
    return x


def stacker_lofo(labels: dict, features: dict, vlm: dict, planogram: dict) -> dict:
    t = build_table(labels, features, vlm, DEFAULT_PARAMS["fill_sigma"])
    keys = [(cam, fn, a) for (cam, fn), a in zip(t["frame"], t["area"])]
    X = np.array([area_vector(features.get(c, {}).get(f, {}).get(a), vlm.get(c, {}).get(f, {}).get(a))
                  for c, f, a in keys])
    frames = sorted(set(t["frame"]))
    frame_of = np.array([frames.index(f) for f in t["frame"]])
    pred: dict = {}
    for i, frame in enumerate(frames):
        train = np.flatnonzero(frame_of != i)
        params = fit(t, train)
        rows = decide_frames([frame], planogram, features, vlm, params)
        sub = to_submission(rows)
        if len(set(t["T"][train])) > 1:
            lr = LogisticRegression(C=0.5, max_iter=2000).fit(X[train], t["T"][train])
            cam, fn = frame
            for j in np.flatnonzero(frame_of == i):
                p = dict.fromkeys(LEVELS, 0.0) | {LEVELS[c]: float(q) for c, q in zip(lr.classes_, lr.predict_proba(X[j:j + 1])[0])}
                level = stock_level(p, params["stock_bias"])
                row = sub[cam][fn][keys[j][2]]
                row["stock_level"], row["stock_value"] = level, STOCK_VALUES[level]
                # recompute compliance for the new level: both directions, not only "empty"
                row["planogram_compliant"] = rows[cam][fn][keys[j][2]]["p_wrong"] <= 0.5 or (level == "empty" and EMPTY_IS_COMPLIANT)
        for cam, fs in sub.items():
            pred.setdefault(cam, {}).update(fs)
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--vlm", type=Path, default=VLM)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    args = ap.parse_args()
    labels, features, vlm = _clean_labels(_load(args.labels)), _load(args.features), _load(args.vlm)
    planogram = load_planogram(args.planogram)
    print(_row("v1", lofo_predictions(labels, features, vlm, planogram), labels))
    print(_row("stacker", stacker_lofo(labels, features, vlm, planogram), labels))
