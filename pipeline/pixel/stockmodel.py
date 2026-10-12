"""Owner B. Learned stock source `gb`: gradient boosting on lighting-normalised Lab/edge crop features, the previous
labelled frame's stock and the kNN probabilities. Called by features.run() after add_neighbours.

Per area the crop vector is taken relative to that area's median over ALL frames (label-free), after subtracting the
frame's mean illumination, so it measures what changed on the shelf, not the light. Never trained on the frame it
predicts (CLAUDE.md rule 6): labelled targets use a frame-grouped fold model, `causal` uses only earlier frames.
"""
import copy
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from pipeline.common.parallel import pmap, tmap
from pipeline.pixel.features import _load_image
from pipeline.pixel.neighbours import LEVELS, _level, _ts, add_neighbours

MIN_TRAIN = 150  # no gb below this many training rows
FOLDS = 5
PRIOR = (0.06, 0.13, 0.81)  # kNN stand-in when an area has no labelled neighbour
# Switches measured one by one on the forward-in-time score (docs/DECISIONS.md); a switch stays on only if it gained.
OPTS = {"soft": False, "model": "multi", "floor": False, "coverage": True}  # model: multi | ordinal. Forward score 3008 (off) vs 3023 soft, 3028 ordinal, 3012 both; coverage 3099 -> 3144


def _crop_vec(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(8x8 Lab grid, 8 row bands of edge density + 8 of saturation) of a warp crop."""
    loaded = _load_image(path) if path else None
    if not loaded:
        return None
    rgb = loaded[0]
    grid = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA) / 255
    edges = (cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 50, 150) > 0).astype(np.float32)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32) / 255
    bands = [edges[i * 32:(i + 1) * 32].mean() * 5 for i in range(8)] + [sat[i * 32:(i + 1) * 32].mean() for i in range(8)]
    return grid, np.array(bands)


def _make() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)


class _Ordinal:
    """Frank and Hall: two binary boosters for P(y >= half) and P(y >= full), turned into three class probabilities."""

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        self.lo = _make().fit(X, y >= 1, sample_weight=sample_weight)
        self.hi = _make().fit(X, y >= 2, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        ge1, ge2 = self.lo.predict_proba(X)[:, 1], self.hi.predict_proba(X)[:, 1]
        ge2 = np.minimum(ge2, ge1)
        p = np.stack([1 - ge1, ge1 - ge2, ge2], axis=1)
        return p / p.sum(axis=1, keepdims=True)


def _new_model():
    return {"multi": _make, "ordinal": _Ordinal}[OPTS["model"]]()


def add_gb(out: dict, index: dict, labels: dict, causal: bool = False) -> None:
    """Add `gb` (stock probs) to every area of `out` that has a crop, in place. Needs `knn`/`temporal` already added
    with the same `causal`. No labels or too few earlier labels: nothing added."""
    rows = [(c, f, a, _level(lab)) for c, fs in labels.items() for f, areas in fs.items() for a, lab in areas.items()
            if _level(lab) and not lab.get("unsure")]
    if len(rows) < MIN_TRAIN:
        return
    crops = [(cam, fn, area) for cam, frames in index.items() for fn, areas in frames.items() for area in areas]
    vecs = {k: v for k, v in zip(crops, tmap(lambda k: _crop_vec(index[k[0]][k[1]][k[2]].get("warp", "")), crops))
            if v is not None}
    illum: dict[tuple, list] = {}
    for (c, f, _), (grid, _b) in vecs.items():
        illum.setdefault((c, f), []).append(grid.mean((0, 1)))
    illum = {k: np.mean(v, 0) for k, v in illum.items()}
    base = {k: np.concatenate([(g - illum[k[:2]]).ravel(), b]) for k, (g, b) in vecs.items()}
    stacks: dict[tuple, list] = {}
    for k, v in base.items():
        stacks.setdefault((k[0], k[2]), []).append(v)
    med = {k: np.median(np.stack(v), 0) for k, v in stacks.items()}
    # floor: label-free empty reference of each area, its crop with the least edge density over all frames (pre-opening
    # and sold-out frames show the bare crate); distance to it tells exposed floor from product
    floor: dict[tuple, np.ndarray] = {}
    for k, v in base.items():
        if (k[0], k[2]) not in floor or vecs[k][1][:8].sum() < floor[(k[0], k[2])][0]:
            floor[(k[0], k[2])] = (vecs[k][1][:8].sum(), v)

    if OPTS["coverage"]:  # produce-vs-fixture share of the crop, trained with the same causal / fold discipline
        from pipeline.pixel.coverage import N_OUT, coverage
        cov = coverage(index, labels, causal)

    # kNN/temporal of the labelled rows themselves, built from earlier labels only (as a target frame sees them)
    lab_frames: dict = {}
    for c, f, a, _ in rows:
        lab_frames.setdefault(c, {}).setdefault(f, {})[a] = {}
    add_neighbours(lab_frames, index, labels, causal=True)

    def feat(k: tuple, f: dict) -> np.ndarray | None:
        if k not in base:
            return None
        t, tmp = f.get("temporal") or {}, f.get("knn")
        prev = np.eye(3)[LEVELS.index(t["prev_stock"])] if t.get("prev_stock") in LEVELS else np.zeros(3)
        dt = (t["dt_prev_s"] or 0) / 3600 if t.get("prev_stock") else 12
        knn = np.log(np.array([tmp[lv] for lv in LEVELS] if tmp else PRIOR) + 1e-3)
        tod = (_ts(k[1]) or 0) % 86400 / 86400
        x = [base[k] - med[(k[0], k[2])], prev, [dt, tod], knn]
        if OPTS["coverage"]:
            c = cov.get(k)
            x += [c if c is not None else np.zeros(N_OUT), [float(c is not None)]]
        if OPTS["floor"]:
            fl = floor[(k[0], k[2])][1]
            x += [[np.linalg.norm(base[k] - fl)], base[k][-16:-8] - fl[-16:-8]]  # distance and edge bands vs the floor
        return np.concatenate(x)

    train = [(k, x, LEVELS.index(lv)) for c, f, a, lv in rows
             if (x := feat(k := (c, f, a), lab_frames[c][f][a])) is not None and _ts(f) is not None]
    # soft rows: a half/empty label that the area's earlier labelled crops all contradict (kNN gives it < 10%) is
    # probably borderline, so it counts half as much
    weight = {k: 0.5 if OPTS["soft"] and y < 2 and (lab_frames[k[0]][k[1]][k[2]].get("knn") or {}).get(LEVELS[y], 1) < 0.1 else 1.0
              for k, _, y in train}
    frames = sorted({(k[0], k[1]) for k, _, _ in train})
    fold_of = {fr: i % FOLDS for i, fr in enumerate(frames)}
    tmax = sorted({_ts(k[1]) for k, _, _ in train})[::5]  # coarse refit grid: 1 fit per ~5 labelled frames

    def plan(cam: str, fn: str) -> tuple | None:
        """(model key, training-row filter) of a frame, None when no model applies."""
        if causal:  # refit on a coarse time grid: the model for a frame sees only labels from earlier frames
            cut = max((s for s in tmax if s < (_ts(fn) or 0)), default=None)
            return (("t", cut), lambda k, cut=cut: _ts(k[1]) <= cut) if cut else None
        if (cam, fn) in fold_of:
            i = fold_of[(cam, fn)]
            return ("f", i), lambda k, i=i: fold_of[k[:2]] != i
        return ("all",), lambda k: True

    def fit(keep) -> GradientBoostingClassifier | None:
        sub = [(x, y, weight[k]) for k, x, y in train if keep(k)]
        if len(sub) < MIN_TRAIN or len({y for _, y, _ in sub}) < 3:
            return None
        return _new_model().fit(np.array([x for x, _, _ in sub]), [y for _, y, _ in sub],
                                sample_weight=np.array([w for _, _, w in sub]))

    plans = {(cam, fn): plan(cam, fn) for cam, fs in out.items() for fn in fs}
    keep_of = {pl[0]: pl[1] for pl in plans.values() if pl}
    keys = list(keep_of)
    models = dict(zip(keys, pmap(lambda key: fit(keep_of[key]), keys)))  # independent fits, one process each
    for cam, fr_dict in out.items():
        for fn, areas in fr_dict.items():
            m = models[plans[(cam, fn)][0]] if plans[(cam, fn)] else None
            if m is None:
                continue
            for area, feats in areas.items():
                x = feat((cam, fn, area), feats)
                if x is not None:
                    feats["gb"] = {lv: round(float(p), 4) for lv, p in zip(LEVELS, m.predict_proba(x[None])[0])}
