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

from pipeline.pixel.features import _load_image
from pipeline.pixel.neighbours import LEVELS, _level, _ts, add_neighbours

MIN_TRAIN = 150  # no gb below this many training rows
FOLDS = 10
PRIOR = (0.06, 0.13, 0.81)  # kNN stand-in when an area has no labelled neighbour
# Switches measured one by one on the forward-in-time score (docs/DECISIONS.md); a switch stays on only if it gained.
OPTS = {"refs": False, "soft": False, "model": "multi"}  # model: multi | ordinal | both
FAR = (0.05, 0.10, 0.15)  # Lab-grid cell distance (in 1/255 units) beyond which a cell no longer looks like the full bank


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


class _Both:
    def fit(self, X, y, sample_weight=None):
        self.a = _make().fit(X, y, sample_weight=sample_weight)
        self.b = _Ordinal().fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return (self.a.predict_proba(X) + self.b.predict_proba(X)) / 2


def _new_model():
    return {"multi": _make, "ordinal": _Ordinal, "both": _Both}[OPTS["model"]]()


def add_gb(out: dict, index: dict, labels: dict, causal: bool = False) -> None:
    """Add `gb` (stock probs) to every area of `out` that has a crop, in place. Needs `knn`/`temporal` already added
    with the same `causal`. No labels or too few earlier labels: nothing added."""
    rows = [(c, f, a, _level(lab)) for c, fs in labels.items() for f, areas in fs.items() for a, lab in areas.items()
            if _level(lab) and not lab.get("unsure")]
    if len(rows) < MIN_TRAIN:
        return
    vecs: dict[tuple, tuple] = {}
    for cam, frames in index.items():
        for fn, areas in frames.items():
            for area, d in areas.items():
                if (v := _crop_vec(d.get("warp", ""))) is not None:
                    vecs[(cam, fn, area)] = v
    illum: dict[tuple, list] = {}
    for (c, f, _), (grid, _b) in vecs.items():
        illum.setdefault((c, f), []).append(grid.mean((0, 1)))
    illum = {k: np.mean(v, 0) for k, v in illum.items()}
    base = {k: np.concatenate([(g - illum[k[:2]]).ravel(), b]) for k, (g, b) in vecs.items()}
    cells = {k: (g - illum[k[:2]]).reshape(64, 3) for k, (g, _b) in vecs.items()}
    stacks: dict[tuple, list] = {}
    for k, v in base.items():
        stacks.setdefault((k[0], k[2]), []).append(v)
    med = {k: np.median(np.stack(v), 0) for k, v in stacks.items()}

    # kNN/temporal of the labelled rows themselves, built from earlier labels only (as a target frame sees them)
    lab_frames: dict = {}
    for c, f, a, _ in rows:
        lab_frames.setdefault(c, {}).setdefault(f, {})[a] = {}
    add_neighbours(lab_frames, index, labels, causal=True)

    lab_by_area: dict[tuple, list] = {}
    for c, f, a, lv in rows:
        if _ts(f) is not None and (c, f, a) in cells:
            lab_by_area.setdefault((c, a), []).append((_ts(f), lv, (c, f, a)))

    def ref_feats(k: tuple) -> np.ndarray:
        """How far the crop is from the area's own labelled `full` and `empty` crops of EARLIER frames: share of the 64
        Lab cells no full crop resembles (3 tolerances, top and bottom half), worst-cell distance, distance to empty."""
        t = _ts(k[1])
        earlier = [(lv, kk) for ts, lv, kk in lab_by_area.get((k[0], k[2]), []) if t is not None and ts < t]
        full = np.stack([cells[kk] for lv, kk in earlier if lv == "full"]) if any(lv == "full" for lv, _ in earlier) else None
        empty = [cells[kk] for lv, kk in earlier if lv == "empty"]
        if full is None:
            fr = np.zeros(8)
        else:
            d = np.linalg.norm(full - cells[k][None], axis=2).min(axis=0)  # (64,) distance to the nearest full crop per cell
            fr = np.array([*(float((d > th).mean()) for th in FAR), float((d[:32] > 0.10).mean()), float((d[32:] > 0.10).mean()),
                           float(d.mean()), float(np.sort(d)[-16:].mean()), 1.0])
        de = float(np.linalg.norm(np.stack(empty) - cells[k][None], axis=2).mean(axis=1).min()) if empty else 0.5
        return np.concatenate([fr, [de, float(bool(empty))]])

    def feat(k: tuple, f: dict) -> np.ndarray | None:
        if k not in base:
            return None
        t, tmp = f.get("temporal") or {}, f.get("knn")
        prev = np.eye(3)[LEVELS.index(t["prev_stock"])] if t.get("prev_stock") in LEVELS else np.zeros(3)
        dt = (t["dt_prev_s"] or 0) / 3600 if t.get("prev_stock") else 12
        knn = np.log(np.array([tmp[lv] for lv in LEVELS] if tmp else PRIOR) + 1e-3)
        tod = (_ts(k[1]) or 0) % 86400 / 86400
        extra = ref_feats(k) if OPTS["refs"] else []
        return np.concatenate([base[k] - med[(k[0], k[2])], prev, [dt, tod], knn, extra])

    train = [(k, x, LEVELS.index(lv)) for c, f, a, lv in rows
             if (x := feat(k := (c, f, a), lab_frames[c][f][a])) is not None and _ts(f) is not None]
    # soft rows: a half/empty label that the area's earlier labelled crops all contradict (kNN gives it < 10%) is
    # probably borderline, so it counts half as much
    weight = {k: 0.5 if OPTS["soft"] and y < 2 and (lab_frames[k[0]][k[1]][k[2]].get("knn") or {}).get(LEVELS[y], 1) < 0.1 else 1.0
              for k, _, y in train}
    frames = sorted({(k[0], k[1]) for k, _, _ in train})
    fold_of = {fr: i % FOLDS for i, fr in enumerate(frames)}
    models: dict = {}

    def model(key, keep) -> GradientBoostingClassifier | None:
        if key not in models:
            sub = [(x, y, weight[k]) for k, x, y in train if keep(k)]
            ok = len(sub) >= MIN_TRAIN and len({y for _, y, _ in sub}) == 3
            models[key] = _new_model().fit(np.array([x for x, _, _ in sub]), [y for _, y, _ in sub],
                                           sample_weight=np.array([w for _, _, w in sub])) if ok else None
        return models[key]

    tmax = sorted({_ts(k[1]) for k, _, _ in train})[::5]  # coarse refit grid: 1 fit per ~5 labelled frames
    for cam, fr_dict in out.items():
        for fn, areas in fr_dict.items():
            t = _ts(fn)
            if causal:  # refit on a coarse time grid: the model for a frame sees only labels from earlier frames
                cut = max((s for s in tmax if s < (t or 0)), default=None)
                m = model(("t", cut), lambda k, cut=cut: _ts(k[1]) <= (cut or -1)) if cut else None
            elif (cam, fn) in fold_of:
                m = model(("f", fold_of[(cam, fn)]), lambda k, i=fold_of[(cam, fn)]: fold_of[k[:2]] != i)
            else:
                m = model(("all",), lambda k: True)
            if m is None:
                continue
            for area, feats in areas.items():
                x = feat((cam, fn, area), feats)
                if x is not None:
                    feats["gb"] = {lv: round(float(p), 4) for lv, p in zip(LEVELS, m.predict_proba(x[None])[0])}
