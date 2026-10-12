"""Owner B (experiment, after competition). Produce-vs-fixture coverage of a warp crop, an extra input of the `gb` stock
model. The mentors define stock by covered share (empty 0-10%, half 10-50%, full > 50%), so measure it directly.

The crop is split into 16x16 cells. A cell classifier is trained by weak supervision: every cell of a crop labelled
empty counts as fixture, every cell of a crop labelled full as produce (half crops are skipped, their cells are mixed).
Cell features are colour, texture and the same values relative to that cell's median over all frames of the area
(label-free). Never trained on the frame it scores (CLAUDE.md rule 6): `causal` uses only labels of earlier frames,
otherwise a frame-grouped fold model scores the labelled frames and a model on every label scores the rest.
"""
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.common.parallel import tmap
from pipeline.pixel.features import _load_image
from pipeline.pixel.neighbours import _level, _ts

CELL = 16  # px, on the 256x256 warp: 16 x 16 = 256 cells
MIN_CELLS = 2000  # no model below this many training cells of each class
MAX_CELLS = 40000  # training cells per class, sampled
FOLDS = 5
N_OUT = 6  # mean p, share of cells with p > 0.5, mean p of 4 row bands (front to back of the tray)


def cell_feats(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(cells x 9 features, valid mask) of a warp crop: Lab mean, Lab std, saturation, edge density, Laplacian energy."""
    loaded = _load_image(path) if path else None
    if not loaded:
        return None
    rgb, mask = loaded
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32) / 255
    edge = (cv2.Canny(gray, 50, 150) > 0).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)) / 64
    n = rgb.shape[0] // CELL

    def blocks(a: np.ndarray) -> np.ndarray:  # (H, W[, C]) -> (cells, CELL*CELL[, C])
        a = a[:n * CELL, :n * CELL]
        return a.reshape(n, CELL, n, CELL, *a.shape[2:]).swapaxes(1, 2).reshape(n * n, CELL * CELL, *a.shape[2:])

    lb = blocks(lab)
    f = np.concatenate([lb.mean(1), lb.std(1), blocks(sat).mean(1)[:, None], blocks(edge).mean(1)[:, None],
                        blocks(lap).mean(1)[:, None]], axis=1)
    return f, blocks(mask.astype(np.float32)).mean(1) > 0.5


def _summary(p: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return np.zeros(N_OUT)
    n = int(round(len(p) ** 0.5))
    rows = np.array_split(np.arange(n), 4)
    grid, vg = p.reshape(n, n), valid.reshape(n, n)
    bands = [grid[r][vg[r]].mean() if vg[r].any() else p[valid].mean() for r in rows]
    return np.array([p[valid].mean(), (p[valid] > 0.5).mean(), *bands])


def coverage(index: dict, labels: dict, causal: bool) -> dict[tuple, np.ndarray]:
    """{(cam, filename, area): N_OUT coverage values} for every crop that has a scoring model."""
    crops = [(c, f, a) for c, fs in index.items() for f, areas in fs.items() for a in areas]
    raw = {k: v for k, v in zip(crops, tmap(lambda k: cell_feats(index[k[0]][k[1]][k[2]].get("warp", "")), crops)) if v}
    stacks: dict[tuple, list] = {}
    for (c, f, a), (x, _) in raw.items():
        stacks.setdefault((c, a), []).append(x)
    med = {k: np.median(np.stack(v), 0) for k, v in stacks.items()}
    X = {k: np.concatenate([x, x - med[(k[0], k[2])]], axis=1) for k, (x, _) in raw.items()}

    rows = [(c, f, a, lv) for c, fs in labels.items() for f, areas in fs.items() for a, lab in areas.items()
            if (lv := _level(lab)) in ("empty", "full") and not lab.get("unsure") and (c, f, a) in X and _ts(f) is not None]
    frames = sorted({(c, f) for c, f, _, _ in rows})
    fold_of = {fr: i % FOLDS for i, fr in enumerate(frames)}
    tmax = sorted({_ts(f) for _, f in frames})[::5]  # coarse refit grid, as stockmodel

    def plan(cam: str, fn: str):
        if causal:
            cut = max((s for s in tmax if s < (_ts(fn) or 0)), default=None)
            return (("t", cut), lambda r, cut=cut: _ts(r[1]) <= cut) if cut else None
        if (cam, fn) in fold_of:
            i = fold_of[(cam, fn)]
            return ("f", i), lambda r, i=i: fold_of[(r[0], r[1])] != i
        return ("all",), lambda r: True

    def fit(keep):
        rng = np.random.default_rng(0)
        parts = []
        for want in ("empty", "full"):
            cells = [X[(c, f, a)][raw[(c, f, a)][1]] for c, f, a, lv in rows if lv == want and keep((c, f, a))]
            cells = np.concatenate(cells) if cells else np.zeros((0, 18))
            if len(cells) < MIN_CELLS:
                return None
            parts.append(cells[rng.choice(len(cells), min(len(cells), MAX_CELLS), replace=False)])
        y = np.r_[np.zeros(len(parts[0])), np.ones(len(parts[1]))]
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0).fit(np.concatenate(parts), y)

    plans = {(c, f): plan(c, f) for c, fs in index.items() for f in fs}
    keep_of = {pl[0]: pl[1] for pl in plans.values() if pl}
    keys = list(keep_of)
    models = {key: fit(keep_of[key]) for key in keys}  # serial: each fit already uses every core (OpenMP)
    out = {}
    for k in X:
        pl = plans[(k[0], k[1])]
        m = models.get(pl[0]) if pl else None
        if m is not None:
            out[k] = _summary(m.predict_proba(X[k])[:, 1], raw[k][1])
    return out


if __name__ == "__main__":
    p = np.r_[np.ones(128), np.zeros(128)]  # front half of the rows covered
    s = _summary(p, np.ones(256, bool))
    assert abs(s[0] - 0.5) < 1e-9 and s[1] == 0.5 and s[2] == 1 and s[5] == 0, s
    print("ok")
