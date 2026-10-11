"""Owner B. kNN and temporal prior from labelled crops of the same area. Called by features.run().

Cameras are fixed, so an area always shows the same background and perspective: a crop is compared only with
labelled crops of the same area in OTHER frames (never the frame itself, CLAUDE.md rule 6).
"""
import math
from pathlib import Path

import numpy as np

from pipeline.common.config import STOCK_VALUES
from pipeline.pixel.features import _chi2_dist, _hsv_hist, _load_image

LEVELS = ("empty", "half", "full")
_LEVEL_OF_VALUE = {v: k for k, v in STOCK_VALUES.items()}
# ponytail: HSV histogram + fixed smoothing; the embedding probe (DINOv2/CLIP) is the upgrade if LOFO says so.
ALPHA = 1.0  # weight of the prior in the vote; one identical neighbour gives (prior + onehot) / 2
# Neighbour weight also decays with time-of-day distance, exp(-dt / TIME_SCALE_S): evening frames vote mostly with
# evening frames (stock runs down and lighting changes late in the day). Time of day, not absolute time, so a
# validation frame from another day still finds neighbours. None = off. LOFO (no VLM): 7 to 16 h all beat off
# (624 -> 633..643, the gain is compliance), 10 h is the middle of that plateau.
TIME_SCALE_S: float | None = 10 * 3600
DAY_S = 86400


def _level(lab: dict) -> str | None:
    return lab.get("stock_level") or _LEVEL_OF_VALUE.get(lab.get("stock_value"))


def _ts(filename: str) -> int | None:
    try:
        return int(Path(filename).stem)
    except ValueError:
        return None


def _before(f: str, filename: str) -> bool:
    """True if frame f was taken before filename (timestamps in the file names)."""
    a, b = _ts(f), _ts(filename)
    return a is not None and b is not None and a < b


def _tod_dist(a: int, b: int) -> int:
    """Seconds between two timestamps' times of day, wrapping at midnight."""
    d = abs(a - b) % DAY_S
    return min(d, DAY_S - d)


def _usable(labels: dict) -> list[tuple[str, str, str, dict]]:
    """(cam, filename, area, label) for every label with a stock level and not marked unsure."""
    return [(c, f, a, lab) for c, fs in labels.items() for f, areas in fs.items() for a, lab in areas.items()
            if _level(lab) and not lab.get("unsure")]


def _vote(prior: dict, votes: list[tuple[float, str]]) -> dict:
    """Smoothed weighted vote: (ALPHA * prior + sum w * onehot) / (ALPHA + sum w)."""
    total = ALPHA + sum(w for w, _ in votes)
    return {k: (ALPHA * prior[k] + sum(w for w, v in votes if v == k)) / total for k in prior}


def add_neighbours(out: dict, index: dict, labels: dict, causal: bool = False) -> None:
    """Add `knn` and `temporal` (CONTRACTS.md features.json) to every area of `out`, in place.

    `index` is the whole crops index (train crops are the bank during a validation run). No labels: nothing added.
    causal: only labels from EARLIER frames of the same camera are used (forward-in-time evaluation).
    """
    rows = _usable(labels)
    if not rows:
        return

    hists: dict[str, np.ndarray | None] = {}

    def hist(cam: str, filename: str, area: str) -> np.ndarray | None:
        path = index.get(cam, {}).get(filename, {}).get(area, {}).get("warp", "")
        if path not in hists:
            loaded = _load_image(path) if path else None
            hists[path] = _hsv_hist(*loaded) if loaded else None
        return hists[path]

    by_area: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    for c, f, a, lab in rows:
        by_area.setdefault((c, a), []).append((f, lab))

    priors: dict[tuple[str, str], tuple[dict, float]] = {}

    def prior(cam: str, filename: str) -> tuple[dict, float]:
        """Stock distribution and wrong-product rate of all labels outside this frame (add-one smoothed)."""
        if (cam, filename) not in priors:
            rest = [lab for c, f, _, lab in rows if (c, f) != (cam, filename) and (not causal or _before(f, filename))]
            n = len(rest) + len(LEVELS)
            stock = {k: (1 + sum(_level(lab) == k for lab in rest)) / n for k in LEVELS}
            wrong = (1 + sum(lab.get("planogram_compliant") is False for lab in rest)) / (len(rest) + 2)
            priors[(cam, filename)] = stock, wrong
        return priors[(cam, filename)]

    taus: dict[tuple[str, str], float] = {}

    def tau(cam: str, area: str) -> float:
        """Distance scale of an area: median pairwise distance between all its crops (label-free)."""
        if (cam, area) not in taus:
            hs = [h for f in index.get(cam, {}) if (h := hist(cam, f, area)) is not None]
            d = [_chi2_dist(hs[i], hs[j]) for i in range(len(hs)) for j in range(i + 1, len(hs))]
            taus[(cam, area)] = max(float(np.median(d)) if d else 1.0, 1e-6)
        return taus[(cam, area)]

    for cam, frames in out.items():
        for filename, areas in frames.items():
            p_stock, p_wrong = prior(cam, filename)
            t = _ts(filename)
            for area, feats in areas.items():
                others = [(f, lab) for f, lab in by_area.get((cam, area), []) if f != filename and (not causal or _before(f, filename))]
                h = hist(cam, filename, area)
                stock_votes, wrong_votes = [], []
                if h is not None:
                    for f, lab in others:
                        hn = hist(cam, f, area)
                        if hn is None:
                            continue
                        w = math.exp(-_chi2_dist(h, hn) / tau(cam, area))
                        if TIME_SCALE_S and t is not None and _ts(f) is not None:
                            w *= math.exp(-_tod_dist(t, _ts(f)) / TIME_SCALE_S)
                        stock_votes.append((w, _level(lab)))
                        wrong_votes.append((w, lab.get("planogram_compliant") is False))
                if stock_votes:  # no labelled neighbour: no knn, so decide sees no source and queues the area for the VLM
                    knn = _vote(p_stock, stock_votes)
                    knn["p_wrong"] = _vote({True: p_wrong, False: 1 - p_wrong}, wrong_votes)[True]
                    feats["knn"] = {k: round(v, 4) for k, v in knn.items()}

                timed = sorted((_ts(f), lab) for f, lab in others if _ts(f) is not None) if t is not None else []
                prev = next(((ft, lab) for ft, lab in reversed(timed) if ft < t), None)
                nxt = next(((ft, lab) for ft, lab in timed if ft > t), None)
                feats["temporal"] = {
                    "prev_stock": _level(prev[1]) if prev else None,
                    "next_stock": _level(nxt[1]) if nxt else None,
                    # product in the previous labelled frame: its observed id when it was not the planogram one
                    "prev_product": prev[1].get("observed_product_id") if prev and prev[1].get("planogram_compliant") is False else None,
                    "dt_prev_s": t - prev[0] if prev else None,
                    "dt_next_s": nxt[0] - t if nxt else None,
                }

