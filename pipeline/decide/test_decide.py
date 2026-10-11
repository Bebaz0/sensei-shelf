"""Spec for pipeline/decide/decide.py: EV rule, compliance threshold, every area always output."""
import itertools
import json

from pipeline.common.planogram import load_planogram
from pipeline.decide.decide import (
    DEFAULT_PARAMS, LEVELS, decide_area, decide_frames, is_compliant, p_wrong, stock_level, to_submission,
    vlm_stock_probs, write_queue,
)
from pipeline.vlm.ask import read_area_list

PARAMS = json.loads(json.dumps(DEFAULT_PARAMS)) | {"stock_bias": 0.0}
GAIN = {(t, t): 1 for t in LEVELS} | {
    (t, p): (-2 if LEVELS.index(p) < LEVELS.index(t) else -3) for t in LEVELS for p in LEVELS if p != t
}


def _p(e, h, f):
    return {"empty": e, "half": h, "full": f}


def test_ev_rule_examples():
    assert stock_level(_p(1 / 3, 1 / 3, 1 / 3), 0) == "empty"
    assert stock_level(_p(0.1, 0.4, 0.5), 0) == "half"
    assert stock_level(_p(0, 0, 1), 0) == "full"
    assert stock_level(_p(0, 1, 0), 0) == "half"


def test_ev_rule_maximises_expected_score_on_a_grid():
    for i, j in itertools.product(range(21), repeat=2):
        if i + j > 20:
            continue
        p = _p(i / 20, j / 20, (20 - i - j) / 20)
        ev = {pred: sum(p[t] * GAIN[(t, pred)] for t in LEVELS) for pred in LEVELS}
        assert ev[stock_level(p, 0)] >= max(ev.values()) - 1e-9


def test_bias_never_raises_the_level():
    for i, j in itertools.product(range(11), repeat=2):
        if i + j > 10:
            continue
        p = _p(i / 10, j / 10, (10 - i - j) / 10)
        assert LEVELS.index(stock_level(p, 0.3)) <= LEVELS.index(stock_level(p, 0))


def test_compliance_threshold():
    assert is_compliant(0.49) and not is_compliant(0.51)


def test_vlm_fill_pct_uses_mentor_thresholds():
    assert max(vlm_stock_probs({"fill_pct": 0}, 10), key=vlm_stock_probs({"fill_pct": 0}, 10).get) == "empty"
    assert max(vlm_stock_probs({"fill_pct": 30}, 10), key=vlm_stock_probs({"fill_pct": 30}, 10).get) == "half"
    assert max(vlm_stock_probs({"fill_pct": 80}, 10), key=vlm_stock_probs({"fill_pct": 80}, 10).get) == "full"


def test_p_wrong_needs_a_confident_mismatch():
    def vlm(match, conf, observed="x"):
        return {"gemini": {"matches_expected": match, "confidence": conf, "observed": observed}}
    assert not is_compliant(p_wrong(None, vlm(False, 0.9), PARAMS))
    assert is_compliant(p_wrong(None, vlm(False, 0.5), PARAMS))
    assert is_compliant(p_wrong(None, vlm(True, 0.9), PARAMS))
    assert p_wrong(None, vlm(False, 0.9, "empty"), PARAMS) < 0.5
    assert p_wrong(None, None, PARAMS) == 0.0


def test_empty_area_is_compliant():
    vlm = {"gemini": {"stock": "empty", "fill_pct": 0, "confidence": 1.0, "matches_expected": False, "observed": "123"}}
    row = decide_area({}, None, vlm, PARAMS)
    assert row["pred"]["stock_level"] == "empty" and row["pred"]["planogram_compliant"]


def test_broken_input_falls_back():
    row = decide_area({}, {"knn": "garbage"}, {"gemini": 42}, PARAMS)
    assert row["pred"]["planogram_compliant"] is True
    assert row["pred"]["stock_level"] == "half" and row["reason"].startswith("fallback")


def test_every_area_of_every_frame_with_no_inputs():
    planogram = load_planogram()
    frames = [(cam, "1.jpg") for cam in planogram] + [("6215", "2.jpg")]
    sub = to_submission(decide_frames(frames, planogram, {}, {}, PARAMS))
    for cam, fn in frames:
        assert set(sub[cam][fn]) == {a["area_id"] for a in planogram[cam]}
        for pred in sub[cam][fn].values():
            assert set(pred) == {"planogram_compliant", "stock_level", "stock_value"}
    assert sum(len(a) for fs in sub.values() for a in fs.values()) == 152 + len(planogram["6215"])


def _synthetic(n_frames=8, seed=0):
    """Labels where gemini reads the truth and knn is noise; a few frames hold the wrong product."""
    import random
    rng = random.Random(seed)
    planogram = load_planogram()
    cam = "6215"
    labels, features, vlm = {cam: {}}, {cam: {}}, {cam: {}}
    for f in range(n_frames):
        fn = f"{1000 + f}.jpg"
        for area in planogram[cam]:
            a = area["area_id"]
            truth = rng.choice(LEVELS)
            wrong = a == planogram[cam][0]["area_id"] and truth != "empty"
            fill = {"empty": 3, "half": 30, "full": 80}[truth]
            labels[cam].setdefault(fn, {})[a] = {"planogram_compliant": not wrong, "stock_value": {"empty": 0.0, "half": 0.5, "full": 1.0}[truth]}
            vlm[cam].setdefault(fn, {})[a] = {"gemini": {"stock": truth, "fill_pct": fill, "confidence": 0.9,
                                                         "matches_expected": not wrong, "observed": "x"}, "claude": None}
            noise = rng.choice(LEVELS)
            features[cam].setdefault(fn, {})[a] = {"knn": {k: float(k == noise) for k in LEVELS} | {"p_wrong": 0.1}}
    return planogram, labels, features, vlm


def test_vectorised_fit_matches_decide():
    import numpy as np
    from pipeline.decide.lofo import SOURCES, build_table, fit, p_wrong_vec, stock_pred
    planogram, labels, features, vlm = _synthetic()
    t = build_table(labels, features, vlm, PARAMS["fill_sigma"])
    idx = np.arange(len(t["T"]))
    params = fit(t, idx)
    assert params["weights"]["gemini"] > params["weights"]["knn"]
    w = np.array([params["weights"][s] for s in SOURCES])
    fb = np.array([params["fallback_p"][k] for k in LEVELS])
    levels = stock_pred(t, idx, w, params["stock_bias"], fb, params["temporal_below"])
    pws = p_wrong_vec(t, idx, params["platt"]["a"], params["platt"]["b"], params["knn_wrong_weight"])
    rows = decide_frames(sorted({f for f in t["frame"]}), planogram, features, vlm, params)
    flat = [r for fn in sorted(rows["6215"]) for r in rows["6215"][fn].values()]
    assert [LEVELS.index(r["pred"]["stock_level"]) for r in flat] == list(levels)
    assert np.allclose([r["p_wrong"] for r in flat], pws, atol=1e-3)


def test_lofo_beats_prior_on_synthetic_labels():
    import numpy as np
    from pipeline.decide.lofo import GAIN, LEVEL_OF_VALUE, lofo_predictions
    planogram, labels, features, vlm = _synthetic()

    def stock_points(pred):
        return sum(GAIN[LEVELS.index(LEVEL_OF_VALUE[g["stock_value"]]), LEVELS.index(pred["6215"][fn][a]["stock_level"])]
                   for fn, areas in labels["6215"].items() for a, g in areas.items())
    assert stock_points(lofo_predictions(labels, features, vlm, planogram)) > stock_points(
        lofo_predictions(labels, features, vlm, planogram, allowed=()))
    assert np.isfinite(stock_points(lofo_predictions(labels, features, vlm, planogram, allowed=("knn",))))


def test_stacker_runs_on_synthetic_labels():
    from pipeline.decide.fuse import stacker_lofo
    planogram, labels, features, vlm = _synthetic()
    pred = stacker_lofo(labels, features, vlm, planogram)
    assert all(set(pred["6215"][fn]) == set(areas) for fn, areas in labels["6215"].items())


def test_vectorised_matches_scalar_on_random_inputs(monkeypatch):
    import random
    import numpy as np
    import pipeline.decide.decide as decide_mod
    from pipeline.decide.lofo import SOURCES, build_table, p_wrong_vec, stock_pred
    monkeypatch.setattr(decide_mod, "product_changed", lambda *a: False)  # restock rule is not in the vectorised path
    rng = random.Random(1)
    planogram = load_planogram()
    cam = "6215"
    labels, features, vlm = {cam: {}}, {cam: {}}, {cam: {}}
    for f in range(3):
        fn = f"{2000 + f}.jpg"
        for area in planogram[cam]:
            a = area["area_id"]
            labels[cam].setdefault(fn, {})[a] = {"planogram_compliant": True, "stock_value": 0.5}
            ans = lambda: rng.choice([None, {"stock": rng.choice(LEVELS), "fill_pct": rng.uniform(0, 100), "confidence": rng.random(),
                                              "matches_expected": rng.random() < 0.7, "observed": rng.choice(["x", "empty", "occluded", "other"])}])
            vlm[cam].setdefault(fn, {})[a] = {"gemini": ans(), "claude": ans()}
            probs = lambda: (lambda w: {k: x / sum(w) for k, x in zip(LEVELS, w)})([rng.random() + 0.01 for _ in LEVELS])
            features[cam].setdefault(fn, {})[a] = {k: v for k, v in {"knn": rng.choice([None, probs() | {"p_wrong": rng.random()}]),
                                                                    "probe": rng.choice([None, probs()]), "gb": rng.choice([None, probs()]),
                                                                    "temporal": {"prev_stock": rng.choice([None, *LEVELS])}}.items() if v}
    params = {**PARAMS, "weights": {"gemini": 1.0, "claude": 0.5, "knn": 2.0, "probe": 1.5, "gb": 1.0, "temporal": 0.5},
              "knn_wrong_weight": 0.4, "stock_bias": 0.1}
    t = build_table(labels, features, vlm, params["fill_sigma"])
    idx = np.arange(len(t["T"]))
    w = np.array([params["weights"][s] for s in SOURCES])
    fb = np.array([params["fallback_p"][k] for k in LEVELS])
    levels = stock_pred(t, idx, w, params["stock_bias"], fb, params["temporal_below"])
    pws = p_wrong_vec(t, idx, params["platt"]["a"], params["platt"]["b"], params["knn_wrong_weight"])
    rows = decide_frames(sorted(set(t["frame"])), planogram, features, vlm, params)
    flat = [r for fn in sorted(rows[cam]) for r in rows[cam][fn].values()]
    assert [LEVELS.index(r["pred"]["stock_level"]) for r in flat] == list(levels)
    assert np.allclose([r["p_wrong"] for r in flat], pws, atol=1e-3)


def test_ablation_removes_a_source_from_p_wrong_too():
    from pipeline.decide.lofo import restrict
    feats = {"c": {"f": {"a": {"knn": {"p_wrong": 0.9}, "fill_ratio": 0.2}}}}
    vlm = {"c": {"f": {"a": {"gemini": {"stock": "full"}, "claude": {"stock": "half"}}}}}
    f2, v2 = restrict(feats, vlm, ("gemini",))
    assert "knn" not in f2["c"]["f"]["a"] and f2["c"]["f"]["a"]["fill_ratio"] == 0.2
    assert "claude" not in v2["c"]["f"]["a"] and "gemini" in v2["c"]["f"]["a"]


def test_gate_queues_only_unsure_areas(tmp_path):
    sure = {"knn": {"empty": 0.0, "half": 0.05, "full": 0.95, "p_wrong": 0.02}}
    split = {"knn": {"empty": 0.3, "half": 0.4, "full": 0.3, "p_wrong": 0.02}}
    results = {"6215": {"1.jpg": {
        "6215_1": decide_area({}, sure, None, PARAMS),
        "6215_2": decide_area({}, split, None, PARAMS),
        "6215_3": decide_area({}, None, None, PARAMS),  # no source at all
    }}}
    path = tmp_path / "queue.txt"
    assert write_queue(results, path) == 2
    assert read_area_list(path) == {("6215", "1.jpg", "6215_2"), ("6215", "1.jpg", "6215_3")}


def test_temporal_only_breaks_ties():
    params = PARAMS | {"weights": PARAMS["weights"] | {"knn": 2.0, "temporal": 0.25}}
    prev_half = {"temporal": {"prev_stock": "half"}}
    split_knn = {"knn": {"empty": 0.05, "half": 0.4, "full": 0.55, "p_wrong": 0.0}}
    assert decide_area({}, split_knn, None, params)["pred"]["stock_level"] == "full"  # without the previous frame
    split = decide_area({}, prev_half | split_knn, None, params)
    sure = decide_area({}, prev_half | {"knn": {"empty": 0.0, "half": 0.05, "full": 0.95, "p_wrong": 0.0}}, None, params)
    assert split["pred"]["stock_level"] == "half"  # split decision follows the previous frame
    assert sure["pred"]["stock_level"] == "full"  # restocked, clearly full shelf is not overridden
    alone = decide_area({}, prev_half, None, params)
    assert alone["low_confidence"]  # the previous frame alone is no evidence: still queued for the VLM


def test_temporal_joins_only_when_visual_is_unsure():
    from pipeline.decide.decide import DEFAULT_PARAMS, stock_probs
    params = {**DEFAULT_PARAMS, "weights": {"knn": 1.0, "temporal": 1.0}}
    prev_empty = {"empty": 0.8, "half": 0.1, "full": 0.1}
    sure = {"knn": {"empty": 0.05, "half": 0.05, "full": 0.9}, "temporal": prev_empty}
    unsure = {"knn": {"empty": 0.2, "half": 0.3, "full": 0.5}, "temporal": prev_empty}
    assert stock_probs(sure, params)["full"] == 0.9
    assert stock_probs(unsure, params)["empty"] > 0.4


def test_product_change_drops_temporal_and_votes_full():
    from pipeline.decide.decide import DEFAULT_PARAMS, decide_area
    area = {"area_id": "x_1", "product_id": "111"}
    feats = {"knn": {"empty": 0.4, "half": 0.3, "full": 0.3, "p_wrong": 0.1},
             "temporal": {"prev_stock": "empty", "prev_product": "222"}}
    vlm = {"gemini": {"matches_expected": True, "observed": "111", "confidence": 0.9}}
    changed = decide_area(area, feats, vlm, DEFAULT_PARAMS)
    same = decide_area(area, {**feats, "temporal": {"prev_stock": "empty", "prev_product": None}}, vlm, DEFAULT_PARAMS)
    assert changed["p_stock"]["full"] > same["p_stock"]["full"]
