"""Spec for pipeline/decide/decide.py: EV rule, compliance threshold, every area always output."""
import itertools
import json

from pipeline.common.planogram import load_planogram
from pipeline.decide.decide import (
    DEFAULT_PARAMS, LEVELS, decide_area, decide_frames, is_compliant, p_wrong, stock_level, to_submission,
    vlm_stock_probs,
)

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
