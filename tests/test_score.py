"""Spec for pipeline/eval/score.py: the brief's worked example must give 320."""
from pipeline.eval.score import score


def test_brief_example_gives_320():
    # 270 id ok, 19 id wrong, 230 stock ok, 35 below, 24 above (289 areas)
    pred, gt = {"cam": {"f.jpg": {}}}, {"cam": {"f.jpg": {}}}
    stock = ["ok"] * 230 + ["low"] * 35 + ["high"] * 24  # relative to truth
    truth_val = {"ok": 0.5, "low": 0.5, "high": 0.5}
    pv = {"ok": 0.5, "low": 0.0, "high": 1.0}
    for i in range(289):
        s = stock[i]
        gt["cam"]["f.jpg"][f"c_{i}"] = {"planogram_compliant": True, "stock_value": truth_val[s]}
        pred["cam"]["f.jpg"][f"c_{i}"] = {"planogram_compliant": i < 270, "stock_value": pv[s]}
    assert score(pred, gt)["total"] == 270 - 2 * 19 + 230 - 2 * 35 - 3 * 24


def test_missing_area_costs_five():
    gt = {"cam": {"f.jpg": {"a": {"planogram_compliant": True, "stock_value": 0.5}}}}
    assert score({"cam": {"f.jpg": {}}}, gt)["total"] == -5
