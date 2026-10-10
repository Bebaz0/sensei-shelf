from pipeline.eval.merge_labels import _resolve


def lab(level, compliant=True):
    return {"stock_level": level, "stock_value": {"empty": 0.0, "half": 0.5, "full": 1.0}[level], "planogram_compliant": compliant}


def test_majority_wins():
    assert _resolve([lab("full"), lab("half"), lab("half")])["stock_level"] == "half"


def test_tie_takes_lowest_stock_and_non_compliant():
    r = _resolve([lab("full"), lab("half", False)])
    assert r["stock_value"] == 0.5 and r["planogram_compliant"] is False
