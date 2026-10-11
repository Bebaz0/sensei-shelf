from pipeline.common.planogram import load_planogram
from pipeline.eval.check_submission import problems


def test_check_submission_flags_missing_extra_and_bad_values():
    planogram = load_planogram()
    frames = [("6215", "1.jpg")]
    ok = {"6215": {"1.jpg": {a["area_id"]: {"planogram_compliant": True, "stock_level": "full", "stock_value": 1.0}
                             for a in planogram["6215"]}}}
    assert problems(ok, frames, planogram) == []
    first = planogram["6215"][0]["area_id"]
    bad = {"6215": {"1.jpg": {**ok["6215"]["1.jpg"], first: {"planogram_compliant": True, "stock_level": "half", "stock_value": 1.0}},
                    "2.jpg": {}}}
    del bad["6215"]["1.jpg"][planogram["6215"][1]["area_id"]]
    found = problems(bad, frames, planogram)
    assert any(p.startswith("bad values") for p in found) and any(p.startswith("missing area") for p in found)
    assert any(p.startswith("extra image") for p in found)
