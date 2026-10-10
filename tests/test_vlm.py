"""Spec for pipeline/vlm: prompt content, answer checks, disk cache, null answers, run-frame selection. No network."""
import json

import pytest

from pipeline.common.config import ROOT
from pipeline.vlm.ask import run
from pipeline.vlm.prompt import answer_schema, area_prompt, clean_answer

ANSWER = {"reason": "bananas", "observed": "11908", "matches_expected": True, "fill_pct": 55, "stock": "half", "confidence": 0.8}
PRODUCTS = {"11908": {"name_pt": "BANANA CAT I CNT KG", "name_en": "Banana (standard)", "unit": "loose", "hint": "long yellow"}}


def test_prompt_names_the_expected_product_and_every_candidate():
    text = area_prompt("11908", {"11908": "BANANA CAT I CNT KG", "11906": "LARANJA KG"}, PRODUCTS)
    assert "11908: Banana (standard) [loose] - long yellow" in text
    assert "11906: LARANJA KG" in text  # not in products_en.json: falls back to the Portuguese name


def test_schema_limits_observed_to_candidates_and_no_product_values():
    assert answer_schema(["2", "1"])["properties"]["observed"]["enum"] == ["1", "2", "other", "empty", "occluded"]


def test_clean_answer_keeps_contract_keys_and_rejects_bad_values():
    assert clean_answer({**ANSWER, "fill_pct": 140, "extra": 1}, ["11908"]) == {
        "stock": "half", "fill_pct": 100, "matches_expected": True, "observed": "11908", "confidence": 0.8, "reason": "bananas"}
    with pytest.raises(ValueError):
        clean_answer({**ANSWER, "stock": "low"}, ["11908"])
    with pytest.raises(ValueError):
        clean_answer({**ANSWER, "observed": "99999"}, ["11908"])


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """A two-frame index with one area each, crops on disk; run_frames lists only the second frame."""
    monkeypatch.setattr("pipeline.vlm.ask.VLM_RPM", 0)
    monkeypatch.setattr("pipeline.vlm.ask.time.sleep", lambda seconds: None)
    index = {"6215": {}}
    for filename in ("1.jpg", "2.jpg"):
        entry = {"product_id": "11908", "product_name": "BANANA CAT I CNT KG"}
        for kind in ("ctx", "warp"):
            crop = tmp_path / f"{filename}_{kind}.png"
            crop.write_bytes(f"{filename} {kind}".encode())
            entry[kind] = str(crop)
        index["6215"][filename] = {"6215_2": entry}
    (tmp_path / "index.json").write_text(json.dumps(index))
    (tmp_path / "run_frames.json").write_text(json.dumps([["6215", "2.jpg"]]))
    (tmp_path / "products.json").write_text(json.dumps(PRODUCTS))
    return dict(index_path=tmp_path / "index.json", run_frames_path=tmp_path / "run_frames.json",
                out_path=tmp_path / "vlm.json", cache_dir=tmp_path / "cache", products_path=tmp_path / "products.json")


def test_run_asks_only_run_frames_and_caches_answers(paths):
    calls = []

    def call(images, system, text, schema):
        calls.append(images)
        return dict(ANSWER), {"input_tokens": 10, "output_tokens": 5}

    run(**paths, call=call)
    run(**paths, call=call)  # second run is served from the cache

    assert len(calls) == 1 and calls[0] == [b"2.jpg ctx", b"2.jpg warp"]
    out = json.loads(paths["out_path"].read_text())
    assert out == {"6215": {"2.jpg": {"6215_2": {"gemini": clean_answer(ANSWER, ["11908"]), "claude": None}}}}

    run(**paths, call=call, all_frames=True)
    assert set(json.loads(paths["out_path"].read_text())["6215"]) == {"1.jpg", "2.jpg"}


def test_no_api_and_failing_calls_write_null_for_every_area(paths):
    def failing_call(images, system, text, schema):
        raise RuntimeError("503")

    for kwargs in ({"no_api": True}, {"call": failing_call}):
        run(**paths, **kwargs)
        assert json.loads(paths["out_path"].read_text()) == {"6215": {"2.jpg": {"6215_2": {"gemini": None, "claude": None}}}}
    assert not paths["cache_dir"].exists()  # failures are never cached


def test_products_en_covers_the_planogram():
    from pipeline.common.planogram import load_planogram
    products = json.loads((ROOT / "data" / "products_en.json").read_text(encoding="utf-8"))
    planogram_ids = {a["product_id"] for areas in load_planogram().values() for a in areas}
    assert set(products) == planogram_ids
    assert all(p["unit"] in ("loose", "package") and p["name_en"] and p["hint"] for p in products.values())
