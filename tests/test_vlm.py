"""Spec for pipeline/vlm: prompt content, answer checks, disk cache, null answers, run-frame selection. No network."""
import json

import pytest

from pipeline.common.config import ROOT
from pipeline.vlm.ask import run
from pipeline.vlm.prompt import clean_answer, clean_frame_answer, frame_parts, frame_schema

ANSWER = {"reason": "bananas", "observed": "11908", "matches_expected": True, "fill_pct": 55, "stock": "half", "confidence": 0.8}
PRODUCTS = {"11908": {"name_pt": "BANANA CAT I CNT KG", "name_en": "Banana (standard)", "unit": "loose", "hint": "long yellow"}}


def test_prompt_lists_candidates_once_and_puts_each_picture_after_its_area_line():
    parts = frame_parts([("6215_1", "11908", b"img1"), ("6215_2", "11906", b"img2")],
                        {"11908": "BANANA CAT I CNT KG", "11906": "LARANJA KG"}, PRODUCTS)
    assert "11908: Banana (standard) [loose] - long yellow" in parts[0]
    assert "11906: LARANJA KG" in parts[0]  # not in products_en.json: falls back to the Portuguese name
    assert parts[1:5] == ["Area 6215_1, expected product 11908: Banana (standard)", b"img1",
                          "Area 6215_2, expected product 11906: LARANJA KG", b"img2"]


def test_schema_limits_areas_and_observed_values():
    schema = frame_schema(["6215_1", "6215_2"], ["2", "1"])["properties"]["areas"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["properties"]["area_id"]["enum"] == ["6215_1", "6215_2"]
    assert schema["items"]["properties"]["observed"]["enum"] == ["1", "2", "other", "empty", "occluded"]


def test_frame_answer_keeps_good_areas_and_nulls_missing_or_bad_ones():
    raw = {"areas": [{**ANSWER, "area_id": "a"}, {**ANSWER, "area_id": "b", "stock": "low"}, {**ANSWER, "area_id": "zz"}]}
    assert clean_frame_answer(raw, ["a", "b", "c"], ["11908"]) == {"a": clean_answer(ANSWER, ["11908"]), "b": None, "c": None}


def test_clean_answer_keeps_contract_keys_and_rejects_bad_values():
    assert clean_answer({**ANSWER, "fill_pct": 140, "extra": 1}, ["11908"]) == {
        "stock": "half", "fill_pct": 100, "matches_expected": True, "observed": "11908", "confidence": 0.8, "reason": "bananas"}
    with pytest.raises(ValueError):
        clean_answer({**ANSWER, "stock": "low"}, ["11908"])
    with pytest.raises(ValueError):
        clean_answer({**ANSWER, "observed": "99999"}, ["11908"])


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """A two-frame index with two areas each, crops on disk; run_frames lists only the second frame."""
    monkeypatch.setattr("pipeline.vlm.ask.VLM_RPM", 0)
    monkeypatch.setattr("pipeline.vlm.ask.time.sleep", lambda seconds: None)
    index = {"6215": {}}
    for filename in ("1.jpg", "2.jpg"):
        index["6215"][filename] = {}
        for area_id in ("6215_2", "6215_3"):
            entry = {"product_id": "11908", "product_name": "BANANA CAT I CNT KG"}
            for kind in ("ctx", "warp"):
                crop = tmp_path / f"{filename}_{area_id}_{kind}.png"
                crop.write_bytes(f"{filename} {area_id} {kind}".encode())
                entry[kind] = str(crop)
            index["6215"][filename][area_id] = entry
    (tmp_path / "index.json").write_text(json.dumps(index))
    (tmp_path / "run_frames.json").write_text(json.dumps([["6215", "2.jpg"]]))
    (tmp_path / "products.json").write_text(json.dumps(PRODUCTS))
    return dict(index_path=tmp_path / "index.json", run_frames_path=tmp_path / "run_frames.json",
                out_path=tmp_path / "vlm.json", cache_dir=tmp_path / "cache", products_path=tmp_path / "products.json")


def frame_call(calls: list):
    """Mock provider: answers every area of the schema, records the images of each call."""
    def call(parts, system, schema):
        calls.append([part for part in parts if isinstance(part, bytes)])
        area_ids = schema["properties"]["areas"]["items"]["properties"]["area_id"]["enum"]
        return {"areas": [{**ANSWER, "area_id": a} for a in area_ids]}, {"input_tokens": 10, "output_tokens": 5}
    return call


def test_run_asks_only_run_frames_in_one_call_per_frame_and_caches_answers(paths):
    calls = []
    run(**paths, call=frame_call(calls))
    run(**paths, call=frame_call(calls))  # second run is served from the cache

    assert calls == [[b"2.jpg 6215_2 ctx", b"2.jpg 6215_3 ctx"]]  # context crops only, one call for the frame
    out = json.loads(paths["out_path"].read_text())
    answer = {"gemini": clean_answer(ANSWER, ["11908"]), "claude": None}
    assert out == {"6215": {"2.jpg": {"6215_2": answer, "6215_3": answer}}}

    run(**paths, call=frame_call(calls), all_frames=True)
    assert set(json.loads(paths["out_path"].read_text())["6215"]) == {"1.jpg", "2.jpg"}


def test_per_area_makes_one_call_per_area(paths, tmp_path):
    calls = []
    run(**{**paths, "cache_dir": tmp_path / "cache_per_area"}, call=frame_call(calls), per_area=True)
    assert sorted(calls) == [[b"2.jpg 6215_2 ctx"], [b"2.jpg 6215_3 ctx"]]  # calls run in parallel


def test_no_api_and_failing_calls_write_null_for_every_area(paths):
    def failing_call(parts, system, schema):
        raise RuntimeError("503")

    null = {"gemini": None, "claude": None}
    for kwargs in ({"no_api": True}, {"call": failing_call}):
        run(**paths, **kwargs)
        assert json.loads(paths["out_path"].read_text()) == {"6215": {"2.jpg": {"6215_2": null, "6215_3": null}}}
    assert not paths["cache_dir"].exists()  # failures are never cached


def test_products_en_covers_the_planogram():
    from pipeline.common.planogram import load_planogram
    products = json.loads((ROOT / "data" / "products_en.json").read_text(encoding="utf-8"))
    planogram_ids = {a["product_id"] for areas in load_planogram().values() for a in areas}
    assert set(products) == planogram_ids
    assert all(p["unit"] in ("loose", "package") and p["name_en"] and p["hint"] for p in products.values())
