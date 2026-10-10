"""Pure-function tests for pipeline/eval/label_tool.py (no cv2 UI)."""
import json
from pathlib import Path

from pipeline.eval.label_tool import (
    all_products, already_labelled, filter_products, load_worklist, write_label, _unique_products,
)


def test_load_worklist_from_calibration_file(tmp_path):
    f = tmp_path / "cal.txt"
    f.write_text(
        "# header comment\n"
        "6215 1751001268.jpg 6215_1  # 06:14 BANANAS\n"
        "\n"
        "6371 1751001169.jpg 6371_4\n"
    )
    wl = load_worklist(f, run_frames_path=Path("ignored"), planogram={})
    assert wl == [("6215", "1751001268.jpg", "6215_1"), ("6371", "1751001169.jpg", "6371_4")]


def test_load_worklist_full_mode_crosses_frames_with_planogram(tmp_path):
    run_frames = tmp_path / "run.json"
    run_frames.write_text(json.dumps([["6215", "a.jpg"], ["6215", "b.jpg"]]))
    planogram = {"6215": [{"area_id": "6215_1", "product_id": "p", "product_name": "x",
                           "image_coordinates": {}}]}
    wl = load_worklist(None, run_frames, planogram)
    assert wl == [("6215", "a.jpg", "6215_1"), ("6215", "b.jpg", "6215_1")]


def test_already_labelled_and_write_label_round_trip(tmp_path):
    labels: dict = {}
    assert not already_labelled(labels, "6215", "a.jpg", "6215_1")
    path = tmp_path / "diogo.json"
    write_label(path, labels, "6215", "a.jpg", "6215_1",
                {"planogram_compliant": True, "stock_level": "full", "stock_value": 1.0})
    assert already_labelled(labels, "6215", "a.jpg", "6215_1")
    reloaded = json.loads(path.read_text())
    assert reloaded["6215"]["a.jpg"]["6215_1"]["stock_value"] == 1.0


def test_unique_products_deduplicates_by_id():
    areas = [
        {"product_id": "p1", "product_name": "BANANA"},
        {"product_id": "p1", "product_name": "BANANA"},
        {"product_id": "p2", "product_name": "APPLE"},
    ]
    out = _unique_products(areas)
    assert [p["product_id"] for p in out] == ["p1", "p2"]


def test_full_mode_uses_given_frames_over_run_frames(tmp_path):
    planogram = {"6215": [{"area_id": "6215_1", "product_id": "p", "product_name": "x", "image_coordinates": {}}]}
    wl = load_worklist(None, tmp_path / "missing.json", planogram, frames=[("6215", "z.jpg")])
    assert wl == [("6215", "z.jpg", "6215_1")]


def test_picker_lists_every_product_this_camera_first_and_filters():
    mk = lambda i, n: {"area_id": f"a{i}", "product_id": str(i), "product_name": n, "image_coordinates": {}}
    planogram = {"1": [mk(1, "BANANA KG"), mk(2, "MACA GALA KG")], "2": [mk(2, "MACA GALA KG"), mk(3, "LARANJA KG")]}
    products = all_products(planogram, "2", {"3": {"name_en": "Orange"}})
    assert [p["product_id"] for p in products] == ["2", "3", "1"]  # camera 2 first, then the rest, no duplicates
    assert [p["product_id"] for p in filter_products(products, "orange")] == ["3"]
    assert [p["product_id"] for p in filter_products(products, "kg ban")] == ["1"]
