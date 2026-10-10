import cv2
import numpy as np

from pipeline.pixel.neighbours import add_neighbours

RED, GREEN, BLUE = (40, 40, 200), (40, 200, 40), (200, 40, 40)


def _setup(tmp_path, colours):
    """One area c1_1 over frames 100.jpg, 200.jpg, ... with the given BGR fills."""
    index = {"c1": {}}
    for i, colour in enumerate(colours):
        path = tmp_path / f"{i}.png"
        img = np.full((32, 32, 4), 255, dtype=np.uint8)
        img[:, :, :3] = colour
        cv2.imwrite(str(path), img)
        index["c1"][f"{(i + 1) * 100}.jpg"] = {"c1_1": {"warp": str(path)}}
    out = {"c1": {f: {"c1_1": {}} for f in index["c1"]}}
    return index, out


def _lab(level, compliant=True):
    return {"stock_level": level, "planogram_compliant": compliant}


def test_no_labels_adds_nothing(tmp_path):
    index, out = _setup(tmp_path, [RED, GREEN])
    add_neighbours(out, index, {})
    assert out["c1"]["100.jpg"]["c1_1"] == {}


def test_identical_neighbour_wins_and_own_frame_is_never_used(tmp_path):
    index, out = _setup(tmp_path, [RED, GREEN, RED])
    labels = {"c1": {"100.jpg": {"c1_1": _lab("full")}, "200.jpg": {"c1_1": _lab("empty", False)}}}
    add_neighbours(out, index, labels)
    knn = out["c1"]["300.jpg"]["c1_1"]["knn"]
    assert max(("empty", "half", "full"), key=knn.get) == "full"  # 300 looks like 100
    assert knn["p_wrong"] < 0.5
    own = out["c1"]["100.jpg"]["c1_1"]["knn"]  # only neighbour is 200 (green, far): close to the prior
    assert own["full"] < 0.5


def test_no_candidates_gives_no_knn_and_temporal_handles_last_frame(tmp_path):
    index, out = _setup(tmp_path, [RED, GREEN])
    labels = {"c1": {"100.jpg": {"c1_1": _lab("full")}}}
    add_neighbours(out, index, labels)
    a = out["c1"]["100.jpg"]["c1_1"]  # its own label is the only one: no candidates, no knn (decide queues it for the VLM)
    assert "knn" not in a
    assert a["temporal"]["prev_stock"] is None and a["temporal"]["next_stock"] is None
    b = out["c1"]["200.jpg"]["c1_1"]["temporal"]
    assert b == {"prev_stock": "full", "next_stock": None, "dt_prev_s": 100, "dt_next_s": None}
