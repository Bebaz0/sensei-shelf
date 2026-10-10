"""Tests for pipeline.eval.make_worklists (pure functions, no filesystem IO)."""
from pipeline.eval.make_worklists import build_worklist, pick_frames


def test_pick_frames_returns_all_when_fewer_than_n():
    assert pick_frames(["b.jpg", "a.jpg"], 4) == ["a.jpg", "b.jpg"]


def test_pick_frames_always_includes_first_and_last():
    files = [f"{i:03d}.jpg" for i in range(20)]
    picked = pick_frames(files, 4)
    assert len(picked) == 4
    assert picked[0] == files[0]
    assert picked[-1] == files[-1]
    assert picked == sorted(picked)


def test_build_worklist_only_assigned_cameras_and_dedup_calibration():
    planogram = {
        "6215": [{"area_id": "6215_1"}, {"area_id": "6215_2"}],
        "6371": [{"area_id": "6371_1"}],
    }
    frames = {"6215": ["a.jpg", "b.jpg"], "6371": ["x.jpg"]}
    calibration = [
        ("6215", "a.jpg", "6215_1"),   # duplicate, must not appear twice
        ("6215", "c.jpg", "6215_2"),   # new frame via calibration
        ("6371", "x.jpg", "6371_1"),   # not assigned, must be skipped
    ]
    items = build_worklist(["6215"], frames, planogram, calibration, frames_per_camera=2)
    assert items == [
        ("6215", "a.jpg", "6215_1"),
        ("6215", "a.jpg", "6215_2"),
        ("6215", "b.jpg", "6215_1"),
        ("6215", "b.jpg", "6215_2"),
        ("6215", "c.jpg", "6215_2"),
    ]
