import json

import cv2
import numpy as np

from pipeline.pixel import features


def _warp(path, fill, hole=False):
    img = np.full((256, 256, 4), 255, dtype=np.uint8)
    img[:, :, :3] = fill
    if hole:
        img[:, :128, :3] = 0
        img[:, :128, 3] = 0
    cv2.imwrite(str(path), img)


def test_off_image_pixels_do_not_count(tmp_path):
    full, half = tmp_path / "full.png", tmp_path / "half.png"
    _warp(full, (30, 120, 200))
    _warp(half, (30, 120, 200), hole=True)
    rgb_f, mask_f = features._load_image(full)
    f_full = features.area_features(rgb_f, mask=mask_f)
    rgb, mask = features._load_image(half)
    assert not mask[:, :128].any() and mask[:, 128:].all()
    f_half = features.area_features(rgb, mask=mask)
    assert f_half["fill_ratio"] == f_full["fill_ratio"]  # black hole ignored, same visible content


def test_reference_is_a_compliant_labelled_frame_never_the_frame_itself(tmp_path, monkeypatch):
    for n, c in (("a", (30, 120, 200)), ("b", (200, 40, 40))):
        _warp(tmp_path / f"{n}.png", c)
    index = {"c1": {f"{n}.jpg": {"c1_1": {"warp": str(tmp_path / f"{n}.png"), "visible_fraction": 1.0}} for n in "ab"}}
    labels = tmp_path / "dev.json"
    labels.write_text(json.dumps({"c1": {"a.jpg": {"c1_1": {"planogram_compliant": True}}}}))
    monkeypatch.setattr(features, "LABELS", labels)
    (tmp_path / "idx.json").write_text(json.dumps(index))
    features.run(tmp_path / "idx.json", tmp_path / "out.json")
    out = json.loads((tmp_path / "out.json").read_text())["c1"]
    assert out["a.jpg"]["c1_1"]["hist_dist_ref"] == 0.0  # its own label is not a reference for itself
    assert out["b.jpg"]["c1_1"]["hist_dist_ref"] > 0.5  # different colour vs the compliant reference
