"""Spec for pipeline/crop/crop.py: warp geometry, off-image handling, index merge on --append."""
import json

import cv2
import numpy as np

from pipeline.crop.crop import context_crop, run, warp_area

BLUE, RED = (255, 0, 0), (0, 0, 255)


def frame() -> np.ndarray:
    """1280x720 BGR frame, blue on the left half and red on the right."""
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :640], img[:, 640:] = BLUE, RED
    return img


def test_warp_keeps_orientation_and_is_fully_visible():
    warp, visible = warp_area(frame(), [(540, 100), (740, 100), (740, 300), (540, 300)], size=64)
    assert warp.shape == (64, 64, 4)
    assert visible == 1.0
    assert tuple(warp[32, 8, :3]) == BLUE and tuple(warp[32, 56, :3]) == RED


def test_warp_masks_the_part_above_the_image():
    # Top half of the quad is off-image (y < 0).
    warp, visible = warp_area(frame(), [(100, -100), (300, -100), (300, 100), (100, 100)], size=64)
    assert abs(visible - 0.5) < 0.02
    alpha = warp[..., 3]
    assert not alpha[:30].any() and alpha[34:].all()
    assert not warp[..., :3][alpha == 0].any()


def test_context_crop_pads_and_fills_off_image():
    ctx = context_crop(frame(), [(100, -100), (300, -100), (300, 100), (100, 100)], pad=0.25)
    assert ctx.shape == (300, 300, 3)  # 200 px box + 50 px per side
    assert tuple(ctx[10, 10]) == (114, 114, 114)  # above the image: fill
    assert tuple(ctx[290, 10]) == BLUE  # inside the image, outside the quad outline


def write_planogram(path):
    quad = {"top_left": [10, 10], "top_right": [110, 10], "bottom_right": [110, 110], "bottom_left": [10, 110]}
    area = {"area_id": "6215_1", "product_id": "1", "product_name": "BANANA KG", "image_coordinates": quad}
    path.write_text(json.dumps({"Frutas e Legumes": {"cameras": {"6215": {"products": [area]}}}}))


def test_append_keeps_earlier_frames_and_run_frames_lists_only_the_new(tmp_path):
    planogram = tmp_path / "planogram.json"
    write_planogram(planogram)
    for name, folder in (("1.jpg", "train"), ("2.jpg", "val")):
        (tmp_path / folder / "6215").mkdir(parents=True)
        cv2.imwrite(str(tmp_path / folder / "6215" / name), frame())
    paths = dict(planogram_path=planogram, out_dir=tmp_path / "crops", index_path=tmp_path / "index.json",
                 run_frames_path=tmp_path / "run_frames.json")

    run(tmp_path / "train", **paths)
    run(tmp_path / "val", **paths, append=True)

    index = json.loads(paths["index_path"].read_text())
    assert set(index["6215"]) == {"1.jpg", "2.jpg"}
    assert json.loads(paths["run_frames_path"].read_text()) == [["6215", "2.jpg"]]
    entry = index["6215"]["2.jpg"]["6215_1"]
    assert entry["quad"] == [[10, 10], [110, 10], [110, 110], [10, 110]] and entry["visible_fraction"] == 1.0
    assert cv2.imread(entry["warp"], cv2.IMREAD_UNCHANGED).shape == (256, 256, 4)

    run(tmp_path / "val", **paths)  # without --append the index is rebuilt
    assert set(json.loads(paths["index_path"].read_text())["6215"]) == {"2.jpg"}
