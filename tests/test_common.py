"""Spec for pipeline/common: planogram loading, the loose/package rule, image discovery."""
import cv2
import numpy as np
import pytest

from pipeline.common.images import list_images, load_image
from pipeline.common.planogram import area_polygon, is_packaged, load_planogram


def test_planogram_has_14_cameras_and_152_areas():
    planogram = load_planogram()
    assert len(planogram) == 14
    assert sum(len(areas) for areas in planogram.values()) == 152


def test_polygon_order_is_tl_tr_br_bl_and_keeps_negative_coordinates():
    area = next(a for a in load_planogram()["6383"] if a["area_id"] == "6383_5")
    assert area_polygon(area) == [(531, -69), (789, -67), (785, 118), (541, 114)]


@pytest.mark.parametrize("name, packaged", [
    ("BANANA CAT I CNT KG", False),
    ("MACA FUJI CNT CAL. 73/85 KG IMP", False),  # 73/85 is a calibre, not a weight
    ("ABÓBORA HOKAIDO R0 KG", False),
    ("KIWI ZESPRI 500 GR", True),
    ("BATATA BRANCA CNT EMB. 3KG", True),
    ("MAÇA GOLDEN NACIONAL CAL. 65/70 1 KG", True),
    ("CITR ALGARVE IGP LARANJA 3/4/5 1.5 KG", True),
])
def test_is_packaged(name, packaged):
    assert is_packaged(name) is packaged


def test_list_images_reads_camera_folders_and_skips_zip_debris(tmp_path):
    img = np.zeros((72, 128, 3), np.uint8)
    for rel in ("unzipped/6215/2.jpg", "unzipped/6215/1.jpg", "unzipped/6371/1.jpg",
                "unzipped/6215/._1.jpg", "__MACOSX/6215/1.jpg"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / rel), img)
    (tmp_path / "unzipped/6215/notes.txt").write_text("x")

    assert list_images(tmp_path) == [("6215", "1.jpg"), ("6215", "2.jpg"), ("6371", "1.jpg")]
    assert load_image("6215", "2.jpg", tmp_path).shape == (72, 128, 3)
