"""Smoke tests for the inspection API. Fixtures are minimal JSONs written to tmp_path."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PLANOGRAM = {
    "Frutas e Legumes": {
        "cameras": {
            "6215": {
                "products": [
                    {
                        "area_id": "1",
                        "product_id": "p1",
                        "product_name": "LARANJAS",
                        "image_coordinates": {
                            "top_left": [0, 0], "top_right": [10, 0],
                            "bottom_right": [10, 10], "bottom_left": [0, 10],
                        },
                    },
                ],
            },
        },
    },
}
SUBMISSION = {"6215": {"f.jpg": {"1": {"planogram_compliant": True, "stock_level": "full", "stock_value": 1.0}}}}
LABELS = {"6215": {"f.jpg": {"1": {"planogram_compliant": True, "stock_level": "full", "stock_value": 1.0}}}}
RESULTS = {"6215": {"f.jpg": {"1": {
    "pred": {"planogram_compliant": True, "stock_level": "full", "stock_value": 1.0},
    "p_stock": {"empty": 0.01, "half": 0.1, "full": 0.89},
    "p_wrong": 0.05, "observed_product_id": None, "low_confidence": False,
    "reason": "knn: full; p_wrong 0.05", "crop": "out/crops/6215/f_6215_1_ctx.png",
}}}}
FEATURES = {"6215": {"f.jpg": {"1": {"fill_ratio": 0.42, "edge_density": 0.1}}}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI client with every artefact path redirected to tmp_path."""
    from PIL import Image

    from api import cache, dashboard, routes

    paths = {
        "PLANOGRAM": tmp_path / "planogram.json",
        "SUBMISSION": tmp_path / "submission.json",
        "LABELS": tmp_path / "dev.json",
        "RESULTS": tmp_path / "results.json",
        "FEATURES": tmp_path / "features.json",
    }
    paths["PLANOGRAM"].write_text(json.dumps(PLANOGRAM))
    paths["SUBMISSION"].write_text(json.dumps(SUBMISSION))
    paths["LABELS"].write_text(json.dumps(LABELS))
    paths["RESULTS"].write_text(json.dumps(RESULTS))
    paths["FEATURES"].write_text(json.dumps(FEATURES))

    for name, p in paths.items():
        monkeypatch.setattr(routes, name, p)
    monkeypatch.setattr(dashboard, "RESULTS", paths["RESULTS"])

    frames_dir = tmp_path / "frames"
    crops_dir = tmp_path / "crops"
    cache_dir = tmp_path / "api_cache"
    (frames_dir / "6215").mkdir(parents=True)
    (crops_dir / "6215").mkdir(parents=True)
    Image.new("RGB", (120, 80), (200, 100, 50)).save(frames_dir / "6215" / "f.jpg", "JPEG")
    Image.new("RGBA", (64, 64), (0, 200, 0, 255)).save(crops_dir / "6215" / "f_6215_1_ctx.png")

    monkeypatch.setattr(dashboard, "TRAIN_IMAGE_DIRS", (frames_dir,))
    monkeypatch.setattr(dashboard, "CROPS_DIR", crops_dir)
    monkeypatch.setattr(dashboard, "CACHE_DIR", cache_dir)

    cache._cache.clear()
    from api.main import app
    return TestClient(app), paths


def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_cameras(client):
    c, _ = client
    r = c.get("/api/cameras")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1 and data[0]["camera_id"] == "6215"
    assert data[0]["products"][0] == {"area_id": "1", "product_id": "p1", "product_name": "LARANJAS"}


def test_score(client):
    c, _ = client
    r = c.get("/api/score")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and body["max"] == 2


def test_score_missing_submission(client):
    c, paths = client
    paths["SUBMISSION"].unlink()
    r = c.get("/api/score")
    assert r.status_code == 404


def test_stats(client):
    c, _ = client
    r = c.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["n_frames"] == 1 and body["n_areas"] == 1
    assert body["stock"] == {"empty": 0, "half": 0, "full": 1}
    assert body["compliance"] == {"true": 1, "false": 0}
    assert body["n_low_confidence"] == 0 and body["n_fallback"] == 0


def test_area(client):
    c, _ = client
    r = c.get("/api/area/6215/f.jpg/1")
    assert r.status_code == 200
    body = r.json()
    assert body["product_name"] == "LARANJAS"
    assert body["result"]["pred"]["stock_level"] == "full"
    assert body["features"]["fill_ratio"] == 0.42
    assert body["label"]["stock_level"] == "full"


def test_area_unknown(client):
    c, _ = client
    r = c.get("/api/area/6215/f.jpg/999")
    assert r.status_code == 404


def test_data_results_rewrites_crops(client):
    c, _ = client
    r = c.get("/data/results.json")
    assert r.status_code == 200
    row = r.json()["6215"]["f.jpg"]["1"]
    assert row["crop"] == "/data/crops/6215/f_6215_1_ctx.jpg"
    assert row["pred"]["stock_level"] == "full"


def test_data_frames_streams(client):
    c, _ = client
    r = c.get("/data/frames/6215/f.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and len(r.content) > 100
    r2 = c.get("/data/frames/6215/f.jpg?max=64")
    assert r2.status_code == 200 and len(r2.content) > 100
    r3 = c.get("/data/frames/6215/missing.jpg")
    assert r3.status_code == 404


def test_data_crops_converts_png(client):
    c, _ = client
    r = c.get("/data/crops/6215/f_6215_1_ctx.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    r2 = c.get("/data/crops/6215/f_6215_1_ctx.jpg?max=32")
    assert r2.status_code == 200
    r3 = c.get("/data/crops/6215/missing.jpg")
    assert r3.status_code == 404
