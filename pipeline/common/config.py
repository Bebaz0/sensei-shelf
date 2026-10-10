"""Shared paths and constants. Every module imports from here; nobody hardcodes paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
PLANOGRAM = DATA / "planogram.json"
PRODUCTS_EN = DATA / "products_en.json"
# <camera_id>/<unix_ts>.jpg: train and test are mixed, there are no split folders.
IMAGES_DIR = DATA / "Frutas e Legumes (train & test)"

OUT = ROOT / "out"
RUN_FRAMES = OUT / "run_frames.json"
CROPS_DIR = OUT / "crops"
CROPS_INDEX = OUT / "crops_index.json"
QA_DIR = OUT / "qa"
CACHE_DIR = OUT / "cache"
FEATURES = OUT / "features.json"
VLM = OUT / "vlm.json"
RESULTS = OUT / "results.json"
SUBMISSION = OUT / "submission.json"

LABELS = ROOT / "labels" / "dev.json"
CALIBRATION = ROOT / "labels" / "calibration.txt"

STOCK_VALUES = {"empty": 0.0, "half": 0.5, "full": 1.0}

# Crop stage.
FRAME_SIZE = (1280, 720)  # (width, height) the planogram coordinates refer to
WARP_SIZE = 256  # side of the square warped crop, px
CTX_PAD = 0.25  # context crop padding, as a fraction of the quad's bounding box, per side
CTX_FILL = (114, 114, 114)  # BGR fill for context pixels that fall outside the camera image
CTX_OUTLINE = (255, 0, 255)  # BGR colour of the quad outline (magenta: no produce has it)

# Camera identification for images that are not inside a <camera_id>/ folder.
THUMB_SIZE = (128, 72)  # (width, height) of the edge thumbnail compared against each camera
# Below this correlation with the best camera the match is still used, but a warning is printed.
THUMB_MIN_CORR = 0.5

# VLM stage. Bump PROMPT_VERSION in pipeline/vlm/prompt.py when the prompt changes (it is part of the cache key).
VLM_CACHE = CACHE_DIR / "vlm"
GEMINI_MODEL = "gemini-3.1-flash-lite"  # pinned, see docs/DECISIONS.md
VLM_WORKERS = 16  # parallel API calls
VLM_RPM = 15  # requests per minute allowed by the API key (free tier: 15); 0 = no limit (paid key)
VLM_RETRIES = 5  # attempts per call, waiting 4, 8, 16, 32 s in between

# Decision stage.
EMPTY_IS_COMPLIANT = True  # pending mentor answer, see docs/DECISIONS.md
STOCK_BIAS = 0.0  # > 0 leans the stock decision lower

if __name__ == "__main__":
    import argparse

    argparse.ArgumentParser(description="Print resolved paths and constants.").parse_args()
    for k, v in dict(globals()).items():
        if k.isupper():
            print(f"{k:18} {v}")
