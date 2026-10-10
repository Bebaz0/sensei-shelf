"""Copy a small, committable demo run into dashboard/public/data/.

Reads out/results.json, keeps FRAMES_PER_CAM evenly spaced frames per camera (always including the
last), writes JPG thumbnails (<=320 px) of each crop and a <=960 px JPG of each kept full frame, and
rewrites crop paths so the dashboard can fetch them. Usage: python scripts/sync_dashboard.py [N]
"""
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.common.config import IMAGES_DIR, RESULTS  # noqa: E402

DEST = ROOT / "dashboard" / "public" / "data"


def pick(files: list[str], n: int) -> list[str]:
    files = sorted(files)
    if len(files) <= n:
        return files
    idx = {round(i * (len(files) - 1) / (n - 1)) for i in range(n)} if n > 1 else {len(files) - 1}
    return [files[i] for i in sorted(idx)]


def save_jpg(src: Path, dst: Path, max_side: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img.thumbnail((max_side, max_side))
    img.save(dst, "JPEG", quality=80)


def main(n: int) -> None:
    results = json.loads(Path(RESULTS).read_text())
    shutil.rmtree(DEST, ignore_errors=True)
    kept: dict = {}
    for cam, files in results.items():
        for fname in pick(list(files), n):
            kept.setdefault(cam, {})[fname] = files[fname]
            save_jpg(Path(IMAGES_DIR) / cam / fname, DEST / "frames" / cam / fname, 960)
            for area_id, a in kept[cam][fname].items():
                src = ROOT / a["crop"]
                rel = Path("crops") / cam / (src.stem + ".jpg")
                save_jpg(src, DEST / rel, 320)
                a["crop"] = f"data/{rel.as_posix()}"
    (DEST / "results.json").write_text(json.dumps(kept))
    (DEST / ".gitkeep").touch()
    print(f"synced {sum(len(v) for v in kept.values())} frames to {DEST}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
