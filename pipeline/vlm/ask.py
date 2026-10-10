"""Ask a VLM (Gemini, key in .env as GEMINI_API_KEY) what is in each crop. Reads crops_index.json, writes vlm.json."""
import argparse
from pathlib import Path

from pipeline.common.config import CROPS_INDEX, VLM


def ask_area(crop_path: Path, product_name: str) -> dict:
    """TODO: ask 'is this <product_name>? empty/half/full?' and return the parsed answer. Cache responses."""
    raise NotImplementedError


def run(index_path: Path, out_path: Path) -> None:
    """TODO: ask_area for every crop in the index, write vlm.json."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    ap.add_argument("--out", type=Path, default=VLM)
    args = ap.parse_args()
    run(args.index, args.out)
