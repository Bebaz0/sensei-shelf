"""Check a submission before handing it in: every planogram area of every image, only the official keys and values.

  python -m pipeline.eval.check_submission --images <validation dir>   # exit 1 and a list of problems if it is not valid
"""
import argparse
import json
import sys
from pathlib import Path

from pipeline.common.config import PLANOGRAM, STOCK_VALUES, SUBMISSION
from pipeline.common.images import list_images
from pipeline.common.planogram import load_planogram

KEYS = {"planogram_compliant", "stock_level", "stock_value"}


def problems(submission: dict, frames: list[tuple[str, str]], planogram: dict) -> list[str]:
    """Empty list when the submission covers exactly the given frames with valid answers for every area."""
    out = []
    expected = {(c, f) for c, f in frames}
    got = {(c, f) for c, fs in submission.items() for f in fs}
    out += [f"missing image {c}/{f}" for c, f in sorted(expected - got)]
    out += [f"extra image {c}/{f}" for c, f in sorted(got - expected)]
    for c, f in sorted(expected & got):
        areas = submission[c][f]
        want = {a["area_id"] for a in planogram.get(c, [])}
        out += [f"missing area {c}/{f}/{a}" for a in sorted(want - set(areas))]
        out += [f"extra area {c}/{f}/{a}" for a in sorted(set(areas) - want)]
        for a, pred in areas.items():
            if not isinstance(pred, dict) or set(pred) != KEYS:
                out.append(f"bad keys {c}/{f}/{a}: {sorted(pred) if isinstance(pred, dict) else pred}")
            elif not isinstance(pred["planogram_compliant"], bool) or pred["stock_level"] not in STOCK_VALUES \
                    or pred["stock_value"] != STOCK_VALUES[pred["stock_level"]]:
                out.append(f"bad values {c}/{f}/{a}: {pred}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="the folder the submission was made from")
    ap.add_argument("--submission", type=Path, default=SUBMISSION)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    args = ap.parse_args()
    frames, planogram = list_images(args.images), load_planogram(args.planogram)
    found = problems(json.loads(args.submission.read_text()), frames, planogram)
    n = sum(len(planogram.get(c, [])) for c, _ in frames)
    print("\n".join(found[:50]) or f"OK: {len(frames)} images, {n} areas, {args.submission}")
    sys.exit(1 if found else 0)
