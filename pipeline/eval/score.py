"""Owner C. Score submission.json against labels with the official rules.

Per area:
  planogram_compliant: correct +1, wrong -2
  stock_value:         correct +1, predicted below truth -2, predicted above truth -3
  area in labels but missing from prediction: -2 and -3
"""
import argparse
from pathlib import Path

from pipeline.common.config import LABELS, SUBMISSION


def score(pred: dict, gt: dict) -> dict:
    """TODO: return {total, max, breakdown per case}. Both dicts: {camera_id: {filename: {area_id: {...}}}}."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a submission against labels.")
    ap.add_argument("--pred", type=Path, default=SUBMISSION)
    ap.add_argument("--gt", type=Path, default=LABELS)
    args = ap.parse_args()
    # TODO: load both, print score(pred, gt).
    raise NotImplementedError
