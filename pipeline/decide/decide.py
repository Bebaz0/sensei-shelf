"""Owner C. Combine pixel features + VLM answers into per-area verdicts. Writes results.json (dashboard) and submission.json (scored)."""
import argparse
from pathlib import Path

from pipeline.common.config import FEATURES, PLANOGRAM, RESULTS, SUBMISSION, VLM


def decide_area(area: dict, features: dict | None, vlm: dict | None) -> dict:
    """TODO: return {planogram_compliant, stock_level, stock_value, ...}.
    Scoring is asymmetric: over-estimating stock costs -3, under -2, so lean low when unsure."""
    raise NotImplementedError


def run(planogram: Path, features_path: Path, vlm_path: Path, results_path: Path, submission_path: Path) -> None:
    """TODO: decide every area of every image; write results.json and submission.json. Never omit an area (missing = -5)."""
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planogram", type=Path, default=PLANOGRAM)
    ap.add_argument("--features", type=Path, default=FEATURES)
    ap.add_argument("--vlm", type=Path, default=VLM)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--submission", type=Path, default=SUBMISSION)
    args = ap.parse_args()
    run(args.planogram, args.features, args.vlm, args.results, args.submission)
