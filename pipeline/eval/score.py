"""Owner B. Score submission.json against labels with the official rules.

Per area:
  planogram_compliant: correct +1, wrong -2
  stock_value:         correct +1, predicted below truth -2, predicted above truth -3
  area in labels but missing from prediction: -2 and -3
"""
import argparse
import json
from pathlib import Path

from pipeline.common.config import LABELS, SUBMISSION


def score(pred: dict, gt: dict) -> dict:
    """Return {total, max, breakdown} scoring pred against gt.

    Iterates only over gt entries; ignores pred entries not in gt.
    Both dicts: {camera_id: {filename: {area_id: {planogram_compliant, stock_value, ...}}}}.
    """
    total = 0
    max_score = 0
    breakdown: dict = {}

    for cam_id, cam_gt in gt.items():
        for filename, frame_gt in cam_gt.items():
            for area_id, area_gt in frame_gt.items():
                max_score += 2

                area_pred = pred.get(cam_id, {}).get(filename, {}).get(area_id)

                # A prediction without a field counts as wrong, never as a default answer.
                if area_pred is None:
                    p_pts, s_pts = -2, -3
                else:
                    gt_compliant = area_gt.get("planogram_compliant", True)
                    pred_compliant = area_pred.get("planogram_compliant")
                    p_pts = 1 if pred_compliant == gt_compliant else -2

                    gt_val = float(area_gt.get("stock_value", 0.5))
                    pred_val = area_pred.get("stock_value")
                    if pred_val is None:
                        s_pts = -3
                    elif float(pred_val) == gt_val:
                        s_pts = 1
                    elif float(pred_val) < gt_val:
                        s_pts = -2
                    else:
                        s_pts = -3

                total += p_pts + s_pts
                breakdown.setdefault(cam_id, {}).setdefault(filename, {})[area_id] = {
                    "planogram": p_pts,
                    "stock": s_pts,
                }

    return {"total": total, "max": max_score, "breakdown": breakdown}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score a submission against labels.")
    ap.add_argument("--pred", type=Path, default=SUBMISSION)
    ap.add_argument("--gt", type=Path, default=LABELS)
    args = ap.parse_args()

    pred = json.loads(args.pred.read_text())
    gt = json.loads(args.gt.read_text())
    result = score(pred, gt)
    print(f"Score: {result['total']} / {result['max']}")
