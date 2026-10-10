"""Owner B. Merge labels/<name>.json files into labels/dev.json.

Merge rule: files are processed in alphabetical order; last write wins per area.
Use --agreement to report areas where labellers disagree instead of merging.
"""
import argparse
import json
import sys
from pathlib import Path

LABELS_DIR = Path(__file__).resolve().parents[2] / "labels"
DEV_JSON = LABELS_DIR / "dev.json"
_STOCK = {"empty": 0.0, "half": 0.5, "full": 1.0}


def _load_label_files(labels_dir: Path) -> dict[str, dict]:
    """Return {labeller_name: data} for every <name>.json except dev.json."""
    result = {}
    for p in sorted(labels_dir.glob("*.json")):
        if p.name == "dev.json":
            continue
        try:
            result[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"warning: skipping {p.name}: {e}", file=sys.stderr)
    return result


def merge(labels_dir: Path = LABELS_DIR, out_path: Path = DEV_JSON) -> dict:
    """Merge all labeller files into dev.json; return the merged dict."""
    labellers = _load_label_files(labels_dir)
    if not labellers:
        print("No label files found.", file=sys.stderr)
        merged: dict = {}
    else:
        merged = {}
        for _name, data in labellers.items():
            for cam_id, cam in data.items():
                for filename, frame in cam.items():
                    for area_id, area in frame.items():
                        merged.setdefault(cam_id, {}).setdefault(filename, {})[area_id] = area

    out_path.write_text(json.dumps(merged, indent=2))
    total = sum(
        len(areas)
        for cam in merged.values()
        for areas in cam.values()
    )
    print(f"Merged {len(labellers)} labeller(s), {total} areas -> {out_path}")
    return merged


def agreement(labels_dir: Path = LABELS_DIR) -> dict:
    """Report areas with disagreements between labellers. Returns {area_key: {labeller: value}}."""
    labellers = _load_label_files(labels_dir)
    if len(labellers) < 2:
        print("Need at least 2 labellers for agreement report.")
        return {}

    all_keys: dict[tuple, dict] = {}
    for name, data in labellers.items():
        for cam_id, cam in data.items():
            for filename, frame in cam.items():
                for area_id, area in frame.items():
                    key = (cam_id, filename, area_id)
                    all_keys.setdefault(key, {})[name] = area

    disagreements = {}
    for key, votes in all_keys.items():
        if len(votes) < 2:
            continue
        stock_vals = {n: v.get("stock_level") for n, v in votes.items()}
        compliant_vals = {n: v.get("planogram_compliant") for n, v in votes.items()}
        stock_set = set(v for v in stock_vals.values() if v is not None)
        compliant_set = set(v for v in compliant_vals.values() if v is not None)
        if len(stock_set) > 1 or len(compliant_set) > 1:
            disagreements["/".join(key)] = {
                "stock": stock_vals,
                "planogram_compliant": compliant_vals,
            }

    n = len(disagreements)
    total_shared = sum(1 for v in all_keys.values() if len(v) >= 2)
    pct = 100 * n / total_shared if total_shared else 0
    print(f"Disagreements: {n} / {total_shared} shared areas ({pct:.1f}%)")
    for area_key, diff in list(disagreements.items())[:20]:
        print(f"  {area_key}: stock={diff['stock']} compliant={diff['planogram_compliant']}")
    if n > 20:
        print(f"  ... ({n - 20} more)")
    return disagreements


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    ap.add_argument("--out", type=Path, default=DEV_JSON)
    ap.add_argument("--agreement", action="store_true", help="Report disagreements instead of merging")
    args = ap.parse_args()

    if args.agreement:
        agreement(args.labels_dir)
    else:
        merge(args.labels_dir, args.out)
