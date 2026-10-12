"""Owner B. Merge labels/<name>.json files into labels/dev.json.

Merge rule: per area the majority of labellers wins, a tie takes the lowest stock level (cheapest in the score).
Use --agreement to report areas where labellers disagree instead of merging.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

LABELS_DIR = Path(__file__).resolve().parents[2] / "labels"
DEV_JSON = LABELS_DIR / "dev.json"
_STOCK = {"empty": 0.0, "half": 0.5, "full": 1.0}


def _load_label_files(labels_dir: Path, only: list[str] | None = None) -> dict[str, dict]:
    """Return {labeller_name: data} for every <name>.json except dev.json (only the named ones when `only` is given)."""
    result = {}
    for p in sorted(labels_dir.glob("*.json")):
        if p.name == "dev.json" or (only is not None and p.stem not in only):
            continue
        try:
            result[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"warning: skipping {p.name}: {e}", file=sys.stderr)
    return result


def _resolve(areas: list[dict]) -> dict:
    """Majority stock level and compliance; ties take the lowest stock and non-compliant."""
    stock = Counter(a.get("stock_value", _STOCK.get(a.get("stock_level"))) for a in areas)
    comp = Counter(a.get("planogram_compliant") for a in areas)
    best_stock = min(v for v, n in stock.items() if n == max(stock.values()))
    best_comp = min((c for c, n in comp.items() if n == max(comp.values())), key=lambda c: c is not False)
    pick = next(a for a in areas if a.get("stock_value", _STOCK.get(a.get("stock_level"))) == best_stock)
    return {**pick, "planogram_compliant": best_comp, "stock_value": best_stock}


def merge(labels_dir: Path = LABELS_DIR, out_path: Path = DEV_JSON, only: list[str] | None = None) -> dict:
    """Merge all labeller files (or only the named ones) into dev.json; return the merged dict."""
    labellers = _load_label_files(labels_dir, only)
    if not labellers:
        print("No label files found.", file=sys.stderr)
        merged: dict = {}
    else:
        votes: dict[tuple, list[dict]] = {}
        for data in labellers.values():
            for cam_id, cam in data.items():
                for filename, frame in cam.items():
                    for area_id, area in frame.items():
                        votes.setdefault((cam_id, filename, area_id), []).append(area)
        merged = {}
        for (cam_id, filename, area_id), areas in votes.items():
            merged.setdefault(cam_id, {}).setdefault(filename, {})[area_id] = _resolve(areas)

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
    ap.add_argument("--only", help="comma-separated label file names to merge, e.g. andre,sensei_validation")
    args = ap.parse_args()

    if args.agreement:
        agreement(args.labels_dir)
    else:
        merge(args.labels_dir, args.out, args.only.split(",") if args.only else None)
