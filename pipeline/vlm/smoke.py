"""Owner V. Live smoke test: one real Gemini call on one labelled frame, then checks the answers and the billed tokens.

Uses a fresh temporary cache, so the call is always made: 1 request, about 7k tokens in, well under 1 cent.
Run it after every change to gemini.py, prompt.py or the model id. Without GEMINI_API_KEY it prints a note and
exits 0. Needs out/crops_index.json (run the crop stage first).

  python -m pipeline.vlm.smoke
"""
import json
import tempfile
from pathlib import Path

from pipeline.common.config import CROPS_INDEX, ROOT
from pipeline.vlm import ask, gemini

CAMERA, FILENAME = "6215", "1751001268.jpg"  # first frame of 6215, labelled in labels/tomas.json
MAX_TOKENS_PER_AREA = 900  # about 620 expected; 1,600 or more means media_resolution was ignored
PRICE_PER_M = (0.25, 1.50)  # USD per million tokens in, out (gemini-3.1-flash-lite preview list price, an upper bound)
LABELS = ROOT / "labels" / "tomas.json"


def main() -> None:
    if gemini.make_call() is None:
        print("no GEMINI_API_KEY in .env: smoke test skipped")
        return
    area_ids = list(json.loads(CROPS_INDEX.read_text())[CAMERA][FILENAME])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "areas.txt").write_text("".join(f"{CAMERA} {FILENAME} {a}\n" for a in area_ids))
        summary = ask.run(CROPS_INDEX, tmp / "no_run_frames.json", tmp / "vlm.json", cache_dir=tmp / "cache",
                          areas_path=tmp / "areas.txt")
        answers = json.loads((tmp / "vlm.json").read_text())[CAMERA][FILENAME]

    labels = json.loads(LABELS.read_text()).get(CAMERA, {}).get(FILENAME, {}) if LABELS.exists() else {}
    for area_id in area_ids:
        got, label = answers[area_id][gemini.PROVIDER], labels.get(area_id)
        seen = f"{got['stock']:5} match={got['matches_expected']!s:5} {got['observed']:8}" if got else "null"
        truth = f"label {label['stock_level']:5} compliant={label['planogram_compliant']}" if label else ""
        print(f"  {area_id:8} {seen}  {truth}")

    cost = (summary["tokens_in"] * PRICE_PER_M[0] + summary["tokens_out"] * PRICE_PER_M[1]) / 1e6
    per_area = summary["tokens_in"] / max(1, summary["fresh_areas"])
    print(f"cost at most {cost:.5f} USD; {per_area:.0f} input tokens per area (limit {MAX_TOKENS_PER_AREA})")
    assert summary["fresh_calls"] == 1, f"expected 1 real call, got {summary['fresh_calls']}"
    assert summary["answered"] == len(area_ids), f"{len(area_ids) - summary['answered']} areas without an answer"
    assert per_area < MAX_TOKENS_PER_AREA, "too many input tokens per area: is media_resolution applied?"
    print("smoke test passed")


if __name__ == "__main__":
    main()
