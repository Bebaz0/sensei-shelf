"""Owner V. Ask a vision model about every area crop: stock level and whether the product is the expected one.

Reads crops_index.json and run_frames.json, writes vlm.json (see CONTRACTS.md). One call per frame covers
all its areas (--per-area: one call per area, for comparison). Every answer is cached on disk, so a rerun
only pays for new crops or a new PROMPT_VERSION.
"""
import argparse
import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pipeline.common.config import (
    CROPS_INDEX, PRODUCTS_EN, ROOT, RUN_FRAMES, VLM, VLM_CACHE, VLM_RETRIES, VLM_RPM, VLM_WORKERS,
)
from pipeline.vlm import gemini
from pipeline.vlm.prompt import PROMPT_VERSION, SYSTEM, clean_frame_answer, frame_parts, frame_schema

PROVIDERS = ("gemini", "claude")  # keys of every area entry in vlm.json; claude is not wired yet
NO_RETRY_CODES = (400, 401, 403, 404)  # bad request or bad key: trying again cannot help


def cache_key(provider: str, model: str, parts: list[str | bytes], schema: dict) -> str:
    """sha256 over everything that determines the answer (so an edited prompt never reuses an old answer).

    model includes every provider setting that changes the answer (e.g. the image resolution tier).
    """
    fixed = (provider, model, PROMPT_VERSION, SYSTEM, json.dumps(schema, sort_keys=True))
    digest = hashlib.sha256()
    for part in (*fixed, *parts):
        digest.update(hashlib.sha256(part.encode() if isinstance(part, str) else part).digest())
    return digest.hexdigest()


def make_throttle(per_minute: int):
    """Return wait(): blocks so that calls start at most per_minute times a minute, across threads. 0 = no limit."""
    lock = threading.Lock()
    next_start = [0.0]

    def wait() -> None:
        if not per_minute:
            return
        with lock:
            start = max(time.monotonic(), next_start[0])
            next_start[0] = start + 60 / per_minute
        time.sleep(max(0.0, start - time.monotonic()))

    return wait


def ask_job(
    areas: list[tuple[str, dict]], candidates: dict[str, str], products: dict, call, provider: str, model: str,
    cache_dir: Path, throttle=lambda: None,
) -> dict:
    """Ask one provider about some areas of one frame in a single call.

    areas is [(area_id, crops_index entry)]; candidates is {product_id: name_pt} for its camera.
    Returns {"answers": {area_id: dict | None}, "usage": dict, "cached": bool}. throttle() is called before
    every API attempt. Never raises: after VLM_RETRIES failed attempts every answer is None. An area the model
    skipped or answered badly is None, the other areas are kept.
    """
    area_ids = [area_id for area_id, _ in areas]
    failed = {"answers": dict.fromkeys(area_ids), "usage": {}, "cached": False}
    try:
        crops = [(area_id, entry["product_id"], (ROOT / entry["ctx"]).read_bytes()) for area_id, entry in areas]
        parts = frame_parts(crops, candidates, products)
        schema = frame_schema(area_ids, list(candidates))
        cache_file = cache_dir / f"{cache_key(provider, model, parts, schema)}.json"
        if cache_file.exists():
            return {**json.loads(cache_file.read_text()), "cached": True}
    except Exception as error:
        print(f"WARNING cannot prepare the question for {areas[0][1].get('ctx')}: {error!r}")
        return failed

    for attempt in range(VLM_RETRIES):
        try:
            throttle()
            raw, usage = call(parts, SYSTEM, schema)
            answers = clean_frame_answer(raw, area_ids, list(candidates))
            if all(answer is None for answer in answers.values()):
                raise ValueError(f"no valid area in the answer: {str(raw)[:200]}")
            result = {"answers": answers, "usage": usage}
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({**result, "provider": provider, "model": model, "prompt_version": PROMPT_VERSION}))
            return {**result, "cached": False}
        except Exception as error:
            code = getattr(error, "code", None) or getattr(error, "status_code", None)
            if code in NO_RETRY_CODES or attempt == VLM_RETRIES - 1:
                print(f"WARNING {provider} gave no answer for {areas[0][1]['ctx']}: {error!r}"[:300])
                break
            time.sleep(4 * 2 ** attempt + random.random())  # 4, 8, 16, 32 s plus jitter (rate limits, 5xx)
    return failed


def read_area_list(path: Path) -> set[tuple[str, str, str]]:
    """Areas to keep, one `camera_id filename area_id` per line, # starts a comment (e.g. labels/calibration.txt)."""
    lines = (line.split("#")[0].split() for line in path.read_text(encoding="utf-8").splitlines())
    return {tuple(parts) for parts in lines if len(parts) == 3}


def run(
    index_path: Path, run_frames_path: Path, out_path: Path, cache_dir: Path = VLM_CACHE, products_path: Path = PRODUCTS_EN,
    no_api: bool = False, all_frames: bool = False, areas_path: Path | None = None, limit: int | None = None, call=None,
    per_area: bool = False,
) -> dict:
    """Write vlm.json for the frames of the current run (all_frames=True: every frame in the index).

    no_api=True, a missing API key or a failed call give null for that provider; the area is still written.
    areas_path and limit (first N calls) restrict the run (prompt testing). call overrides the provider (tests).
    per_area=True makes one call per area instead of one per frame. Returns the run summary (see the end).
    """
    index = json.loads(index_path.read_text())
    products = json.loads(products_path.read_text(encoding="utf-8")) if products_path.exists() else {}
    if all_frames or not run_frames_path.exists():
        frames = [(camera_id, filename) for camera_id, files in index.items() for filename in files]
    else:
        frames = [tuple(frame) for frame in json.loads(run_frames_path.read_text())]
    keep = read_area_list(areas_path) if areas_path else None

    jobs = []  # (camera_id, filename, [(area_id, entry)], candidates): one API call each
    for camera_id, filename in sorted(frames):
        areas = index.get(camera_id, {}).get(filename, {})  # a frame whose crop failed has no entry
        candidates = {entry["product_id"]: entry["product_name"] for entry in areas.values()}
        kept = [(a, e) for a, e in areas.items() if keep is None or (camera_id, filename, a) in keep]
        for group in ([[area] for area in kept] if per_area else [kept] if kept else []):
            jobs.append((camera_id, filename, group, candidates))
    jobs = jobs[:limit]

    if call is None and not no_api:
        call = gemini.make_call()
        if call is None:
            print("WARNING no GEMINI_API_KEY in .env: writing null answers")
    if no_api or call is None:
        results = [{"answers": {a: None for a, _ in job[2]}, "usage": {}, "cached": False} for job in jobs]
    else:
        throttle = make_throttle(VLM_RPM)

        def ask(job: tuple) -> dict:
            return ask_job(job[2], job[3], products, call, gemini.PROVIDER, gemini.SETTINGS, cache_dir, throttle)

        with ThreadPoolExecutor(VLM_WORKERS) as pool:
            results = []
            for result in pool.map(ask, jobs):
                results.append(result)
                if len(results) % 20 == 0:
                    print(f"  {len(results)}/{len(jobs)} calls")

    out: dict = {}
    for (camera_id, filename, _, _), result in zip(jobs, results):
        for area_id, answer in result["answers"].items():
            answers = dict.fromkeys(PROVIDERS)
            answers[gemini.PROVIDER] = answer
            out.setdefault(camera_id, {}).setdefault(filename, {})[area_id] = answers
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))

    fresh = [r for r in results if not r["cached"] and r["usage"]]
    summary = {
        "calls": len(jobs),
        "fresh_calls": len(fresh),
        "areas": sum(len(r["answers"]) for r in results),
        "answered": sum(a is not None for r in results for a in r["answers"].values()),
        "fresh_areas": sum(len(r["answers"]) for r in fresh),
        "tokens_in": sum(r["usage"].get("input_tokens", 0) for r in fresh),
        "tokens_out": sum(r["usage"].get("output_tokens", 0) for r in fresh),
    }
    print(f"{gemini.PROVIDER} {gemini.SETTINGS} prompt {PROMPT_VERSION}: {summary['answered']}/{summary['areas']} areas "
          f"answered in {len(jobs)} calls ({len(jobs) - len(fresh)} from cache or failed) -> {out_path}")
    per_area_in = summary["tokens_in"] / max(1, summary["fresh_areas"])
    print(f"new API calls: {len(fresh)}, tokens in {summary['tokens_in']} ({per_area_in:.0f} per area), "
          f"out {summary['tokens_out']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=CROPS_INDEX)
    ap.add_argument("--run-frames", type=Path, default=RUN_FRAMES)
    ap.add_argument("--out", type=Path, default=VLM)
    ap.add_argument("--cache-dir", type=Path, default=VLM_CACHE)
    ap.add_argument("--products", type=Path, default=PRODUCTS_EN)
    ap.add_argument("--no-api", action="store_true", help="make no calls: write null for every provider")
    ap.add_argument("--all", action="store_true", help="every frame in the index, not only run_frames.json")
    ap.add_argument("--areas", type=Path, help="only the areas listed in this file (format of labels/calibration.txt)")
    ap.add_argument("--limit", type=int, help="only the first N calls")
    ap.add_argument("--per-area", action="store_true", help="one call per area instead of one per frame")
    args = ap.parse_args()
    run(args.index, args.run_frames, args.out, args.cache_dir, args.products,
        no_api=args.no_api, all_frames=args.all, areas_path=args.areas, limit=args.limit, per_area=args.per_area)
