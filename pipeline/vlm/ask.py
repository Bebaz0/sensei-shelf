"""Owner V. Ask a vision model about every area crop: stock level and whether the product is the expected one.

Reads crops_index.json and run_frames.json, writes vlm.json (see CONTRACTS.md). Every answer is cached
on disk, so a rerun only pays for new crops or a new PROMPT_VERSION.
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
from pipeline.vlm.prompt import PROMPT_VERSION, PROMPT_VERSION_REF, SYSTEM, SYSTEM_REF, answer_schema, area_prompt, clean_answer

PROVIDERS = ("gemini", "claude")  # keys of every area entry in vlm.json; claude is not wired yet
NO_RETRY_CODES = (400, 401, 403, 404)  # bad request or bad key: trying again cannot help


def cache_key(provider: str, model: str, images: list[bytes], text: str, schema: dict, system: str = SYSTEM, version: str = PROMPT_VERSION) -> str:
    """sha256 over everything that determines the answer (so an edited prompt never reuses an old answer)."""
    parts = (provider, model, version, system, text, json.dumps(schema, sort_keys=True))
    digest = hashlib.sha256()
    for part in (*(p.encode() for p in parts), *images):
        digest.update(hashlib.sha256(part).digest())
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


LEVEL_ORDER = ("full", "half", "empty")


def exemplars_for(camera_id: str, filename: str, area_id: str, labels: dict, index: dict) -> list[tuple[str, bytes]]:
    """[(level, flattened crop PNG)] of this same area: per level the most recent labelled frame STRICTLY EARLIER than
    this one (rule 6), in LEVEL_ORDER. Empty when the area has no earlier labelled frame."""
    ts = lambda name: int(Path(name).stem)
    t, picks = ts(filename), {}
    for f, areas in labels.get(camera_id, {}).items():
        lab = areas.get(area_id)
        if not lab or lab.get("unsure") or ts(f) >= t or lab.get("stock_level") not in LEVEL_ORDER:
            continue
        crop = index.get(camera_id, {}).get(f, {}).get(area_id, {}).get("warp")
        if crop and (lab["stock_level"] not in picks or ts(f) > picks[lab["stock_level"]][0]):
            picks[lab["stock_level"]] = (ts(f), crop)
    return [(lv, (ROOT / picks[lv][1]).read_bytes()) for lv in LEVEL_ORDER if lv in picks]


def merge_votes(answers: list[dict]) -> dict:
    """One entry from several votes of the same area: stock by vote (ties go to the LOWER level), fill_pct mean,
    matches_expected by majority (ties: true), plus `votes`, the share of each stock level."""
    levels = ("empty", "half", "full")
    votes = {lv: sum(a["stock"] == lv for a in answers) / len(answers) for lv in levels}
    stock = max(levels, key=lambda lv: (votes[lv], -levels.index(lv)))
    first = next(a for a in answers if a["stock"] == stock)
    return {**first, "stock": stock, "votes": votes,
            "fill_pct": round(sum(a["fill_pct"] for a in answers) / len(answers)),
            "matches_expected": sum(a["matches_expected"] for a in answers) * 2 >= len(answers),
            "confidence": sum(a["confidence"] for a in answers) / len(answers)}


def ask_area(
    entry: dict, candidates: dict[str, str], products: dict, call, provider: str, model: str, cache_dir: Path,
    throttle=lambda: None, exemplars: list[tuple[str, bytes]] | None = None,
) -> dict:
    """Ask one provider about one area. Returns {"answer": dict | None, "usage": dict, "cached": bool}.

    entry is the area's crops_index entry; candidates is {product_id: name_pt} for its camera.
    throttle() is called before every API attempt. Never raises: after VLM_RETRIES failed attempts the answer is None.
    """
    try:
        images = [(ROOT / entry[kind]).read_bytes() for kind in ("ctx", "warp")]
        ref = exemplars is not None
        text = area_prompt(entry["product_id"], candidates, products, tuple(lv for lv, _ in exemplars or ()))
        schema = answer_schema(list(candidates), evidence=ref)
        images += [im for _, im in exemplars or ()]
        system, version = (SYSTEM_REF, PROMPT_VERSION_REF) if ref else (SYSTEM, PROMPT_VERSION)
        cache_file = cache_dir / f"{cache_key(provider, model, images, text, schema, system, version)}.json"
        if cache_file.exists():
            return {**json.loads(cache_file.read_text()), "cached": True}
    except Exception as error:
        print(f"WARNING cannot prepare the question for {entry.get('ctx')}: {error!r}")
        return {"answer": None, "usage": {}, "cached": False}

    for attempt in range(VLM_RETRIES):
        try:
            throttle()
            raw, usage = call(images, system, text, schema)
            result = {"answer": clean_answer(raw, list(candidates)), "usage": usage}
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({**result, "provider": provider, "model": model, "prompt_version": version}))
            return {**result, "cached": False}
        except Exception as error:
            code = getattr(error, "code", None) or getattr(error, "status_code", None)
            if code in NO_RETRY_CODES or attempt == VLM_RETRIES - 1:
                print(f"WARNING {provider} gave no answer for {entry['ctx']}: {error!r}"[:300])
                break
            time.sleep(4 * 2 ** attempt + random.random())  # 4, 8, 16, 32 s plus jitter (rate limits, 5xx)
    return {"answer": None, "usage": {}, "cached": False}


def read_area_list(path: Path) -> set[tuple[str, str, str]]:
    """Areas to keep, one `camera_id filename area_id` per line, # starts a comment (e.g. labels/calibration.txt)."""
    lines = (line.split("#")[0].split() for line in path.read_text(encoding="utf-8").splitlines())
    return {tuple(parts) for parts in lines if len(parts) == 3}


def run(
    index_path: Path, run_frames_path: Path, out_path: Path, cache_dir: Path = VLM_CACHE, products_path: Path = PRODUCTS_EN,
    no_api: bool = False, all_frames: bool = False, areas_path: Path | None = None, limit: int | None = None, call=None,
    exemplars_path: Path | None = None,
) -> None:
    """Write vlm.json for the frames of the current run (all_frames=True: every frame in the index).

    no_api=True, a missing API key or a failed call give null for that provider; the area is still written.
    areas_path and limit restrict the run to a few areas (prompt testing). call overrides the provider (tests).
    """
    index = json.loads(index_path.read_text())
    products = json.loads(products_path.read_text(encoding="utf-8")) if products_path.exists() else {}
    if all_frames or not run_frames_path.exists():
        frames = [(camera_id, filename) for camera_id, files in index.items() for filename in files]
    else:
        frames = [tuple(frame) for frame in json.loads(run_frames_path.read_text())]
    keep = read_area_list(areas_path) if areas_path else None

    jobs = []  # (camera_id, filename, area_id, entry, candidates)
    for camera_id, filename in sorted(frames):
        areas = index.get(camera_id, {}).get(filename, {})  # a frame whose crop failed has no entry
        candidates = {entry["product_id"]: entry["product_name"] for entry in areas.values()}
        for area_id, entry in areas.items():
            if keep is None or (camera_id, filename, area_id) in keep:
                jobs.append((camera_id, filename, area_id, entry, candidates))
    jobs = jobs[:limit]

    if call is None and not no_api:
        call = gemini.make_call()
        if call is None:
            print("WARNING no GEMINI_API_KEY in .env: writing null answers")
    if no_api or call is None:
        results = [{"answer": None, "usage": {}, "cached": False} for _ in jobs]
    else:
        throttle = make_throttle(VLM_RPM)

        labels = json.loads(exemplars_path.read_text()) if exemplars_path else None

        def ask(job: tuple) -> dict:
            if labels is None:
                return ask_area(job[3], job[4], products, call, gemini.PROVIDER, gemini.MODEL, cache_dir, throttle)
            ex = exemplars_for(job[0], job[1], job[2], labels, index)  # prompt v3: same-area labelled references
            tries = [ex] + ([ex[::-1]] if len(ex) > 1 else [])  # order swap = a second vote
            rs = [ask_area(job[3], job[4], products, call, gemini.PROVIDER, gemini.MODEL, cache_dir, throttle, e) for e in tries]
            good = [r for r in rs if r["answer"]]
            return {"answer": merge_votes([r["answer"] for r in good]) if good else None,
                    "usage": {k: sum(r["usage"].get(k, 0) for r in good) for k in ("input_tokens", "output_tokens")},
                    "cached": all(r["cached"] for r in good) if good else False}

        with ThreadPoolExecutor(VLM_WORKERS) as pool:
            results = []
            for result in pool.map(ask, jobs):
                results.append(result)
                if len(results) % 200 == 0:
                    print(f"  {len(results)}/{len(jobs)} areas")

    out: dict = {}
    for (camera_id, filename, area_id, _, _), result in zip(jobs, results):
        answers = dict.fromkeys(PROVIDERS)
        answers[gemini.PROVIDER] = result["answer"]
        out.setdefault(camera_id, {}).setdefault(filename, {})[area_id] = answers
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))

    answered = sum(r["answer"] is not None for r in results)
    fresh = [r for r in results if r["answer"] is not None and not r["cached"]]
    tokens_in = sum(r["usage"].get("input_tokens", 0) for r in fresh)
    tokens_out = sum(r["usage"].get("output_tokens", 0) for r in fresh)
    print(f"{gemini.PROVIDER} {gemini.MODEL} prompt {PROMPT_VERSION}: {answered}/{len(jobs)} areas answered "
          f"({answered - len(fresh)} from cache, {len(jobs) - answered} null) -> {out_path}")
    print(f"new API calls: {len(fresh)}, tokens in {tokens_in}, out {tokens_out}")


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
    ap.add_argument("--limit", type=int, help="only the first N areas")
    ap.add_argument("--exemplars", type=Path, help="labels json (labels/dev.json): prompt v3 with same-area labelled reference crops, 2 votes")
    args = ap.parse_args()
    run(args.index, args.run_frames, args.out, args.cache_dir, args.products,
        no_api=args.no_api, all_frames=args.all, areas_path=args.areas, limit=args.limit, exemplars_path=args.exemplars)
