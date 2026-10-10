# Data contracts

Every module talks to the next only through these files in `out/`. Change a contract = PR touching this file + tell the consumer. Owner of this file: C.

Pipeline: `crop` -> `pixel` + `vlm` -> `decide` -> `eval` / `dashboard`

All files are nested `{camera_id: {filename: {area_id: ...}}}` so they join directly with `submission.json` and `labels/dev.json`. Quads are `[x, y]` pixels in TL, TR, BR, BL order and may be negative (area partly off-image).

## run_frames.json
Producer: `pipeline/crop` (A), rewritten on every run. Consumers: `pixel`, `vlm`, `decide`.
```json
[["6215", "1751052241.jpg"], ["6371", "1751052250.jpg"]]
```
The frames of the current run. `pixel`, `vlm` and `decide` process only these frames (flag `--all` processes the whole index). `decide` takes the areas of every camera from `data/planogram.json` (always available) and the frames from this file, so it outputs every area of every frame even when crops, features or VLM answers are missing (per-area fallback). If this file is missing, `decide` lists the frames of its `--images <dir>` folder (default `IMAGES_DIR` from `config.py`) via `pipeline/common/images.py`; `run_all.sh` passes `--images "$IMAGES"` for validation runs.

## crops_index.json
Producer: `pipeline/crop` (A). Consumers: `pixel`, `vlm`, `decide`.
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "product_id": "11908", "product_name": "BANANA CAT I CNT KG",
  "quad": [[299,350],[464,347],[479,438],[331,438]],
  "visible_fraction": 1.0,
  "warp": "out/crops/6215/1751052241_6215_2_warp.png",
  "ctx": "out/crops/6215/1751052241_6215_2_ctx.png"
}}}}
```
`crop --images <dir> --append` merges new frames into the existing index and never deletes entries, so train crops stay available as the kNN and reference bank during a validation run. Without `--append` the index is rebuilt from the given folder. Crops are PNG. The warped crop is 256x256 with off-image pixels masked by `visible_fraction`; `ctx` is the axis-aligned crop padded 25% with the quad outline drawn.

## features.json
Producer: `pipeline/pixel` (B). Consumer: `decide`.
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "fill_ratio": 0.62, "hist_dist_ref": 0.18, "edge_density": 0.31, "visible_fraction": 1.0,
  "knn": {"empty": 0.0, "half": 0.7, "full": 0.3, "p_wrong": 0.05},
  "temporal": {"prev_stock": "full", "next_stock": null, "dt_prev_s": 3900, "dt_next_s": null}
}}}}
```
`knn` and `temporal` are present only when labels exist and never use the frame itself. `next_stock` and `dt_next_s` are `null` when there is no later frame (validation frames may be later than every train frame).

## vlm.json
Producer: `pipeline/vlm` (V). Consumer: `decide`.
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "gemini": {"stock": "half", "fill_pct": 55, "matches_expected": true,
             "observed": "11908", "confidence": 0.8, "reason": "bananas, about half full"},
  "claude": null
}}}}
```
`observed` is a candidate product_id, `other`, `empty` or `occluded`. Temperature 0, cached under `out/cache/vlm/`. A provider that was skipped (`--no-api`, outage) is `null`.

## results.json
Producer: `pipeline/decide` (C). Consumer: `dashboard` (D).
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "pred": {"planogram_compliant": true, "stock_level": "half", "stock_value": 0.5},
  "p_stock": {"empty": 0.1, "half": 0.6, "full": 0.3}, "p_wrong": 0.05,
  "observed_product_id": null, "low_confidence": false, "reason": "...",
  "crop": "out/crops/6215/1751052241_6215_2_ctx.png"
}}}}
```
`scripts/sync_dashboard.sh [N]` copies a small demo run into `dashboard/public/data/`: N frames per camera (default 3, evenly spaced, always the last), JPG thumbnails of crops (<=320 px) and of the kept full frames (<=960 px, at `data/frames/<cam>/<file>`), with `crop` paths rewritten to `data/crops/...`. The committed demo stays a few MB; never commit full-size PNG crops. Run it after a train run: it reads the full frames from `IMAGES_DIR`.

## submission.json
Producer: `pipeline/decide` (C). Consumer: `pipeline/eval/score.py`, organisers.

Fixed by the organisers, same as `annotations.json` and `labels/dev.json`. Must contain every area of every image.

```json
{"6215": {"1751052241.jpg": {"6215_2": {"planogram_compliant": true, "stock_level": "half", "stock_value": 0.5}}}}
```

`stock_level` in `empty | half | full` maps to `stock_value` 0 / 0.5 / 1. No extra keys.

## labels
`labels/<name>.json` (one per labeller) and the merged `labels/dev.json` use the submission format plus optional per area `observed_product_id` and `unsure`. The scorer ignores extra keys. Merge rule: last write wins per area; `merge_labels --agreement` reports disagreements.
