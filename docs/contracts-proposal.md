# Proposed schemas for the TODO sections of CONTRACTS.md

Owner C reviews and moves these into `CONTRACTS.md`. All files are nested `{camera_id: {filename: {area_id: ...}}}` so they join directly with `submission.json` and `labels/dev.json`. Quads are `[x, y]` pixels in TL, TR, BR, BL order and may be negative.

## crops_index.json (producer: crop)
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "product_id": "11908", "product_name": "BANANA CAT I CNT KG",
  "quad": [[299,350],[464,347],[479,438],[331,438]],
  "visible_fraction": 1.0,
  "warp": "out/crops/6215/1751052241_6215_2_warp.png",
  "ctx": "out/crops/6215/1751052241_6215_2_ctx.png"
}}}}
```

## features.json (producer: pixel)
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "fill_ratio": 0.62, "hist_dist_ref": 0.18, "edge_density": 0.31, "visible_fraction": 1.0,
  "knn": {"empty": 0.0, "half": 0.7, "full": 0.3, "p_wrong": 0.05},
  "temporal": {"prev_stock": "full", "next_stock": "half", "dt_prev_s": 3900, "dt_next_s": 4100}
}}}}
```
`knn` and `temporal` are present only when labels exist and never use the frame itself.

## vlm.json (producer: vlm)
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "gemini": {"stock": "half", "fill_pct": 55, "matches_expected": true,
             "observed": "11908", "confidence": 0.8, "reason": "bananas, about half full"},
  "claude": null
}}}}
```
`observed` is a candidate product_id, `other`, `empty` or `occluded`. Temperature 0, cached.

## results.json (producer: decide, consumer: dashboard)
```json
{"6215": {"1751052241.jpg": {"6215_2": {
  "pred": {"planogram_compliant": true, "stock_level": "half", "stock_value": 0.5},
  "p_stock": {"empty": 0.1, "half": 0.6, "full": 0.3}, "p_wrong": 0.05,
  "observed_product_id": null, "low_confidence": false, "reason": "...",
  "crop": "out/crops/6215/1751052241_6215_2_ctx.png"
}}}}
```

## submission.json and labels
Unchanged from `CONTRACTS.md`. Labels may add optional `observed_product_id` and `unsure` per area; the scorer ignores extra keys.
