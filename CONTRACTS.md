# Data contracts

Every module talks to the next only through these files in `out/`. Change a contract = PR touching this file + tell the consumer.

Pipeline: `crop` → `pixel` + `vlm` → `decide` → `eval` / `dashboard`

## crops_index.json
Producer: `pipeline/crop` (A). Consumers: `pixel`, `vlm`, `decide`.

TODO: schema + example (one entry per camera × image × area; crop path, polygon, visible fraction).

## features.json
Producer: `pipeline/pixel` (B). Consumer: `decide`.

TODO: schema + example.

## vlm.json
Producer: `pipeline/vlm`. Consumer: `decide`.

TODO: schema + example.

## results.json
Producer: `pipeline/decide` (C). Consumer: `dashboard` (D).

TODO: schema + example (per area verdict plus whatever the UI needs: confidence, reasons, crop path).

## submission.json
Producer: `pipeline/decide` (C). Consumer: `pipeline/eval/score.py`, organisers.

Fixed by the organisers, same as `annotations.json` and `labels/dev.json`. Must contain every area of every image.

```json
{"6215": {"1751052241.jpg": {"6215_2": {"planogram_compliant": true, "stock_level": "half", "stock_value": 0.5}}}}
```

`stock_level` ∈ `empty | half | full` → `stock_value` 0 / 0.5 / 1.
