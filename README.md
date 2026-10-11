# sensei-shelf

JunctionX Lisbon 2026, Sensei challenge "Fresh eyes on fresh produce": for each ceiling camera image, say per planogram area whether the product is compliant and its stock level (empty / half / full).

## Setup

Requires Python >= 3.10 (the pipeline uses `X | None` syntax). If your `python3` is older, use the explicit interpreter: `/opt/homebrew/bin/python3.12 -m venv .venv` on Mac, `python3.12 -m venv .venv` on Linux.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GEMINI_API_KEY

cd dashboard && npm install && npm run dev
```

Data is not in git. Unzip the organisers' dataset so you have:

```
data/
├── planogram.json                          # 14 cameras, 152 areas
└── Frutas e Legumes (train & test)/
    └── <camera_id>/<unix_ts>.jpg           # train + test mixed, no split folders
```

All paths live in `pipeline/common/config.py`.

## Running

Every module runs on its own from the repo root (`--help` for flags):

```bash
python -m pipeline.crop.crop
python -m pipeline.pixel.features
python -m pipeline.vlm.ask
python -m pipeline.decide.decide
python -m pipeline.eval.score --pred out/submission.json --gt labels/dev.json
./scripts/run_all.sh            # all of the above, in order (VLM off)
IMAGES=<validation dir> ./scripts/run_all.sh   # submission: out/submission.json, validation frames only
USE_VLM=1 ./scripts/run_all.sh  # also ask Gemini about the low-confidence areas (needs GEMINI_API_KEY)
WORKERS=1 ./scripts/run_all.sh  # serial, if the parallel stages ever misbehave (same output)
```

Until your upstream module works, drop a hand-written fake in `mocks/` and point `--index` / `--features` / etc. at it.

## Inspection API

Read-only HTTP surface over `out/` and `labels/`. No pipeline execution; just lookups.

```bash
uvicorn api.main:app --reload   # http://localhost:8000
```

| Endpoint | What it returns |
|---|---|
| `GET /api/cameras` | planogram cameras and their products |
| `GET /api/score` | `score(submission, labels)` as `{total, max, breakdown}` |
| `GET /api/stats` | run aggregates: stock, compliance, low_confidence, fallback |
| `GET /api/area/{cam}/{filename}/{area_id}` | one area: planogram + result + features + label |
| `GET /health` | `{status: "ok"}` |

### Dev with live dashboard

The dashboard reads `/data/*` from the API, so run both in parallel:

```bash
uvicorn api.main:app --reload       # terminal 1, port 8000
cd dashboard && npm run dev         # terminal 2, port 5173 (vite proxies /data to 8000)
```

Any `python -m pipeline.decide.decide` is visible in the browser on refresh. No `scripts/sync_dashboard.sh` needed. `/data/frames/{cam}/{file}?max=960` and `/data/crops/{cam}/{stem}.jpg?max=320` thumbnail on the fly (cached in `out/api_cache/`).
