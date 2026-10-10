# sensei-shelf

JunctionX Lisbon 2026, Sensei challenge "Fresh eyes on fresh produce": for each ceiling camera image, say per planogram area whether the product is compliant and its stock level (empty / half / full).

## Setup

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
./scripts/run_all.sh            # all of the above, in order
```

Until your upstream module works, drop a hand-written fake in `mocks/` and point `--index` / `--features` / etc. at it.
