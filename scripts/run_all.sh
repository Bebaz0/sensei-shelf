#!/usr/bin/env bash
# Run the full pipeline in order. Each step reads the previous step's output from out/.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pipeline.crop.crop
python -m pipeline.pixel.features
python -m pipeline.vlm.ask
python -m pipeline.decide.decide
python -m pipeline.eval.score
