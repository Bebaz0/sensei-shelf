#!/usr/bin/env bash
# Run the full pipeline in order. Each step reads the previous step's output from out/.
#   IMAGES=<dir>  validation run: crops of <dir> are APPENDED to the train crops (kNN and temporal
#                 features need them) and every later stage only processes <dir> (out/run_frames.json).
#   NO_API=1      skip the VLM stage (stale vlm.json is removed so it cannot leak in).
# Scoring only runs on train runs that have labels.
set -euo pipefail
cd "$(dirname "$0")/.."

# Generated per run. Keep out/crops*, out/cache/ and out/frames_by_time/.
rm -f out/run_frames.json out/features.json out/vlm.json out/results.json out/submission.json

# Validation run on a fresh clone: build the train crops first, --append needs them as the kNN bank.
if [ -n "${IMAGES:-}" ] && [ ! -s out/crops_index.json ]; then
  python -m pipeline.crop.crop
fi
python -m pipeline.crop.crop ${IMAGES:+--images "$IMAGES" --append}
python -m pipeline.pixel.features
if [ -z "${NO_API:-}" ]; then
  python -m pipeline.vlm.ask
fi
python -m pipeline.decide.decide ${IMAGES:+--images "$IMAGES"}

if [ -z "${IMAGES:-}" ] && [ -s labels/dev.json ] && [ "$(tr -d ' \n' < labels/dev.json)" != "{}" ]; then
  python -m pipeline.eval.score
else
  echo "skip scoring (validation run or no labels)"
fi
