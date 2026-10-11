#!/usr/bin/env bash
# Run the full pipeline in order. Each step reads the previous step's output from out/.
#   IMAGES=<dir>  validation run: crops of <dir> are APPENDED to the train crops (kNN and temporal
#                 features need them) and every later stage only processes <dir> (out/run_frames.json).
#   USE_VLM=1     also ask the VLM about the low-confidence areas (off by default).
#   NO_API=1      with USE_VLM: no VLM calls, vlm.json is written with null for every provider.
# Scoring only runs on train runs that have labels.
set -euo pipefail
cd "$(dirname "$0")/.."

# Generated per run. Keep out/crops*, out/cache/ and out/frames_by_time/.
rm -f out/run_frames.json out/features.json out/vlm.json out/vlm_queue.txt out/results.json out/submission.json

# Validation run on a fresh clone: build the train crops first, --append needs them as the kNN bank.
if [ -n "${IMAGES:-}" ] && [ ! -s out/crops_index.json ]; then
  python -m pipeline.crop.crop
fi
python -m pipeline.crop.crop ${IMAGES:+--images "$IMAGES" --append}
python -m pipeline.pixel.features
# VLM off by default: on the gate it gave +12 / 3818 LOFO (bootstrap interval includes 0) for ~11 min of API calls.
# USE_VLM=1: decide without the VLM, ask the VLM only about the low-confidence areas, then decide again.
if [ -n "${USE_VLM:-}" ]; then
  python -m pipeline.decide.decide ${IMAGES:+--images "$IMAGES"} --queue out/vlm_queue.txt
  python -m pipeline.vlm.ask ${NO_API:+--no-api} --areas out/vlm_queue.txt
fi
python -m pipeline.decide.decide ${IMAGES:+--images "$IMAGES"}

if [ -z "${IMAGES:-}" ] && [ -s labels/dev.json ] && [ "$(tr -d ' \n' < labels/dev.json)" != "{}" ]; then
  python -m pipeline.eval.score
else
  echo "skip scoring (validation run or no labels)"
fi
