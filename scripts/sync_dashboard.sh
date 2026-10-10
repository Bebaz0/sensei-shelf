#!/usr/bin/env bash
# Copy a small demo run (3 frames per camera by default, JPG thumbnails) into dashboard/public/data/.
# Usage: scripts/sync_dashboard.sh [frames_per_camera]
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/sync_dashboard.py "$@"
