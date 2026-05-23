#!/usr/bin/env bash
# setup_env.sh — reproduce the `squint` conda env + assets needed to run
# eval1 / eval2 / eval3 (final_utils/eval{1,2,3}*).
#
# Steps:
#   1. Create the `squint` conda env from environment.yaml
#        - python 3.10
#        - torch / torchvision (CUDA 12.8 wheels)
#        - tensordict, torchrl, mani_skill_nightly, gymnasium, coacd
#        - numpy, tyro, tqdm, wandb, opencv-python
#        - ultralytics      (FastSAM masking — final_utils.fastsam_seg)
#        - rerun-sdk        (eval2/eval3 --save_window .rrd recording)
#        - lerobot[feetech] (real SO101 follower)
#   2. Download FastSAM-s.pt weights into ./weights/ (required for the default
#      --fastsam masking path in eval1/2/3).
#
# Usage:
#   bash setup_env.sh
#   conda activate squint
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- conda ---
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found — install miniconda first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx squint; then
  echo "[setup] squint env already exists — updating from environment.yaml"
  conda env update -n squint -f environment.yaml --prune
else
  echo "[setup] creating squint env from environment.yaml"
  conda env create -f environment.yaml
fi

# --- FastSAM weights ---
mkdir -p weights
if [ ! -s weights/FastSAM-s.pt ] || [ "$(stat -c%s weights/FastSAM-s.pt)" -lt 1000000 ]; then
  echo "[setup] downloading FastSAM-s.pt -> weights/"
  curl -fL --retry 3 -o weights/FastSAM-s.pt \
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/FastSAM-s.pt"
fi
echo "[setup] FastSAM weights: $(ls -lh weights/FastSAM-s.pt | awk '{print $5, $9}')"

echo
echo "[setup] done. Activate with:"
echo "    conda activate squint"
