#!/usr/bin/env bash
# RMSProp with the same 30k recipe and learning rates as Adam, on one synthetic and one real scene.
# (The Adam baselines are trained by tierB_hybrids.sh, which needs their 20k checkpoints.)
set -euo pipefail
for S in lego garden; do python tierB/train3d.py --scene $S --method rmsprop; done
