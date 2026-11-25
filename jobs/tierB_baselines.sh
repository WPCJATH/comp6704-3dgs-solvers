#!/usr/bin/env bash
# RMSProp with the same 30k recipe and learning rates as Adam, on every Tier B scene.
# (The Adam baselines are trained by tierB_hybrids.sh, which needs their 20k checkpoints.)
# lego and garden were run first; chair, ficus and bicycle were added later with the same command.
set -euo pipefail
for S in lego chair ficus garden bicycle; do python tierB/train3d.py --scene $S --method rmsprop; done
