#!/usr/bin/env bash
# Tier B stage 1 (lego), as reported: an Adam baseline run kept under the tag _pilotbase (30k, checkpoint at 20k),
# the finite-difference adjoint check on its checkpoint, and two LM pilots with 4 sampled views per step
# (per-coordinate Jacobi, the method at the time). Both pilots overfit the sampled views; see the results page.
set -euo pipefail
python tierB/train3d.py --scene lego --method adam --tag _pilotbase
python tierB/fd_check.py --scene lego --base_tag _pilotbase
for G in all color; do
  python tierB/train3d.py --scene lego --method hybrid --base_tag _pilotbase --tag _pilot_$G --time_budget 300 \
    --lm_groups $G --lm_views 4 --lm_precond jacobi
done
