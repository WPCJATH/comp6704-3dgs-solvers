#!/usr/bin/env bash
# E6 hybrid on every scene: Adam's own 20k checkpoint, then full-batch LM over the colour (SH)
# coefficients in float64 for exactly the time Adam spent on steps 20k -> 30k. Chosen from the lego pilots:
# LM on 4 sampled views overfits them, and finite-difference Jv is unreliable for geometry and opacity.
set -euo pipefail
R=$(python -c 'from ia3dgs import paths; print(paths.runs())')/tierB
# Adam baseline first (it writes the 20k checkpoint), then the hybrid from it. Real scenes render 4 views
# per chunk so LM's working set stays inside the 48 GB card. Checkpoints are kept until the report is done.
for S in lego chair ficus garden bicycle; do
  C=10; case $S in garden|bicycle) C=4;; esac
  python tierB/train3d.py --scene $S --method adam
  python tierB/train3d.py --scene $S --method hybrid --lm_groups color --lm_views 0 --lm_precond group --lm_chunk $C
done
