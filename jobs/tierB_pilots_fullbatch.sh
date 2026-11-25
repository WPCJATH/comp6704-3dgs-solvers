#!/usr/bin/env bash
# LM pilots on lego from the _pilotbase Adam checkpoint, full batch (all 100 training views per LM step),
# float64 residuals, per-coordinate Jacobi (the method at the time), 300 s each. Run after tierB_stage1.sh.
# Their outcome fixed the final hybrid: colour coefficients only, all views.
set -euo pipefail
for G in all color appearance; do
  python tierB/train3d.py --scene lego --method hybrid --base_tag _pilotbase --tag _fb_$G --time_budget 300 \
    --lm_groups $G --lm_views 0 --lm_precond jacobi
done
