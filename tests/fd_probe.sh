#!/usr/bin/env bash
# Short lego Adam run (kept checkpoint) + finite-difference check, then clean up. Diagnostic only.
set -euo pipefail
python tierB/train3d.py --scene lego --method adam --steps 2000 --switch 2000 --eval_every 1000 --tag _probe
python tierB/fd_check.py --scene lego --base_tag _probe
R=$(python -c 'from ia3dgs import paths; print(paths.runs())')/tierB
rm -rf $R/runs/lego_adam_probe_s0 $R/ckpt/lego_probe_s0_adam_switch.pt
