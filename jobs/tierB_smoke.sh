#!/usr/bin/env bash
# Tier B smoke test (a few minutes): short Adam run with a checkpoint, a 30 s LM phase, the FD check.
# Outputs go to runs/tierB/*smoke* and are deleted at the end.
set -euo pipefail
python tierB/train3d.py --scene lego --method adam --steps 600 --switch 400 --eval_every 200 --tag _smoke
python tierB/train3d.py --scene lego --method hybrid --base_tag _smoke --tag _smoke --time_budget 30
python tierB/fd_check.py --scene lego --base_tag _smoke
R=$(python -c 'from ia3dgs import paths; print(paths.runs())')/tierB
cat $R/runs/lego_adam_smoke_s0/summary.json | head -30; tail -3 $R/runs/lego_hybrid_smoke_s0/log.jsonl
rm -rf $R/runs/lego_*_smoke_s0 $R/ckpt/lego_smoke_s0_adam_switch.pt $R/fd_check/lego.json
