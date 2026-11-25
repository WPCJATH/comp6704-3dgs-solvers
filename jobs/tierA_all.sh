#!/usr/bin/env bash
# Tier A, all experiments in dependency order (pilot -> E2 -> E1, E5, E4 -> E3). Resumable.
# To split the long E1 stage over two GPUs, run a second job with SHARD=1/2 after E2 is done
# and launch this one with SHARD=0/2.
set -euo pipefail
python tierA/grid.py --exp pilot
python tierA/grid.py --exp E2
python tierA/grid.py --exp E1,E5,E4 --shard "${SHARD:-0/1}"
if [ "${SHARD:-0/1}" = "0/1" ] || [ "${SHARD%%/*}" = "0" ]; then python tierA/spectrum2d.py --image kodim23 --seed 0; fi
