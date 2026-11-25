#!/usr/bin/env bash
# Smoke test: one short Tier A run per solver, written to runs/tierA/smoke.
set -e
O=$(python -c 'from ia3dgs import paths; print(paths.runs())')/tierA/smoke
for s in sgd momentum rmsprop adam; do python tierA/run2d.py --solver $s --lr 1e-2 --iters 200 --out $O/$s; done
python tierA/run2d.py --solver lbfgs --iters 50 --out $O/lbfgs
python tierA/run2d.py --solver lm --lm_iters 10 --out $O/lm
python tierA/run2d.py --solver hybrid --lr 1e-2 --hybrid_max_adam 200 --lm_iters 5 --out $O/hybrid
python tierA/run2d.py --solver adam --mode project --iters 200 --out $O/adam_project
python tierA/run2d.py --solver adam --mode barrier --iters 200 --out $O/adam_barrier
