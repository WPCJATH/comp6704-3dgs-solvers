#!/usr/bin/env bash
# Run a command from the repository root inside the conda environment (default name: ia3dgs).
#   bash jobs/env.sh bash jobs/tierA_all.sh
#   bash jobs/env.sh python tierA/run2d.py --solver adam --lr 0.1 --out runs/tmp/adam
# Machine-specific settings (e.g. IA3DGS_ROOT pointing at shared storage) can go in jobs/local.env,
# which is git-ignored and sourced if present.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f jobs/local.env ] && source jobs/local.env
if [ "${CONDA_DEFAULT_ENV:-}" != "${IA3DGS_ENV:-ia3dgs}" ] && command -v conda >/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${IA3DGS_ENV:-ia3dgs}"
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$(python -c 'from ia3dgs import paths; print(paths.cache())')/torch"
exec "$@"
