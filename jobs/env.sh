#!/usr/bin/env bash
# Run a command from the repository root inside the conda environment (default name: ia3dgs).
#   bash jobs/env.sh bash jobs/tierA_all.sh
#   bash jobs/env.sh python tierA/run2d.py --solver adam --lr 0.1 --out runs/tmp/adam
# Machine-specific settings (e.g. IA3DGS_ROOT pointing at shared storage) can go in jobs/local.env,
# which is git-ignored and sourced if present.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f jobs/local.env ] && source jobs/local.env
if [ "${CONDA_DEFAULT_ENV:-}" != "${IA3DGS_ENV:-ia3dgs}" ]; then
  # conda is often not on PATH in non-interactive shells (ssh, job queues): also try the usual install folders
  for base in "$(conda info --base 2>/dev/null)" "${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}" \
              "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
    if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
      source "$base/etc/profile.d/conda.sh"
      conda activate "${IA3DGS_ENV:-ia3dgs}"
      break
    fi
  done
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME="$(python -c 'from ia3dgs import paths; print(paths.cache())')/torch"
exec "$@"
