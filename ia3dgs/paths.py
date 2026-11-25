"""Where datasets and run outputs live.

Defaults (relative to the repository root, both git-ignored):
    data/datasets/   kodak/, nerf_synthetic/, mipnerf360/   (see tools/download_data.sh)
    data/cache/      torch hub weights (LPIPS) and other caches
    runs/            outputs of new runs: runs/tierA/<exp>/<run>/, runs/tierB/..., runs/figures/

The logs behind the report are shipped in results/ (same layout as runs/); point the figure
scripts at them with `--runs results`.

Overrides, in order of precedence:
    IA3DGS_DATA, IA3DGS_RUNS, IA3DGS_CACHE   one directory each
    IA3DGS_ROOT                              <root>/datasets, <root>/runs, <root>/cache
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pick(var, root_sub, default):
    if os.environ.get(var):
        return os.environ[var]
    if os.environ.get("IA3DGS_ROOT"):
        return os.path.join(os.environ["IA3DGS_ROOT"], root_sub)
    return default


def datasets() -> str:
    p = _pick("IA3DGS_DATA", "datasets", os.path.join(REPO, "data", "datasets"))
    if not os.path.isdir(p):
        raise FileNotFoundError(f"dataset folder {p} not found: run tools/download_data.sh or set IA3DGS_DATA")
    return p


def runs() -> str:
    return _pick("IA3DGS_RUNS", "runs", os.path.join(REPO, "runs"))


def cache() -> str:
    p = _pick("IA3DGS_CACHE", "cache", os.path.join(REPO, "data", "cache"))
    os.makedirs(p, exist_ok=True)
    return p
