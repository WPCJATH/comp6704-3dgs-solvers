"""Tier A experiment grids (256 x 256 image, N = 2000 Gaussians, every run at full length).
Runs every configuration of the named experiments in one process and skips runs that already
have a summary.json, so a job can be resumed or split across machines with --shard i/k.

    python tierA/grid.py --exp pilot,E2,E1,E5,E4 [--shard 0/2]

pilot : order-of-magnitude scan of the step size (100 iterations, decades) -> centre of E2
E2    : step-size sensitivity, 5 half-decade values x 4 first-order solvers, 1500 iterations
E1    : convergence, 9 solver variants, kodim23, 3 seeds, equal 10-minute budget
E5    : constraint handling with Adam: reparameterisation vs projection vs log barrier, 3000 iterations
E4    : time per iteration and peak memory against the number of Gaussians N (timing runs)
renders: seed 0 of E1 for seven solvers, saving the initial and final images (for the render figure)
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia3dgs import paths  # noqa: E402
from tierA.run2d import Cfg, run  # noqa: E402

FO = ["sgd", "momentum", "rmsprop", "adam"]
IMAGE = "kodim23"
SEEDS = [0, 1, 2]
LAM0 = 1e-2  # LM's initial damping; the gain-ratio rule adapts it within a few iterations


def out_dir(exp, name):
    return os.path.join(paths.runs(), "tierA", exp, name)


def load(exp):
    res = []
    for f in glob.glob(os.path.join(paths.runs(), "tierA", exp, "*", "summary.json")):
        with open(f) as fh:
            res.append(json.load(fh))
    return res


def _best(exp):
    best = {}
    for r in load(exp):
        c = r["cfg"]
        if r["diverged"]:
            continue
        if c["solver"] not in best or r["final_loss"] < best[c["solver"]][1]:
            best[c["solver"]] = (c["lr"], r["final_loss"])
    missing = [s for s in FO if s not in best]
    if missing:
        raise RuntimeError(f"{exp} has no stable step size for {missing}")
    return {s: v[0] for s, v in best.items()}


def pilot_cfgs():
    return [Cfg(out=out_dir("pilot", f"{s}_lr1e{k}"), solver=s, lr=10.0 ** k, iters=100, image=IMAGE)
            for s in FO for k in range(-4, 6)]


def e2_cfgs():
    c = _best("pilot")
    return [Cfg(out=out_dir("E2", f"{s}_g{j}"), solver=s, lr=c[s] * 10 ** m, iters=1500, image=IMAGE)
            for s in FO for j, m in enumerate([-1, -0.5, 0, 0.5, 1])]


def e2_best():
    return _best("E2")


def e1_cfgs():
    b = e2_best()
    cfgs = []
    for seed in SEEDS:
        common = dict(image=IMAGE, seed=seed, time_budget=600.0, iters=10 ** 6, lm_iters=10 ** 6)
        for s in FO:
            cfgs.append(Cfg(out=out_dir("E1", f"{s}_s{seed}"), solver=s, lr=b[s], **common))
        cfgs.append(Cfg(out=out_dir("E1", f"lbfgs_s{seed}"), solver="lbfgs", **common))
        cfgs.append(Cfg(out=out_dir("E1", f"lm_s{seed}"), solver="lm", lam0=LAM0, **common))
        cfgs.append(Cfg(out=out_dir("E1", f"hybrid_s{seed}"), solver="hybrid", lr=b["adam"], lam0=LAM0,
                        hybrid_max_adam=3000, **common))
        cfgs.append(Cfg(out=out_dir("E1", f"lm_noprec_s{seed}"), solver="lm_noprec", lam0=LAM0, **common))
        cfgs.append(Cfg(out=out_dir("E1", f"lm_group_s{seed}"), solver="lm_group", lam0=LAM0, **common))
    return cfgs


def e5_cfgs():
    b = e2_best()
    return [Cfg(out=out_dir("E5", f"adam_{mode}"), solver="adam", lr=b["adam"], mode=mode, iters=3000, image=IMAGE)
            for mode in ["reparam", "project", "barrier"]]


def e4_cfgs():
    b = e2_best()
    cfgs = []
    for n in [500, 1000, 2000, 4000, 8000]:
        for s in FO:
            cfgs.append(Cfg(out=out_dir("E4", f"{s}_n{n}"), solver=s, lr=b[s], n=n, iters=40, log_every=10))
        cfgs.append(Cfg(out=out_dir("E4", f"lbfgs_n{n}"), solver="lbfgs", n=n, iters=20))
        for s in ["lm", "lm_group", "lm_noprec"]:
            cfgs.append(Cfg(out=out_dir("E4", f"{s}_n{n}"), solver=s, lam0=LAM0, n=n, lm_iters=4))
    return cfgs


def renders_cfgs():
    """Seed 0 of E1 again, with the initial and final renders saved (E1 itself saved only curves).
    Written to runs/tierA/renders so the E1 numbers are not overwritten; the wall-clock budget makes
    the end point differ slightly from the E1 run, so figures quote this run's own PSNR."""
    keep = ["sgd", "momentum", "rmsprop", "adam", "lbfgs", "lm_group", "hybrid"]
    return [Cfg(**{**vars(c), "out": out_dir("renders", os.path.basename(c.out)), "save_image": 1})
            for c in e1_cfgs() if c.seed == 0 and os.path.basename(c.out).rsplit("_s", 1)[0] in keep]


BUILDERS = dict(pilot=pilot_cfgs, E2=e2_cfgs, E1=e1_cfgs, E5=e5_cfgs, E4=e4_cfgs, renders=renders_cfgs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--shard", default="0/1", help="i/k: run every k-th config starting at i")
    a = ap.parse_args()
    i, k = map(int, a.shard.split("/"))
    for exp in a.exp.split(","):
        cfgs = BUILDERS[exp]()[i::k]
        todo = [c for c in cfgs if not os.path.exists(os.path.join(c.out, "summary.json"))]
        print(f"[{exp}] {len(cfgs)} configs in shard, {len(todo)} to run", flush=True)
        for j, c in enumerate(todo):
            s = run(c)
            print(f"[{exp}] {j + 1}/{len(todo)} {os.path.basename(c.out)} loss={s['final_loss']:.5f} "
                  f"psnr={s['final_psnr']:.2f} t={s['time']:.1f}s div={s['diverged']}", flush=True)
