"""One Tier A run: fit one image with one solver. Writes <out>/log.jsonl and <out>/summary.json.

    python tierA/run2d.py --solver adam --lr 1e-2 --image kodim23 --out /tmp/x

Solvers: sgd, momentum (Nesterov), rmsprop, adam, lbfgs (strong Wolfe), lm_group (per-attribute
Jacobi PCG, proposed), lm (per-coordinate Jacobi, ablation), lm_noprec (Levenberg damping, plain CG),
hybrid (Adam, then per-attribute LM once the loss plateaus).
Cost is reported in wall-clock seconds (optimisation only, evaluation excluded) and in render
passes (one forward or one backward pass through the rasterizer = 1).
"""
import argparse
import json
import math
import os
import socket
import sys
import time
import warnings
from dataclasses import asdict, dataclass

import imageio.v2 as imageio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia3dgs import paths  # noqa: E402
from ia3dgs.lm import LM, LeastSquares  # noqa: E402
from ia3dgs.losses import photometric_loss, psnr, residuals  # noqa: E402
from ia3dgs.model2d import Splat2D, load_image  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
FIRST_ORDER = ["sgd", "momentum", "rmsprop", "adam"]


@dataclass
class Cfg:
    out: str = ""
    solver: str = "adam"
    lr: float = 1e-2
    image: str = "kodim23"
    size: int = 256
    n: int = 2000
    mode: str = "reparam"
    seed: int = 0
    iters: int = 3000          # first-order / L-BFGS iteration cap
    lm_iters: int = 150        # LM iteration cap
    time_budget: float = 0.0   # seconds of optimisation; 0 = no cap
    lam0: float = 1e-2
    cg_iters: int = 10
    probes: int = 4
    hybrid_window: int = 100
    hybrid_tol: float = 2e-3
    hybrid_max_adam: int = 2000
    mu0: float = 1e-2
    mu1: float = 1e-5
    log_every: int = 10
    save_image: int = 0        # 1 = also write the initial and final renders as PNG (outside the timed loop)


def make_opt(solver, params, lr):
    if solver == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if solver == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if solver == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-15)
    if solver in ("adam", "hybrid"):
        return torch.optim.Adam(params, lr=lr, eps=1e-15)
    raise ValueError(solver)


class Logger:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.f = open(os.path.join(out, "log.jsonl"), "w")
        self.t_opt = 0.0
        self.t_mark = None

    def start(self):
        torch.cuda.synchronize()
        self.t_mark = time.perf_counter()

    def pause(self):
        torch.cuda.synchronize()
        self.t_opt += time.perf_counter() - self.t_mark

    def write(self, **kw):
        kw["t"] = round(self.t_opt, 4)
        self.f.write(json.dumps(kw) + "\n")

    def close(self):
        self.f.close()


def run(cfg: Cfg):
    torch.manual_seed(cfg.seed)
    dev = "cuda"
    gt = load_image(os.path.join(paths.datasets(), "kodak", cfg.image + ".png"), cfg.size, dev)
    model = Splat2D(gt, cfg.n, cfg.mode, cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    log = Logger(cfg.out)
    x = model.x
    passes = 0
    diverged = False
    infeasible = 0
    phase_switch = None
    it = 0

    @torch.no_grad()
    def evaluate(xs):
        img = model.render(xs)
        return float(photometric_loss(img, gt[None])), psnr(img, gt[None])

    def over_time():
        return cfg.time_budget > 0 and log.t_opt >= cfg.time_budget

    @torch.no_grad()
    def save_png(xs, name):
        img = model.render(xs)[0].clamp(0, 1).cpu().numpy()
        imageio.imwrite(os.path.join(cfg.out, name), (img * 255).round().astype("uint8"))

    loss0, p0 = evaluate(x)
    log.write(it=0, loss=loss0, psnr=p0, passes=0, phase="init")
    if cfg.save_image:
        save_png(x, "init.png")

    # ---------------- first-order and the Adam phase of the hybrid ----------------
    if cfg.solver in FIRST_ORDER + ["hybrid"]:
        params = [t.clone().requires_grad_(True) for t in x]
        opt = make_opt(cfg.solver, params, cfg.lr)
        hist = []
        cap = cfg.hybrid_max_adam if cfg.solver == "hybrid" else cfg.iters
        log.start()
        while it < cap and not over_time():
            it += 1
            opt.zero_grad(set_to_none=True)
            img = model.render(params)
            loss = photometric_loss(img, gt[None])
            obj = loss
            if cfg.mode == "barrier":
                frac = min(1.0, it / cap)
                mu = cfg.mu0 * (cfg.mu1 / cfg.mu0) ** frac
                obj = loss + mu * model.barrier(params)
            obj.backward()
            if cfg.mode == "barrier":
                old = [p.detach().clone() for p in params]
            opt.step()
            passes += 2
            with torch.no_grad():
                if cfg.mode == "project":
                    new, moved = model.project([p.detach() for p in params])
                    infeasible += moved
                    for p, v in zip(params, new):
                        p.copy_(v)
                elif cfg.mode == "barrier":
                    new, cut = model.fraction_to_boundary(old, [p.detach() for p in params])
                    infeasible += cut
                    for p, v in zip(params, new):
                        p.copy_(v)
            lv = float(loss)
            if not math.isfinite(lv) or lv > 10 * max(loss0, 1e-3):
                diverged = True
                break
            hist.append(lv)
            if it % cfg.log_every == 0:
                log.pause()
                log.write(it=it, loss=lv, psnr=psnr(img.detach(), gt[None]), passes=passes, phase="first")
                log.start()
            if cfg.solver == "hybrid" and it > cfg.hybrid_window * 2:
                old_l = sum(hist[-2 * cfg.hybrid_window:-cfg.hybrid_window]) / cfg.hybrid_window
                new_l = sum(hist[-cfg.hybrid_window:]) / cfg.hybrid_window
                if (old_l - new_l) / old_l < cfg.hybrid_tol:
                    break
        log.pause()
        x = [p.detach() for p in params]
        if cfg.solver == "hybrid":
            phase_switch = dict(it=it, t=log.t_opt, passes=passes)

    # ---------------- L-BFGS ----------------
    if cfg.solver == "lbfgs":
        params = [t.clone().requires_grad_(True) for t in x]
        opt = torch.optim.LBFGS(params, lr=1, max_iter=1, history_size=20, line_search_fn="strong_wolfe")
        state = {"loss": None, "img": None}

        def closure():
            nonlocal passes
            opt.zero_grad(set_to_none=True)
            img = model.render(params)
            loss = photometric_loss(img, gt[None])
            loss.backward()
            passes += 2
            state["loss"], state["img"] = float(loss), img.detach()
            return loss

        log.start()
        while it < cfg.iters and not over_time():
            it += 1
            opt.step(closure)
            if not math.isfinite(state["loss"]):
                diverged = True
                break
            if it % max(1, cfg.log_every // 5) == 0:
                log.pause()
                l, p = evaluate(params)
                log.write(it=it, loss=l, psnr=p, passes=passes, phase="lbfgs")
                log.start()
        log.pause()
        x = [p.detach() for p in params]

    # ---------------- Levenberg-Marquardt (also the second phase of the hybrid) ----------------
    if cfg.solver in ("lm", "lm_group", "lm_noprec", "hybrid") and not diverged:
        ls = LeastSquares(lambda xs: residuals(model.render(xs), gt[None]), exact_jvp=True)
        project = (lambda xs: model.project(xs)[0]) if cfg.mode == "project" else None
        lm = LM(ls, lam0=cfg.lam0, cg_iters=cfg.cg_iters, probes=cfg.probes,
                precond={"lm": "jacobi", "lm_noprec": "none"}.get(cfg.solver, "group"), project=project)
        k = 0
        log.start()
        while k < cfg.lm_iters and not over_time():
            k += 1
            it += 1
            x, acc = lm.step(x)
            log.pause()
            l, p = evaluate(x)
            if not math.isfinite(l):
                diverged = True
                break
            h = lm.history[-1]
            log.write(it=it, loss=l, psnr=p, passes=passes + ls.passes, phase="lm", lam=h["lam"],
                      rho=h["rho"], cg=h["cg"], accepted=acc)
            log.start()
        log.pause()
        passes += ls.passes

    lf, pf = evaluate(x)
    if cfg.save_image:
        save_png(x, "final.png")
    summary = dict(cfg=asdict(cfg), host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                   final_loss=lf, final_psnr=pf, init_loss=loss0, iters=it, passes=passes,
                   time=log.t_opt, peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                   diverged=diverged or not math.isfinite(lf), infeasible=infeasible,
                   phase_switch=phase_switch)
    log.close()
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in asdict(Cfg()).items():
        ap.add_argument("--" + k, type=type(v), default=v)
    s = run(Cfg(**vars(ap.parse_args())))
    print(json.dumps({k: s[k] for k in ("final_loss", "final_psnr", "time", "passes", "peak_mem_mb", "diverged")}))
