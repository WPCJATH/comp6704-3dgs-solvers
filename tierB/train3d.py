"""Tier B: 3D Gaussian Splatting with gsplat, standard 30k-iteration recipe.

    python tierB/train3d.py --scene lego --method adam      # baseline; saves a checkpoint at 20k
    python tierB/train3d.py --scene lego --method rmsprop
    python tierB/train3d.py --scene lego --method hybrid    # resumes Adam's 20k checkpoint, then LM

Recipe (gsplat simple_trainer defaults): per-attribute learning rates, means lr decayed x0.01,
SH degree raised every 1000 steps to 3, loss 0.8 L1 + 0.2 D-SSIM, default densification
(start 500, stop 15k, every 100, opacity reset every 3k), one training view per step.
The hybrid replaces Adam's last 10k steps with matrix-free Levenberg-Marquardt on the fixed set
of Gaussians in Adam's 20k checkpoint (densification has stopped by then), for the same wall-clock
time that Adam spent on steps 20k -> 30k. --lm_groups chooses the attributes LM optimises; the
reported hybrid (jobs/tierB_hybrids.sh) uses the colour coefficients only, every training view per
step (--lm_groups color --lm_views 0), chosen from the lego pilots (jobs/tierB_stage1.sh,
jobs/tierB_pilots_fullbatch.sh).

Third-party: rasterisation and densification come from gsplat (Apache-2.0), imported as a library.
The learning rates and schedule follow gsplat's simple_trainer defaults, i.e. the original 3DGS recipe.
"""
import argparse
import json
import math
import os
import socket
import sys
import time
import warnings

import imageio.v2 as imageio
import numpy as np
import torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia3dgs import paths  # noqa: E402
from ia3dgs.data3d import load_scene  # noqa: E402
from ia3dgs.lm import LM, ChunkedLeastSquares, LeastSquares  # noqa: E402
from ia3dgs.losses import photometric_loss, psnr, residuals, ssim_map  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
C0 = 0.28209479177387814
KEYS = ["means", "scales", "quats", "opacities", "sh0", "shN"]


def knn_dist(x, k=3, chunk=4096):
    out = []
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i:i + chunk], x)
        out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(-1))
    return torch.cat(out)


def init_params(scene, seed, device="cuda"):
    torch.manual_seed(seed)
    xyz, rgb = scene.init_xyz, scene.init_rgb
    n = xyz.shape[0]
    dist = knn_dist(xyz).clamp_min(1e-7)
    p = {
        "means": xyz.clone(),
        "scales": torch.log(dist)[:, None].repeat(1, 3),
        "quats": torch.rand(n, 4, device=device),
        "opacities": torch.logit(torch.full((n,), 0.1, device=device)),
        "sh0": ((rgb - 0.5) / C0)[:, None, :],
        "shN": torch.zeros(n, 15, 3, device=device),
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()})


def base_lrs(scene_scale):
    return {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
            "sh0": 2.5e-3, "shN": 2.5e-3 / 20}


def make_optimizers(params, method, scene_scale, lr_mult=1.0):
    opts = {}
    for k, lr in base_lrs(scene_scale).items():
        group = [{"params": params[k], "lr": lr * lr_mult, "name": k}]
        if method in ("adam", "hybrid"):
            opts[k] = torch.optim.Adam(group, eps=1e-15)
        elif method == "rmsprop":
            opts[k] = torch.optim.RMSprop(group, alpha=0.99, eps=1e-15)
        else:
            raise ValueError(method)
    return opts


def render(p, viewmats, Ks, W, H, sh_degree, bg, need_info=False):
    colors = torch.cat([p["sh0"], p["shN"]], 1)
    img, _, info = rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]),
                                 colors, viewmats, Ks, W, H, sh_degree=sh_degree, packed=False,
                                 backgrounds=bg[None].expand(viewmats.shape[0], 3))
    return (img.clamp(0, 1) if not need_info else img), info


class Evaluator:
    def __init__(self, scene, out, n_curve=10):
        self.s, self.out = scene, out
        nt = scene.test.images.shape[0]
        self.curve_idx = np.linspace(0, nt - 1, min(n_curve, nt)).round().astype(int).tolist()
        self.lpips = None

    def _view(self, p, i, deg):
        t = self.s.test
        H, W = t.images.shape[1:3]
        with torch.no_grad():
            img, _ = render(p, t.viewmats[i:i + 1], t.Ks[i:i + 1], W, H, deg, self.s.background)
        return img.clamp(0, 1), t.images[i:i + 1].float() / 255

    def curve_psnr(self, p, deg):
        return float(np.mean([psnr(*self._view(p, i, deg)) for i in self.curve_idx]))

    def full(self, p, deg, tag):
        if self.lpips is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
        ps, ss, ls = [], [], []
        nt = self.s.test.images.shape[0]
        os.makedirs(os.path.join(self.out, "renders"), exist_ok=True)
        for i in range(nt):
            img, gt = self._view(p, i, deg)
            ps.append(psnr(img, gt))
            ss.append(float(ssim_map(img, gt).mean()))
            with torch.no_grad():
                ls.append(float(self.lpips(img.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2))))
            if i in (self.curve_idx[0], self.curve_idx[len(self.curve_idx) // 2]):
                imageio.imwrite(os.path.join(self.out, "renders", f"{tag}_test{i:03d}.png"),
                                (img[0].cpu().numpy() * 255).round().astype(np.uint8))
        return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), lpips=float(np.mean(ls)))


def ckpt_path(scene, seed, tag=""):
    return os.path.join(paths.runs(), "tierB", "ckpt", f"{scene}{tag}_s{seed}_adam_switch.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--method", choices=["adam", "rmsprop", "hybrid"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--switch", type=int, default=20_000, help="Adam checkpoint step used by the hybrid")
    ap.add_argument("--lr_mult", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="suffix of the run name")
    ap.add_argument("--base_tag", default="", help="hybrid: tag of the Adam run whose checkpoint it resumes")
    # LM phase of the hybrid
    ap.add_argument("--lm_views", type=int, default=0, help="views per LM step; 0 = all training views")
    ap.add_argument("--lm_chunk", type=int, default=10, help="views rendered together in full-batch mode")
    ap.add_argument("--lm_cg", type=int, default=8)
    ap.add_argument("--lm_probes", type=int, default=2)
    ap.add_argument("--lm_lam0", type=float, default=1e-2)
    ap.add_argument("--lm_precond", default="group", choices=["group", "jacobi", "none"])
    ap.add_argument("--lm_fd_h", type=float, default=1e-3)
    ap.add_argument("--lm_groups", default="all", help="all | color (sh0, shN) | appearance (+ opacities)")
    ap.add_argument("--lm_fp32", action="store_true", help="residuals in float32 (default float64, see losses.py)")
    ap.add_argument("--time_budget", type=float, default=0.0, help="LM seconds; 0 = Adam's 20k->30k time")
    ap.add_argument("--eval_every", type=int, default=1000)
    a = ap.parse_args()

    name = f"{a.scene}_{a.method}{a.tag}_s{a.seed}"
    out = os.path.join(paths.runs(), "tierB", "runs", name)
    if os.path.exists(os.path.join(out, "summary.json")):
        print("done already:", out)
        return
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    scene = load_scene(paths.datasets(), a.scene)
    tr = scene.train
    H, W = tr.images.shape[1:3]
    ev = Evaluator(scene, out)
    log = open(os.path.join(out, "log.jsonl"), "w")
    t_train = 0.0
    torch.cuda.reset_peak_memory_stats()
    summary = dict(scene=a.scene, method=a.method, seed=a.seed, args=vars(a), host=socket.gethostname(),
                   gpu=torch.cuda.get_device_name(), n_train=tr.images.shape[0], n_test=scene.test.images.shape[0],
                   H=H, W=W)

    def write(**kw):
        kw["t"] = round(t_train, 3)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    if a.method in ("adam", "rmsprop"):
        params = init_params(scene, a.seed)
        opts = make_optimizers(params, a.method, scene.scene_scale, a.lr_mult)
        sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / a.steps))
        strat = DefaultStrategy(verbose=False)
        strat.check_sanity(params, opts)
        st = strat.initialize_state(scene_scale=scene.scene_scale)
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        perm = []
        write(step=0, test_psnr=ev.curve_psnr(params, 0), n=params["means"].shape[0], loss=None)
        for step in range(a.steps):
            if not perm:
                perm = torch.randperm(tr.images.shape[0], generator=g).tolist()
            i = perm.pop()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            deg = min(step // 1000, 3)
            img, info = render(params, tr.viewmats[i:i + 1], tr.Ks[i:i + 1], W, H, deg, scene.background, True)
            gt = tr.images[i:i + 1].float() / 255
            loss = photometric_loss(img, gt)
            strat.step_pre_backward(params, opts, st, step, info)
            loss.backward()
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            sched.step()
            strat.step_post_backward(params, opts, st, step, info, packed=False)
            torch.cuda.synchronize()
            t_train += time.perf_counter() - t0
            lv = float(loss)
            if not math.isfinite(lv):
                summary["diverged_at"] = step
                break
            s1 = step + 1
            if s1 % a.eval_every == 0 or s1 == a.steps:
                write(step=s1, loss=lv, test_psnr=ev.curve_psnr(params, deg), n=params["means"].shape[0])
            if a.method == "adam" and s1 == a.switch:
                os.makedirs(os.path.dirname(ckpt_path(a.scene, a.seed, a.tag)), exist_ok=True)
                torch.save({"params": {k: v.detach() for k, v in params.items()}, "t_train": t_train, "step": s1},
                           ckpt_path(a.scene, a.seed, a.tag))
                summary["t_at_switch"] = t_train
        deg = 3
    else:  # hybrid: resume Adam's checkpoint, then LM
        ck = torch.load(ckpt_path(a.scene, a.seed, a.base_tag), map_location="cuda")
        params = {k: v.clone() for k, v in ck["params"].items()}
        t_train = ck["t_train"]
        base = json.load(open(os.path.join(paths.runs(), "tierB", "runs", f"{a.scene}_adam{a.base_tag}_s{a.seed}", "summary.json")))
        budget = a.time_budget or (base["time_train"] - base["t_at_switch"])
        summary["lm_time_budget"] = budget
        deg = 3
        write(step=ck["step"], test_psnr=ev.curve_psnr(params, deg), n=params["means"].shape[0], phase="adam")
        lm_keys = {"all": KEYS, "color": ["sh0", "shN"], "appearance": ["opacities", "sh0", "shN"]}[a.lm_groups]
        g = torch.Generator(device="cpu").manual_seed(a.seed)
        batch = {"idx": None}

        def res_views(xs, idx):
            p = dict(params)
            p.update(dict(zip(lm_keys, xs)))
            img, _ = render(p, tr.viewmats[idx], tr.Ks[idx], W, H, deg, scene.background, True)
            return residuals(img, tr.images[idx].float() / 255, precise=not a.lm_fp32)

        full_batch = a.lm_views <= 0
        if full_batch:  # deterministic Gauss-Newton over every training view, evaluated chunk by chunk
            n_tr = tr.images.shape[0]
            chunks = [list(range(i, min(i + a.lm_chunk, n_tr))) for i in range(0, n_tr, a.lm_chunk)]
            ls = ChunkedLeastSquares([lambda xs, c=c: res_views(xs, c) for c in chunks], fd_h=a.lm_fd_h)
        else:  # a fresh random batch of views for every LM step
            ls = LeastSquares(lambda xs: res_views(xs, batch["idx"]), fd_h=a.lm_fd_h)
        lm = LM(ls, lam0=a.lm_lam0, cg_iters=a.lm_cg, probes=a.lm_probes, precond=a.lm_precond)
        x = [params[k] for k in lm_keys]
        t_lm, k = 0.0, 0
        while t_lm < budget:
            if not full_batch:
                batch["idx"] = torch.randperm(tr.images.shape[0], generator=g)[:a.lm_views].tolist()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x, acc = lm.step(x)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            t_lm += dt
            t_train += dt
            k += 1
            params.update(dict(zip(lm_keys, x)))
            h = lm.history[-1]
            rec = dict(step=ck["step"], lm_iter=k, phase="lm", E0=h["E0"], E1=h["E1"], rho=h["rho"], lam=h["lam"],
                       cg=h["cg"], accepted=acc, n=params["means"].shape[0])
            if full_batch or k % 5 == 0:
                rec["test_psnr"] = ev.curve_psnr(params, deg)
            write(**rec)
        summary.update(lm_iters=k, lm_accepted=sum(int(r["accepted"]) for r in lm.history),
                       lm_passes=ls.passes, lm_time=t_lm)
        write(step=ck["step"], lm_iter=k, phase="lm_end", test_psnr=ev.curve_psnr(params, deg),
              n=params["means"].shape[0])

    summary.update(time_train=t_train, n_gaussians=int(params["means"].shape[0]),
                   peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20, test=ev.full(params, deg, name))
    log.close()
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("time_train", "n_gaussians", "peak_mem_mb", "test")}), flush=True)


if __name__ == "__main__":
    main()
