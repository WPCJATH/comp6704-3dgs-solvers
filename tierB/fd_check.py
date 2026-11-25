"""Adjoint test on a real scene: does finite-difference J v agree with gsplat's backward J^T u?

    python tierB/fd_check.py --scene lego [--base_tag ""]

Loads the Adam checkpoint used by the hybrid, takes 4 training views, and for each attribute
group reports |u^T(Jv) - (J^T u)^T v| / (|u^T(Jv)| + |(J^T u)^T v|) at several step sizes h.
gsplat skips a Gaussian at a pixel once alpha < 1/255 and clips footprints at 3 sigma, so the
rendered image is piecewise smooth and the two products need not agree.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia3dgs import paths  # noqa: E402
from ia3dgs.data3d import load_scene  # noqa: E402
from ia3dgs.lm import LeastSquares  # noqa: E402
from ia3dgs.losses import residuals  # noqa: E402
from tierB.train3d import KEYS, ckpt_path, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="lego")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_tag", default="")
    a = ap.parse_args()
    scene = load_scene(paths.datasets(), a.scene)
    tr = scene.train
    H, W = tr.images.shape[1:3]
    params = torch.load(ckpt_path(a.scene, a.seed, a.base_tag), map_location="cuda")["params"]
    idx = [0, 1, 2, 3]
    x = [params[k] for k in KEYS]

    def fn(xs, what):
        img, _ = render(dict(zip(KEYS, xs)), tr.viewmats[idx], tr.Ks[idx], W, H, 3, scene.background, True)
        gt = tr.images[idx].float() / 255
        if what == "image":
            return img.reshape(-1)
        part, prec = {"res_l1": ("l1", False), "res_ssim": ("ssim", False), "res_ssim_fp64": ("ssim", True),
                      "residual": ("both", False), "residual_fp64": ("both", True)}[what]
        return residuals(img, gt, precise=prec, parts=part)

    out = {}
    for what in ["image", "res_l1", "res_ssim", "res_ssim_fp64", "residual", "residual_fp64"]:
        for gi, name in enumerate(KEYS + ["all"]):
            for h in [1e-4, 1e-3, 1e-2]:
                errs = []
                for rep in range(3):
                    ls = LeastSquares(lambda xs: fn(xs, what), fd_h=h)
                    torch.manual_seed(rep)
                    v = [torch.randn_like(t) if (name == "all" or i == gi) else torch.zeros_like(t)
                         for i, t in enumerate(x)]
                    jv = ls.J(x, v)
                    u = torch.randn_like(jv)
                    jtu = ls.JT(x, u)
                    lhs, rhs = float(u @ jv), float(sum((p * q).sum() for p, q in zip(jtu, v)))
                    errs.append(abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30))
                out[f"{what}/{name}/h={h:g}"] = sum(errs) / len(errs)
                print(f"{what:14s} {name:10s} h={h:.0e} rel_err={out[f'{what}/{name}/h={h:g}']:.4f}", flush=True)
    d = os.path.join(paths.runs(), "tierB", "fd_check")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{a.scene}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
