"""E3: curvature of the Tier A objective, from exact Jacobian products.

At three points of an Adam run (initialisation, early, late) measure
  * diag(J^T J) per attribute group (Hutchinson, 128 probes);
  * the top eigenvalue of each group's diagonal block J_g^T J_g (power iteration), before and after
    Jacobi scaling D^{-1/2} J^T J D^{-1/2};
  * the leading 60 eigenvalues of J^T J and of the Jacobi-scaled operator (Lanczos with full
    reorthogonalisation);
  * how well Adam's second-moment estimate tracks the Gauss-Newton diagonal (Spearman correlation).
Writes runs/tierA/E3/<image>_s<seed>/spectrum.json and diag_samples.npz.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia3dgs import paths  # noqa: E402
from ia3dgs.lm import LeastSquares  # noqa: E402
from ia3dgs.losses import photometric_loss, residuals  # noqa: E402
from ia3dgs.model2d import GROUPS, Splat2D, load_image  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def flat(xs):
    return torch.cat([t.reshape(-1) for t in xs])


def unflat(v, like):
    out, i = [], 0
    for t in like:
        out.append(v[i:i + t.numel()].view_as(t))
        i += t.numel()
    return out


def lanczos(matvec, dim, k, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dim, generator=g, dtype=torch.float64).to(device)
    q /= q.norm()
    Q = [q]
    alphas, betas = [], []
    for j in range(k):
        w = matvec(Q[-1])
        a = float(w @ Q[-1])
        w = w - a * Q[-1] - (betas[-1] * Q[-2] if betas else 0)
        Qm = torch.stack(Q)
        w = w - Qm.T @ (Qm @ w)  # full reorthogonalisation
        w = w - Qm.T @ (Qm @ w)
        b = float(w.norm())
        alphas.append(a)
        if j == k - 1 or b < 1e-12:
            break
        betas.append(b)
        Q.append(w / b)
    T = np.diag(alphas) + np.diag(betas[:len(alphas) - 1], 1) + np.diag(betas[:len(alphas) - 1], -1)
    return np.sort(np.linalg.eigvalsh(T))[::-1]


def power(matvec, v0, iters=40):
    v = v0 / v0.norm()
    lam = 0.0
    for _ in range(iters):
        w = matvec(v)
        lam = float(v @ w)
        v = w / w.norm().clamp_min(1e-30)
    return lam


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def analyse(model, gt, x, adam_v, probes=128, k=60):
    dev = gt.device
    ls = LeastSquares(lambda xs: residuals(model.render(xs), gt[None]), exact_jvp=True)
    m = ls.F(x).numel()
    D = ls.diag_JTJ(x, probes, m)
    Df = flat(D).double().clamp_min(1e-30)
    sizes = [t.numel() for t in x]
    offs = np.cumsum([0] + sizes)

    def jtj(vf):
        return flat(ls.JTJ(x, [t.float() for t in unflat(vf, x)])).double()

    def jtj_scaled(vf):
        return jtj(vf / Df.sqrt()) / Df.sqrt()

    res = dict(groups={})
    g = torch.Generator(device="cpu").manual_seed(0)
    for gi, name in enumerate(GROUPS):
        mask = torch.zeros(offs[-1], dtype=torch.float64, device=dev)
        mask[offs[gi]:offs[gi + 1]] = 1
        v0 = torch.randn(offs[-1], generator=g, dtype=torch.float64).to(dev) * mask
        d = Df[offs[gi]:offs[gi + 1]].cpu().numpy()
        entry = dict(diag_median=float(np.median(d)), diag_p10=float(np.percentile(d, 10)),
                     diag_p90=float(np.percentile(d, 90)),
                     block_lmax=power(lambda v: jtj(v * mask) * mask, v0),
                     block_lmax_scaled=power(lambda v: jtj_scaled(v * mask) * mask, v0))
        if adam_v is not None:
            av = adam_v[offs[gi]:offs[gi + 1]]
            entry["spearman_adam_v_vs_gn_diag"] = spearman(av, d)
        res["groups"][name] = entry
    res["lanczos_raw"] = lanczos(jtj, offs[-1], k, dev).tolist()
    res["lanczos_jacobi"] = lanczos(jtj_scaled, offs[-1], k, dev).tolist()
    sample = np.random.default_rng(0).choice(offs[-1], size=min(4000, offs[-1]), replace=False)
    samples = dict(gn_diag=Df.cpu().numpy()[sample], group=np.searchsorted(offs, sample, side="right") - 1)
    if adam_v is not None:
        samples["adam_v"] = adam_v[sample]
    return res, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="kodim23")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None, help="Adam step size (default: best from E2)")
    ap.add_argument("--checkpoints", default="0,300,3000")
    a = ap.parse_args()
    if a.lr is None:
        from tierA.grid import e2_best
        a.lr = e2_best()["adam"]
    out = os.path.join(paths.runs(), "tierA", "E3", f"{a.image}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    gt = load_image(os.path.join(paths.datasets(), "kodak", a.image + ".png"))
    model = Splat2D(gt, 2000, "reparam", a.seed)
    params = [t.clone().requires_grad_(True) for t in model.x]
    opt = torch.optim.Adam(params, lr=a.lr, eps=1e-15)
    cps = sorted(int(c) for c in a.checkpoints.split(","))
    report, it = {"lr": a.lr, "checkpoints": {}}, 0
    for cp in cps:
        while it < cp:
            it += 1
            opt.zero_grad(set_to_none=True)
            loss = photometric_loss(model.render(params), gt[None])
            loss.backward()
            opt.step()
        x = [p.detach().clone() for p in params]
        adam_v = None
        if it > 0:
            adam_v = flat([opt.state[p]["exp_avg_sq"] for p in params]).double().cpu().numpy()
        with torch.no_grad():
            lval = float(photometric_loss(model.render(x), gt[None]))
        r, s = analyse(model, gt, x, adam_v)
        r["loss"] = lval
        report["checkpoints"][str(cp)] = r
        np.savez(os.path.join(out, f"diag_samples_it{cp}.npz"), **s)
        print(f"checkpoint {cp}: loss={lval:.5f} top eig raw={r['lanczos_raw'][0]:.3e} "
              f"jacobi={r['lanczos_jacobi'][0]:.3e}", flush=True)
        for name, e in r["groups"].items():
            print(f"   {name:10s} block_lmax={e['block_lmax']:.3e} scaled={e['block_lmax_scaled']:.3e} "
                  f"rho_adam={e.get('spearman_adam_v_vs_gn_diag', float('nan')):.2f}", flush=True)
    with open(os.path.join(out, "spectrum.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
