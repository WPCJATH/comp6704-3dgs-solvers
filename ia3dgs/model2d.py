"""Tier A: fit one image with N 2D Gaussians, alpha-composited front to back in a fixed order.

A pure-PyTorch renderer with no culling thresholds, so the objective is smooth and both
Jacobian products are exact: J^T u by reverse mode (chunked, checkpointed) and J v by forward
mode (torch.func.jvp). gsplat's CUDA rasterizer skips a Gaussian at a pixel once alpha < 1/255
and clips footprints at 3 sigma; those jumps make finite-difference J v disagree with its
backward pass (tierB/fd_check.py), which is why the controlled tier uses this renderer.

Per pixel p:   alpha_i(p) = 0.99 * o_i * exp(-0.5 * d^T Sigma_i^{-1} d),  d = p - mu_i
               C(p) = sum_i c_i alpha_i(p) prod_{j<i} (1 - alpha_j(p))     (black background)
Sigma_i = R(r_i) diag(s_i)^2 R(r_i)^T, with r_i a unit 2-vector (the 2D analogue of the quaternion).

Variables per Gaussian: mean (2), scale (2), rotation (2), opacity (1), colour (3) -> 10 numbers.
Constraints: s > 0, ||r|| = 1, 0 <= o <= 1, 0 <= c <= 1. Inequalities are handled by
  reparam : s = exp(s~), o = sigmoid(o~), c = sigmoid(c~)          (unconstrained problem P2)
  project : optimise s, o, c directly, project onto the box after every step
  barrier : optimise s, o, c directly, add mu * mean(-log slack) and keep iterates strictly
            feasible with a per-coordinate fraction-to-boundary rule
The equality ||r|| = 1 is always handled by normalisation inside the renderer.
"""
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

GROUPS = ["means", "scales", "rotations", "opacities", "colors"]
S_MIN, TAU = 1e-4, 0.99
CHUNK_ELEMS = 2 ** 23  # pixels x Gaussians per chunk


def load_image(path, size=256, device="cuda"):
    img = imageio.imread(path)[..., :3]
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div(255).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")
    return t[0].permute(1, 2, 0).contiguous().to(device)


def _render_chunk(pix, means, scales, rot, opac, colors):
    r = rot / rot.norm(dim=-1, keepdim=True)
    d = pix[:, None, :] - means[None]                       # [P, N, 2]
    u1 = d[..., 0] * r[:, 0] + d[..., 1] * r[:, 1]           # R^T d
    u2 = -d[..., 0] * r[:, 1] + d[..., 1] * r[:, 0]
    q = ((u1 / scales[:, 0]) ** 2 + (u2 / scales[:, 1]) ** 2).clamp(max=60.0)  # exp(-30) ~ 1e-13; avoids inf*0
    alpha = 0.99 * opac * torch.exp(-0.5 * q)                # [P, N]
    log_t = torch.cumsum(torch.log1p(-alpha), dim=1)
    trans = torch.exp(torch.cat([torch.zeros_like(log_t[:, :1]), log_t[:, :-1]], dim=1))
    return (alpha * trans) @ colors                         # [P, 3]


class Splat2D:
    def __init__(self, gt, n, mode="reparam", seed=0):
        self.gt, self.n, self.mode = gt, n, mode
        self.H, self.W = gt.shape[:2]
        dev = gt.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        means = torch.rand(n, 2, generator=g) * torch.tensor([self.W, self.H])
        scale0 = 0.5 * (self.W * self.H / n) ** 0.5
        scales = scale0 * (0.5 + torch.rand(n, 2, generator=g))
        th = torch.rand(n, generator=g) * 2 * torch.pi
        rot = torch.stack([th.cos(), th.sin()], -1)
        opac = torch.full((n,), 0.5)
        colors = 0.1 + 0.8 * torch.rand(n, 3, generator=g)
        if mode == "reparam":
            scales, opac, colors = scales.log(), torch.logit(opac), torch.logit(colors)
        self.x = [t.to(dev).contiguous() for t in (means, scales, rot, opac, colors)]
        ys, xs = torch.meshgrid(torch.arange(self.H, device=dev), torch.arange(self.W, device=dev), indexing="ij")
        self.pix = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5
        self.chunk = max(256, CHUNK_ELEMS // n)

    def actual(self, x):
        means, s, r, o, c = x
        if self.mode == "reparam":
            return means, torch.exp(s), r, torch.sigmoid(o), torch.sigmoid(c)
        return means, s, r, o, c

    def render(self, x):
        """[1, H, W, 3]. Checkpointed per chunk when a graph is being built."""
        params = self.actual(x)
        need_graph = torch.is_grad_enabled() and any(p.requires_grad for p in params)
        out = []
        for i in range(0, self.pix.shape[0], self.chunk):
            pix = self.pix[i:i + self.chunk]
            if need_graph:
                out.append(checkpoint(_render_chunk, pix, *params, use_reentrant=False))
            else:
                out.append(_render_chunk(pix, *params))
        return torch.cat(out).reshape(1, self.H, self.W, 3)

    # ---- constraint handling for the non-reparameterised modes ----
    def bounds(self):
        """(index, lower, upper) for every box-constrained group."""
        return [(1, S_MIN, None), (3, 0.0, 1.0), (4, 0.0, 1.0)]

    @torch.no_grad()
    def project(self, x):
        """Euclidean projection onto the feasible set; returns new list and #coordinates moved."""
        x = [t.clone() for t in x]
        moved = 0
        for i, lo, hi in self.bounds():
            before = x[i].clone()
            x[i].clamp_(min=lo, max=hi)
            moved += int((x[i] != before).sum())
        x[2] = x[2] / x[2].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x, moved

    @torch.no_grad()
    def fraction_to_boundary(self, x_old, x_new):
        """Keep iterates strictly inside: a coordinate that would cross a bound moves TAU of the way."""
        out = [t.clone() for t in x_new]
        cut = 0
        for i, lo, hi in self.bounds():
            lo_viol = out[i] <= lo
            out[i] = torch.where(lo_viol, lo + (1 - TAU) * (x_old[i] - lo), out[i])
            cut += int(lo_viol.sum())
            if hi is not None:
                hi_viol = out[i] >= hi
                out[i] = torch.where(hi_viol, hi - (1 - TAU) * (hi - x_old[i]), out[i])
                cut += int(hi_viol.sum())
        out[2] = out[2] / out[2].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return out, cut

    def barrier(self, x):
        terms = [torch.log(x[1] - S_MIN).mean(), torch.log(x[3]).mean(), torch.log1p(-x[3]).mean(),
                 torch.log(x[4]).mean(), torch.log1p(-x[4]).mean()]
        return -sum(terms) / len(terms)
