"""Photometric loss of 3DGS, L = (1-lam) * L1 + lam * (1 - SSIM), and its least-squares form.

The least-squares form follows 3DGS-LM: each loss term is written as a sum of squared residuals,
r = sqrt(2 * w * rho(e)), so that 0.5 * ||r||^2 equals the loss. rho is a smoothed absolute value,
rho(e) = sqrt(e^2 + delta^2) - delta, which keeps the Jacobian bounded at e = 0.
"""
import math

import torch
import torch.nn.functional as F

SSIM_LAMBDA = 0.2
DELTA = 1e-3


def _gauss_window(size=11, sigma=1.5, channels=3, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


_WIN = {}


def ssim_map(img, ref):
    """Per-pixel, per-channel SSIM. img, ref: [B, H, W, 3] in [0, 1]. Returns [B, 3, H, W]."""
    x = img.permute(0, 3, 1, 2)
    y = ref.permute(0, 3, 1, 2)
    key = (x.device, x.shape[1], x.dtype)
    if key not in _WIN:
        _WIN[key] = _gauss_window(channels=x.shape[1], device=x.device).to(x.dtype)
    w = _WIN[key]
    pad = w.shape[-1] // 2
    c = x.shape[1]
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def photometric_loss(img, ref, lam=SSIM_LAMBDA):
    """The 3DGS training loss (exact L1)."""
    l1 = (img - ref).abs().mean()
    ssim = ssim_map(img, ref).mean()
    return (1 - lam) * l1 + lam * (1 - ssim)


def _rho(e):
    return torch.sqrt(e * e + DELTA * DELTA) - DELTA


def _signed_sqrt_rho(e):
    """sign(e) * sqrt(rho(e)) = e / sqrt(sqrt(e^2 + delta^2) + delta).

    Smooth everywhere, unlike sqrt(rho(e)), which behaves like |e| at e = 0 and has a kink there
    (exactly where a converged pixel or a matching background sits)."""
    return e / torch.sqrt(torch.sqrt(e * e + DELTA * DELTA) + DELTA)


def residuals(img, ref, lam=SSIM_LAMBDA, precise=False, parts="both"):
    """Residual vector r with 0.5 * ||r||^2 = smoothed photometric loss.

    precise=True evaluates the residuals in float64. SSIM's variances are differences of nearly
    equal numbers (E[x^2] - E[x]^2); in float32 that rounding noise swamps a finite-difference J v."""
    if precise:
        img, ref = img.double(), ref.double()
    n = img.numel()
    r1 = math.sqrt(2 * (1 - lam) / n) * _signed_sqrt_rho(img - ref)
    r2 = math.sqrt(2 * lam / n) * _signed_sqrt_rho(1 - ssim_map(img, ref)) if parts != "l1" else None
    if parts == "l1":
        return r1.reshape(-1)
    if parts == "ssim":
        return r2.reshape(-1)
    return torch.cat([r1.reshape(-1), r2.reshape(-1)])


def psnr(img, ref):
    mse = ((img - ref) ** 2).mean().clamp_min(1e-12)
    return float(-10 * math.log10(mse.item()))
