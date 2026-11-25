"""Correctness checks for the matrix-free LM machinery (run on a GPU: pytest -q tests)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ia3dgs.lm import LM, ChunkedLeastSquares, LeastSquares, pcg  # noqa: E402
from ia3dgs.losses import photometric_loss, residuals  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _toy(m=300, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, n, generator=g, dtype=torch.float64).to(DEV)
    b = torch.randn(m, generator=g, dtype=torch.float64).to(DEV)
    return A, b


def test_fd_jvp_matches_exact_on_smooth_function():
    A, b = _toy()
    f = lambda xs: torch.tanh(A @ xs[0]) - b  # noqa: E731
    x = [torch.randn(A.shape[1], dtype=torch.float64, device=DEV) * 0.1]
    v = [torch.randn_like(x[0])]
    ls = LeastSquares(f, fd_h=1e-4)
    _, exact = torch.func.jvp(lambda t: f([t]), (x[0],), (v[0],))
    assert torch.allclose(ls.J(x, v), exact, rtol=1e-5, atol=1e-7)


def test_chunked_least_squares_matches_unchunked():
    A, b = _toy(m=300, n=40)
    f = lambda xs: torch.tanh(A @ xs[0]) - b  # noqa: E731
    parts = [slice(0, 100), slice(100, 220), slice(220, 300)]
    fns = [lambda xs, s=s: (torch.tanh(A[s] @ xs[0]) - b[s]) for s in parts]
    x = [torch.randn(40, dtype=torch.float64, device=DEV) * 0.1]
    full, chunk = LeastSquares(f), ChunkedLeastSquares(fns)
    assert torch.allclose(full.F(x), chunk.F(x))
    u = torch.randn(300, dtype=torch.float64, device=DEV)
    assert torch.allclose(full.JT(x, u)[0], chunk.JT(x, u)[0])
    r1, g1 = full.residual_and_grad(x)
    r2, g2 = chunk.residual_and_grad(x)
    assert torch.allclose(r1, r2) and torch.allclose(g1[0], g2[0])


def test_hutchinson_diagonal_is_unbiased():
    A, b = _toy(m=2000, n=20)
    ls = LeastSquares(lambda xs: A @ xs[0] - b)
    x = [torch.zeros(A.shape[1], dtype=torch.float64, device=DEV)]
    est = ls.diag_JTJ(x, probes=400, m=A.shape[0])[0]
    exact = (A * A).sum(0)
    assert ((est - exact).abs() / exact).max() < 0.25


def test_pcg_solves_spd_system():
    A, _ = _toy(m=100, n=30)
    H = A.T @ A + 1e-2 * torch.eye(30, dtype=torch.float64, device=DEV)
    b = torch.randn(30, dtype=torch.float64, device=DEV)
    x, _ = pcg(lambda v: [H @ v[0]], [b], [1 / H.diag()], iters=200, tol=1e-10)
    assert torch.allclose(H @ x[0], b, atol=1e-6)


def test_lm_reaches_linear_least_squares_solution():
    A, b = _toy()
    ls = LeastSquares(lambda xs: A @ xs[0] - b)
    lm = LM(ls, lam0=1e-3, cg_iters=200, probes=64)
    x = [torch.zeros(A.shape[1], dtype=torch.float64, device=DEV)]
    for _ in range(30):
        x, _ = lm.step(x)
    xstar = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
    assert torch.allclose(x[0], xstar, atol=1e-4)


def _small_scene(mode="reparam"):
    from ia3dgs.model2d import Splat2D
    g = torch.Generator().manual_seed(0)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 64), torch.linspace(0, 1, 64), indexing="ij")
    gt = torch.stack([xx, yy, (xx * 6).sin() * 0.5 + 0.5], -1).to(DEV)
    return Splat2D(gt, 200, mode=mode, seed=0), gt


@cuda
def test_reverse_mode_matches_forward_mode_gradient():
    """Checkpointed chunked backward vs forward-mode directional derivative of the loss."""
    model, gt = _small_scene()
    x = [t.clone().requires_grad_(True) for t in model.x]
    photometric = lambda xs: photometric_loss(model.render(xs), gt[None])  # noqa: E731
    g = torch.autograd.grad(photometric(x), x)
    v = [torch.randn_like(t) for t in model.x]
    _, dd = torch.func.jvp(lambda *t: photometric(list(t)), tuple(model.x), tuple(v))
    gv = sum((a * b).sum() for a, b in zip(g, v))
    assert abs(float(gv - dd)) / abs(float(dd)) < 1e-3


def test_residuals_are_smooth_at_zero_error():
    """A pixel with zero error must not create a kink in the L1 residuals: one-sided slopes agree.
    (The D-SSIM residuals sit at a smooth minimum there, so their one-sided slopes differ by O(h).)"""
    ref = torch.full((1, 16, 16, 3), 0.5, dtype=torch.float64)
    f = lambda t: residuals(ref + t, ref)[:ref.numel()]  # noqa: E731
    h = 1e-6
    d_plus = (f(torch.full_like(ref, h)) - f(torch.zeros_like(ref))) / h
    d_minus = (f(torch.zeros_like(ref)) - f(torch.full_like(ref, -h))) / h
    assert torch.allclose(d_plus, d_minus, rtol=1e-3, atol=1e-9)


@cuda
def test_residuals_reproduce_loss():
    model, gt = _small_scene()
    img = model.render(model.x)
    r = residuals(img, gt[None])
    assert abs(0.5 * float(r @ r) - float(photometric_loss(img, gt[None]))) < 2e-3


@cuda
def test_jacobian_products_are_adjoint():
    """u^T (J v) must equal (J^T u)^T v: forward-mode J v against checkpointed reverse-mode J^T u."""
    model, gt = _small_scene()
    ls = LeastSquares(lambda xs: residuals(model.render(xs), gt[None]), exact_jvp=True)
    torch.manual_seed(1)
    v = [torch.randn_like(t) * 1e-2 for t in model.x]
    jv = ls.J(model.x, v)
    u = torch.randn_like(jv)
    jtu = ls.JT(model.x, u)
    lhs = float(u @ jv)
    rhs = float(sum((a * b).sum() for a, b in zip(jtu, v)))
    assert abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-12) < 1e-4, (lhs, rhs)


@cuda
@pytest.mark.parametrize("precond", ["group", "jacobi", "none"])
def test_lm_decreases_splat_loss(precond):
    model, gt = _small_scene()
    ls = LeastSquares(lambda xs: residuals(model.render(xs), gt[None]), exact_jvp=True)
    lm = LM(ls, lam0=1e-2, cg_iters=10, probes=4, precond=precond)
    x = model.x
    e0 = 0.5 * float(ls.F(x) @ ls.F(x))
    for _ in range(15):
        x, _ = lm.step(x)
    e1 = 0.5 * float(ls.F(x) @ ls.F(x))
    assert e1 < 0.8 * e0, (e0, e1)
