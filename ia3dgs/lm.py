"""Matrix-free Levenberg-Marquardt for residual functions of a list of tensors.

The renderer (gsplat) has a reverse-mode backward pass only, so:
  J^T u  is exact: one backward pass (torch.autograd.grad with grad_outputs=u);
  J v    is a central finite difference: (F(x + h v) - F(x - h v)) / 2h, two forward passes.
The damped normal equations (J^T J + lam * D) dx = -J^T F are solved with preconditioned
conjugate gradients, never forming J^T J. D = diag(J^T J) is estimated with Rademacher probes
(Hutchinson): E[(J^T z)^2] = diag(J^T J). Steps are accepted with the gain-ratio test and the
damping follows Nielsen's rule (Nocedal & Wright Ch. 10; Madsen, Nielsen & Tingleff 2004).
"""
import time

import torch


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def _axpy(alpha, x, y):
    return [alpha * a + b for a, b in zip(x, y)]


class LeastSquares:
    """Wraps res_fn(list_of_tensors) -> 1-D residual tensor at a point xs (list of tensors)."""

    def __init__(self, res_fn, fd_h=1e-3, exact_jvp=False):
        self.res_fn = res_fn
        self.fd_h = fd_h
        self.exact_jvp = exact_jvp  # forward-mode AD (torch.func.jvp) instead of finite differences
        self.passes = 0  # forward-or-backward render passes, the unit of cost

    @torch.no_grad()
    def F(self, xs):
        """Residual vector F(x) (one render pass)."""
        self.passes += 1
        return self.res_fn(xs)

    def JT(self, xs, u):
        """J^T u by reverse-mode AD (one forward and one backward pass)."""
        xs_ = [x.detach().requires_grad_(True) for x in xs]
        with torch.enable_grad():
            r = self.res_fn(xs_)
            g = torch.autograd.grad(r, xs_, grad_outputs=u, allow_unused=True)
        self.passes += 2
        return [torch.zeros_like(x) if gi is None else gi for x, gi in zip(xs, g)]

    def residual_and_grad(self, xs):
        """Residuals F(x) and the gradient J^T F(x) in one forward and one backward pass."""
        xs_ = [x.detach().requires_grad_(True) for x in xs]
        with torch.enable_grad():
            r = self.res_fn(xs_)
            g = torch.autograd.grad(r, xs_, grad_outputs=r.detach(), allow_unused=True)
        self.passes += 2
        return r.detach(), [torch.zeros_like(x) if gi is None else gi for x, gi in zip(xs, g)]

    def J(self, xs, v):
        """J v by forward-mode AD if exact_jvp, else by a central finite difference (two passes either way)."""
        if self.exact_jvp:
            self.passes += 2
            xs_ = tuple(x.detach() for x in xs)
            _, jv = torch.func.jvp(lambda *t: self.res_fn(list(t)), xs_, tuple(v))
            return jv
        return self._J_fd(xs, v)

    @torch.no_grad()
    def _J_fd(self, xs, v):
        vmax = max(float(t.abs().max()) for t in v)
        if vmax == 0:
            return None
        h = self.fd_h / vmax  # largest single-coordinate perturbation is fd_h
        rp = self.F(_axpy(h, v, xs))
        rm = self.F(_axpy(-h, v, xs))
        return (rp - rm) / (2 * h)

    def JTJ(self, xs, v):
        """Gauss-Newton product J^T J v without forming J (four passes)."""
        jv = self.J(xs, v)
        if jv is None:
            return [torch.zeros_like(t) for t in v]
        return self.JT(xs, jv)

    def diag_JTJ(self, xs, probes, m, dtype=None):
        """Hutchinson estimate of diag(J^T J): mean of (J^T z)^2 over `probes` Rademacher vectors z of length m."""
        acc = [torch.zeros_like(x) for x in xs]
        for _ in range(probes):
            z = torch.empty(m, device=xs[0].device, dtype=dtype or xs[0].dtype).bernoulli_(0.5).mul_(2).sub_(1)
            g = self.JT(xs, z)
            acc = [a + gi * gi for a, gi in zip(acc, g)]
        return [a / probes for a in acc]


class ChunkedLeastSquares(LeastSquares):
    """Residuals split into chunks (e.g. groups of camera views) that are evaluated one at a time,
    so a full-batch Gauss-Newton product never holds every view's graph in memory.
    chunk_fns: list of callables, each list_of_tensors -> 1-D residual tensor."""

    def __init__(self, chunk_fns, fd_h=1e-3):
        super().__init__(None, fd_h=fd_h)
        self.fns = chunk_fns
        self.sizes = None

    @torch.no_grad()
    def F(self, xs):
        self.passes += 1
        rs = [f(xs) for f in self.fns]
        self.sizes = [r.numel() for r in rs]
        return torch.cat(rs)

    def _accumulate(self, xs, weights):
        xs_ = [x.detach().requires_grad_(True) for x in xs]
        acc = [torch.zeros_like(x) for x in xs]
        for f, w in zip(self.fns, weights):
            with torch.enable_grad():
                r = f(xs_)
                g = torch.autograd.grad(r, xs_, grad_outputs=w(r), allow_unused=True)
            acc = [a if gi is None else a + gi for a, gi in zip(acc, g)]
        self.passes += 2
        return acc

    def JT(self, xs, u):
        if self.sizes is None:
            self.F(xs)
        parts = list(torch.split(u, self.sizes))
        return self._accumulate(xs, [lambda r, p=p: p.to(r.dtype) for p in parts])

    def residual_and_grad(self, xs):
        rs = []

        def keep(r):
            rs.append(r.detach())
            return r.detach()

        g = self._accumulate(xs, [keep] * len(self.fns))
        self.sizes = [r.numel() for r in rs]
        return torch.cat(rs), g


def pcg(apply_A, b, M_inv, iters, tol=1e-2):
    """Solve A x = b by preconditioned CG from x = 0; stop at relative residual tol."""
    x = [torch.zeros_like(t) for t in b]
    r = [t.clone() for t in b]
    z = [mi * ri for mi, ri in zip(M_inv, r)]
    p = [t.clone() for t in z]
    rz = _dot(r, z)
    b_norm = float(_dot(b, b)) ** 0.5
    k = 0
    for k in range(1, iters + 1):
        Ap = apply_A(p)
        pAp = _dot(p, Ap)
        if pAp <= 0:
            break
        a = rz / pAp
        x = _axpy(a, p, x)
        r = _axpy(-a, Ap, r)
        if float(_dot(r, r)) ** 0.5 <= tol * b_norm:
            break
        z = [mi * ri for mi, ri in zip(M_inv, r)]
        rz_new = _dot(r, z)
        p = _axpy(rz_new / rz, p, z)
        rz = rz_new
    return x, k


class LM:
    """One object per optimisation run; call step() repeatedly.

    precond="group":  per-attribute Jacobi (proposed): D = mean of diag(J^T J) over each parameter tensor
                      (one scale per attribute), used for Marquardt damping and as the CG preconditioner.
    precond="jacobi": the same with a per-coordinate D (Hutchinson, few probes): an ablation.
    precond="none":   Levenberg's original damping D = I and plain CG.
    """

    def __init__(self, ls, lam0=1e-2, cg_iters=10, probes=4, precond="jacobi", project=None):
        self.ls, self.lam, self.nu = ls, lam0, 2.0
        self.cg_iters, self.probes, self.precond = cg_iters, probes, precond
        self.project = project  # optional callable(xs) -> xs, applied to trial points
        self.history = []

    def step(self, xs):
        """One LM step: PCG solve of the damped normal equations, gain-ratio test, damping update.

        Returns the (possibly unchanged) parameters and whether the step was accepted.
        """
        t0 = time.time()
        r0, g = self.ls.residual_and_grad(xs)
        E0 = 0.5 * float(r0 @ r0)
        if self.precond == "jacobi":  # per-coordinate Hutchinson estimate (noisy with few probes)
            D = self.ls.diag_JTJ(xs, self.probes, r0.numel(), r0.dtype)
            D = [d.clamp_min(1e-6 * float(d.mean()) + 1e-20) for d in D]
        elif self.precond == "group":  # per-attribute Jacobi: the Hutchinson estimate averaged over each tensor
            D = self.ls.diag_JTJ(xs, self.probes, r0.numel(), r0.dtype)
            D = [torch.full_like(d, float(d.mean()) + 1e-20) for d in D]
        else:
            D = [torch.ones_like(x) for x in xs]
        lam = self.lam

        def apply_A(v):
            jtj = self.ls.JTJ(xs, v)
            return [a + lam * d * vi for a, d, vi in zip(jtj, D, v)]

        M_inv = [1.0 / ((1 + lam) * d) for d in D] if self.precond != "none" else [torch.ones_like(x) for x in xs]
        dx, k = pcg(apply_A, [-gi for gi in g], M_inv, self.cg_iters)
        jtj_dx = self.ls.JTJ(xs, dx)
        pred = -float(_dot(g, dx)) - 0.5 * float(_dot(dx, jtj_dx))
        trial = _axpy(1.0, dx, xs)
        if self.project is not None:
            trial = self.project(trial)
        r1 = self.ls.F(trial)
        E1 = 0.5 * float(r1 @ r1)
        rho = (E0 - E1) / pred if pred > 0 else -1.0
        accepted = rho > 0 and E1 == E1
        if accepted:
            xs = trial
            self.lam = lam * max(1 / 3, 1 - (2 * rho - 1) ** 3)
            self.nu = 2.0
        else:
            self.lam = lam * self.nu
            self.nu *= 2
        self.history.append(dict(E0=E0, E1=E1, rho=rho, lam=lam, cg=k, accepted=accepted, dt=time.time() - t0))
        return xs, accepted
