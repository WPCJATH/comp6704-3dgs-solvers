"""Figure 1 (teaser): why Adam wins on the splatting objective.

(a) concept: gradient descent, Adam and a Gauss-Newton/LM step on a quadratic whose
    two axes have the 14x curvature gap measured in E3;
(b) measured (E3, iteration 3000): Adam's second moment against the Gauss-Newton
    diagonal for 4000 sampled parameters, with the median of each attribute;
(c) measured (E1, seed 0): PSNR against wall-clock time, 600 s per solver.

Usage: python make_teaser.py --runs results --out results/figures
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

INK, INK2, INK3, GRID = "#1B2433", "#52514e", "#8a8984", "#e4e4e0"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
ATTR = ["position", "scale", "rotation", "opacity", "colour"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK3, "ytick.color": INK3, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def load_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def panel_title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="left", color=INK, fontweight="bold", pad=6)


def concept(ax):
    k = 14.0                      # curvature ratio measured in E3 (rotation vs opacity)
    H = np.diag([1.0, k])
    x0 = np.array([-2.6, 0.55])
    X, Y = np.meshgrid(np.linspace(-3, 0.8, 300), np.linspace(-0.8, 0.8, 300))
    Z = 0.5 * (X ** 2 + k * Y ** 2)
    ax.contour(X, Y, Z, levels=np.geomspace(0.02, 4, 9), colors=GRID, linewidths=0.7)

    eta = 1.7 / k                 # gradient descent, step just inside the 2/L limit
    p, gd = x0.copy(), [x0.copy()]
    for _ in range(12):
        p = p - eta * H @ p
        gd.append(p.copy())
    gd = np.array(gd)
    ax.plot(gd[:, 0], gd[:, 1], "-", color=INK3, lw=1.0)
    ax.plot(gd[:, 0], gd[:, 1], "o", color=INK3, ms=2.2)

    p, ad, v = x0.copy(), [x0.copy()], np.zeros(2)   # Adam-like: per-coordinate scaled steps
    for t in range(1, 13):
        g = H @ p
        v = 0.9 * v + 0.1 * g ** 2
        p = p - 0.32 * g / (np.sqrt(v / (1 - 0.9 ** t)) + 1e-8)
        ad.append(p.copy())
    ad = np.array(ad)
    ax.plot(ad[:, 0], ad[:, 1], "-", color=BLUE, lw=1.6)
    ax.plot(ad[:, 0], ad[:, 1], "o", color=BLUE, ms=2.4)

    ax.add_patch(FancyArrowPatch(x0, (0, 0), arrowstyle="-|>", mutation_scale=7, color=AQUA, lw=1.8,
                                 shrinkA=0, shrinkB=1.5))   # Gauss-Newton: exact on a quadratic
    ax.plot([0], [0], marker="*", color=INK, ms=6, zorder=5)
    ax.plot([x0[0]], [x0[1]], "o", color=INK, ms=3.2, zorder=5)

    ax.text(-2.95, -0.74, "gradient descent: zig-zags", color=INK2, fontsize=6.3, ha="left")
    ax.text(-0.5, -0.5, "Adam: one step size\nper coordinate", color=BLUE, fontsize=6.3, ha="center", fontweight="bold")
    ax.text(-0.85, 0.42, "LM: one exact step,\n≈27× the cost", color="#13805a", fontsize=6.3, ha="center", fontweight="bold")
    ax.set_xlim(-3, 0.8); ax.set_ylim(-0.8, 0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("flat direction (e.g. opacity)")
    ax.set_ylabel("steep direction\n(e.g. rotation)")
    panel_title(ax, "a", "A badly conditioned valley")


def tracking(ax, runs):
    d = np.load(os.path.join(runs, "tierA", "E3", "kodim23_s0", "diag_samples_it3000.npz"))
    gn, v, grp = d["gn_diag"], d["adam_v"], d["group"]
    ok = (gn > 0) & (v > 0)
    ax.scatter(gn[ok], v[ok], s=2.0, color=GRAY, alpha=0.45, lw=0, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    offs = {0: (7, -11), 1: (7, 2), 2: (7, 3), 3: (-7, -9), 4: (-7, 3)}
    for g, name in enumerate(ATTR):
        m = ok & (grp == g)
        x, y = np.median(gn[m]), np.median(v[m])
        ax.plot([x], [y], "o", ms=5.5, mfc=BLUE, mec="white", mew=1.0, zorder=4)
        dx, dy = offs[g]
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.4, color=INK,
                    ha="left" if dx > 0 else "right")
    ax.set_xlim(np.percentile(gn[ok], 0.5) / 3, np.percentile(gn[ok], 99.9) * 3)
    ax.set_ylim(np.percentile(v[ok], 0.5) / 3, np.percentile(v[ok], 99.9) * 3)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
    ax.set_xlabel("Gauss–Newton curvature  diag(JᵀJ)")
    ax.set_ylabel("Adam's second moment  v")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.text(0.03, 0.97, "Spearman ρ = 0.85–0.92\nper attribute", transform=ax.transAxes, va="top",
            fontsize=6.4, color=INK)
    panel_title(ax, "b", "Adam's v tracks curvature")


def race(ax, runs):
    base = os.path.join(runs, "tierA", "E1")
    others = [("sgd", "SGD"), ("momentum", "Nesterov"), ("rmsprop", "RMSProp"), ("lbfgs", "L-BFGS")]
    main = [("lm_group", "LM", AQUA), ("hybrid", "Adam → LM", ORANGE), ("adam", "Adam", BLUE)]
    ends = {}
    for key, name in others:
        L = load_log(os.path.join(base, f"{key}_s0", "log.jsonl"))
        t = [r["t"] for r in L]; p = [r["psnr"] for r in L]
        ax.plot(t, p, color=GRAY, lw=1.0)
        ends[name] = (t[-1], p[-1], GRAY)
    for key, name, c in main:
        L = load_log(os.path.join(base, f"{key}_s0", "log.jsonl"))
        t = [r["t"] for r in L]; p = [r["psnr"] for r in L]
        ax.plot(t, p, color=c, lw=1.8 if key != "lm_group" else 1.5)
        ends[name] = (t[-1], p[-1], c)
    items = sorted(ends.items(), key=lambda kv: kv[1][1])     # direct labels at the right end, nudged apart
    ys, last = [], -1e9
    for name, (t, p, c) in items:
        y = max(p, last + 0.95)
        ys.append((name, t, p, y, c)); last = y
    for name, t, p, y, c in ys:
        if name == "Adam → LM":
            continue
        col = INK2 if c == GRAY else INK
        ax.annotate(f"{name}  {p:.1f}", (t, p), xytext=(612, y), textcoords="data", fontsize=6.2, color=col,
                    va="center", fontweight="bold" if c != GRAY else "normal", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.5) if abs(y - p) > 0.3 else None)
    ax.set_xlim(0, 605); ax.set_ylim(15, 34.2)
    ax.set_xlabel("wall-clock time (s), same budget for every solver")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_xticks([0, 200, 400, 600])

    ins = ax.inset_axes([0.50, 0.15, 0.40, 0.27])            # the last 220 s, Adam vs the hybrid
    for key, c in [("adam", BLUE), ("hybrid", ORANGE)]:
        L = load_log(os.path.join(base, f"{key}_s0", "log.jsonl"))
        t = np.array([r["t"] for r in L]); p = np.array([r["psnr"] for r in L])
        m = t >= 380
        ins.plot(t[m], p[m], color=c, lw=1.4)
    ins.set_xlim(380, 605); ins.set_ylim(32.3, 32.75)
    ins.tick_params(labelsize=5.2, length=2, pad=1)
    ins.set_yticks([32.4, 32.6])
    ins.set_xticks([400, 500, 600])
    for s_ in ins.spines.values():
        s_.set_color(INK3); s_.set_linewidth(0.5)
    ins.set_facecolor("white")
    ins.text(0.96, 0.93, "zoom: last 220 s", transform=ins.transAxes, fontsize=5.4, color=INK2, va="top", ha="right")
    ins.text(0.04, 0.10, "Adam → LM (−0.05 dB)", transform=ins.transAxes, fontsize=5.6, color=ORANGE, ha="left", fontweight="bold")
    ins.text(0.04, 0.30, "Adam", transform=ins.transAxes, fontsize=5.6, color=BLUE, ha="left", fontweight="bold")
    panel_title(ax, "c", "Same time budget: Adam wins")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default=os.path.join("results", "figures"))
    a = ap.parse_args()
    fig = plt.figure(figsize=(7.2, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.25], wspace=0.42, left=0.065, right=0.9, top=0.86, bottom=0.17)
    concept(fig.add_subplot(gs[0]))
    tracking(fig.add_subplot(gs[1]), a.runs)
    race(fig.add_subplot(gs[2]), a.runs)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"ia_fig1_teaser.{ext}"), dpi=300, facecolor="white")
    print("wrote", os.path.join(a.out, "ia_fig1_teaser.{pdf,png}"))


if __name__ == "__main__":
    main()
