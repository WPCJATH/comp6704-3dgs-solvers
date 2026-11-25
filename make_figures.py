"""Regenerate every figure and the numbers behind every table from the saved run logs.

    python make_figures.py [--runs DIR] [--out DIR]     (default: <runs>/figures)
    python make_figures.py --runs results                 (from the logs shipped with the repository; CPU only)

Writes PNG figures and results.json (all table values). Missing experiments are skipped.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ia3dgs import paths  # noqa: E402

# Fixed categorical order (validated: CVD-adjacent dE >= 9.1 on the light surface); colour follows the solver.
SOLVERS = ["sgd", "momentum", "rmsprop", "adam", "lbfgs", "lm_group", "lm_noprec", "hybrid"]  # plotted (8 slots)
COLOR = dict(zip(SOLVERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
TABLE_ONLY = ["lm"]  # per-coordinate Jacobi ablation: in the tables, not the curves
LABEL = {"sgd": "SGD", "momentum": "Nesterov momentum", "rmsprop": "RMSProp", "adam": "Adam", "lbfgs": "L-BFGS",
         "lm": "LM, per-coordinate Jacobi", "lm_group": "LM, per-attribute Jacobi", "lm_noprec": "LM, no scaling (D = I)",
         "hybrid": "Adam → LM (ours)"}
DASH = {"sgd": (0, (1, 1.5)), "momentum": (0, (4, 1.5)), "rmsprop": (0, (6, 2, 1, 2)), "adam": "-",
        "lbfgs": (0, (2, 2)), "lm_group": "-", "lm_noprec": (0, (5, 3)), "hybrid": (0, (6, 2.5))}
GROUP_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e6e6e6"
COL_W = 5.5  # NeurIPS text width in inches; figures are drawn at print size

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.titlelocation": "left", "axes.titlepad": 4, "axes.titleweight": "bold",
    "lines.linewidth": 1.3, "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 1.2,
    "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
})


def loss_ticks(ax, ticks=(0.02, 0.03, 0.05, 0.1, 0.2, 0.3)):
    """Plain-number ticks on a log axis spanning less than two decades."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    lo, hi = ax.get_ylim()
    t = [v for v in ticks if lo <= v <= hi]
    ax.yaxis.set_major_locator(FixedLocator(t))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def save(fig, out, name):
    """PNG for the web page, PDF (vector, embedded fonts) for the LaTeX report."""
    fig.savefig(os.path.join(out, name + ".png"))
    fig.savefig(os.path.join(out, name + ".pdf"))
    plt.close(fig)


def jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summaries(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            s = json.load(fh)
        s["_dir"] = os.path.dirname(f)
        out.append(s)
    return out


def step_interp(xs, ys, grid):
    """Last observed value at or before each grid point (NaN before the first observation)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    idx = np.searchsorted(xs, grid, side="right") - 1
    out = np.where(idx >= 0, ys[np.clip(idx, 0, None)], np.nan)
    out[grid > xs[-1]] = np.nan if len(xs) == 0 else ys[-1]
    return out


def fmt_pm(vals, nd=2):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return dict(mean=round(m, nd + 2), std=round(s, nd + 2), n=len(vals))


# ------------------------------------------------------------------ Tier A
def tier_a(R, out, res):
    base = os.path.join(R, "tierA")
    # ---- E2: step-size sensitivity heatmap
    e2 = summaries(os.path.join(base, "E2", "*", "summary.json"))
    if e2:
        rows = [s for s in ["sgd", "momentum", "rmsprop", "adam"] if any(r["cfg"]["solver"] == s for r in e2)]
        table = {}
        from matplotlib.colors import LinearSegmentedColormap
        cmap_e2 = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
        fig, ax = plt.subplots(figsize=(COL_W, 1.85), layout="constrained")
        M = np.full((len(rows), 5), np.nan)
        for i, s in enumerate(rows):
            rs = sorted([r for r in e2 if r["cfg"]["solver"] == s], key=lambda r: r["cfg"]["lr"])
            table[s] = [dict(lr=r["cfg"]["lr"], psnr=None if r["diverged"] else round(r["final_psnr"], 2),
                             loss=None if r["diverged"] else round(r["final_loss"], 5), diverged=r["diverged"])
                        for r in rs]
            for j, r in enumerate(rs[:5]):
                M[i, j] = np.nan if r["diverged"] else r["final_psnr"]
                txt = "diverged" if r["diverged"] else f"{r['final_psnr']:.1f}"
                dark = False
                if not r["diverged"]:
                    cr, cg, cb_, _ = cmap_e2(min(max((r["final_psnr"] - 10) / 23, 0), 1))
                    dark = 0.299 * cr + 0.587 * cg + 0.114 * cb_ < 0.55
                ax.text(j, i - 0.12, txt, ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if dark else INK)
                ax.text(j, i + 0.22, f"η = {r['cfg']['lr']:.3g}".replace("e+03", "000"), ha="center",
                        va="center", fontsize=5.8, color="white" if dark else MUTED)
        cmap = cmap_e2
        cmap.set_bad("#eeeeee")
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=10, vmax=33)
        ax.set_yticks(range(len(rows)), [LABEL[s] for s in rows])
        ax.set_xticks(range(5), ["×0.1", "×0.32", "×1", "×3.2", "×10"])
        ax.set_xlabel("step size relative to the pilot's best order of magnitude")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.015, aspect=12)
        cb.set_label("PSNR after 1500 it. (dB)")
        cb.outline.set_linewidth(0.5)
        save(fig, out, "ia_e2_stepsize")
        res["E2"] = table

    # ---- E1: convergence
    e1 = summaries(os.path.join(base, "E1", "*", "summary.json"))
    if e1:
        by = defaultdict(list)
        for s in e1:
            by[s["cfg"]["solver"]].append(s)
        tgrid = np.linspace(0, 600, 601)
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.55), sharey=True, layout="constrained")
        curves = {}
        for sv in SOLVERS:
            if sv not in by:
                continue
            Lt, Lp = [], []
            for s in by[sv]:
                log = [r for r in jl(os.path.join(s["_dir"], "log.jsonl")) if r.get("loss") is not None]
                t = [r["t"] for r in log]
                Lt.append(step_interp(t, [r["loss"] for r in log], tgrid))
                Lp.append(([r["passes"] for r in log], [r["loss"] for r in log]))
            A = np.log10(np.array(Lt))
            m, sd = np.nanmean(A, 0), np.nanstd(A, 0)
            kw = dict(color=COLOR[sv], linestyle=DASH[sv], lw=1.8 if sv == "hybrid" else 1.2, label=LABEL[sv],
                      zorder=4 if sv == "hybrid" else 3)
            axes[0].plot(tgrid, 10 ** m, **kw)
            axes[0].fill_between(tgrid, 10 ** (m - sd), 10 ** (m + sd), color=COLOR[sv], alpha=0.15, lw=0, zorder=2)
            pgrid = np.unique(np.concatenate([np.asarray(p, float) for p, _ in Lp]))
            pgrid = np.linspace(0, min(max(p) for p, _ in Lp), 400)
            B = np.log10(np.array([step_interp(p, l, pgrid) for p, l in Lp]))
            axes[1].plot(pgrid, 10 ** np.nanmean(B, 0), **{k: v for k, v in kw.items() if k != "label"})
            curves[sv] = dict(t=tgrid[::10].tolist(), loss=(10 ** m)[::10].tolist())
        for ax in axes:
            ax.set_yscale("log")
        axes[0].set_xlim(0, 600)
        axes[0].set_xlabel("wall-clock time (s)")
        axes[1].set_xlabel("render passes (forward or backward)")
        axes[1].set_xscale("log")
        axes[0].set_ylabel("training loss")
        loss_ticks(axes[0])
        axes[0].set_title("(a) against time")
        axes[1].set_title("(b) against render passes")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=4)
        save(fig, out, "ia_e1_convergence")
        # table: final loss / PSNR, time to target, LM statistics
        adam_final = statistics.median(s["final_loss"] for s in by.get("adam", [])) if "adam" in by else None
        target = adam_final * 1.05 if adam_final else None
        tab = {}
        for sv in SOLVERS + TABLE_ONLY:
            if sv not in by:
                continue
            tts = []
            acc = []
            for s in by[sv]:
                log = jl(os.path.join(s["_dir"], "log.jsonl"))
                hit = [r["t"] for r in log if r.get("loss") is not None and target and r["loss"] <= target]
                tts.append(hit[0] if hit else None)
                lm = [r for r in log if r.get("phase") == "lm"]
                if lm:
                    acc.append(sum(1 for r in lm if r["accepted"]) / len(lm))
            reached = [t for t in tts if t is not None]
            tab[sv] = dict(final_loss=fmt_pm([s["final_loss"] for s in by[sv]], 4),
                           final_psnr=fmt_pm([s["final_psnr"] for s in by[sv]]),
                           iters=fmt_pm([s["iters"] for s in by[sv]], 0),
                           passes=fmt_pm([s["passes"] for s in by[sv]], 0),
                           peak_mem_mb=fmt_pm([s["peak_mem_mb"] for s in by[sv]], 0),
                           time_to_target=fmt_pm(reached, 1) if len(reached) == len(tts) else None,
                           reached=f"{len(reached)}/{len(tts)}", diverged=sum(s["diverged"] for s in by[sv]),
                           lm_accept_rate=fmt_pm(acc) if acc else None,
                           switch_it=fmt_pm([s["phase_switch"]["it"] for s in by[sv] if s.get("phase_switch")], 0)
                           if sv == "hybrid" else None)
        res["E1"] = dict(target_loss=target, solvers=tab)

    # ---- E3: curvature
    sp = glob.glob(os.path.join(base, "E3", "*", "spectrum.json"))
    if sp:
        S = json.load(open(sp[0]))
        cps = sorted(S["checkpoints"], key=int)
        last = S["checkpoints"][cps[-1]]
        groups = list(last["groups"])
        from matplotlib.lines import Line2D
        fig, axes = plt.subplots(1, 3, figsize=(COL_W + 0.9, 2.45), layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.05]))
        x = np.arange(len(groups))
        short = {"means": "mean", "scales": "scale", "rotations": "rot.", "opacities": "opac.", "colors": "colour"}
        raw = [last["groups"][g]["block_lmax"] for g in groups]
        sc = [last["groups"][g]["block_lmax_scaled"] for g in groups]
        axes[0].bar(x - 0.19, raw, 0.36, color="#2a78d6", label="JᵀJ")
        axes[0].bar(x + 0.19, sc, 0.36, color="#eb6834", label="Jacobi-scaled")
        axes[0].set_yscale("log")
        axes[0].set_xticks(x, [short.get(g, g) for g in groups], rotation=35, ha="right", rotation_mode="anchor")
        axes[0].tick_params(axis="x", length=0)
        axes[0].set_ylabel("largest eigenvalue of block")
        axes[0].set_title("(a) per-attribute λmax")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
        styles = ["-", (0, (4, 1.5)), (0, (1, 1.2))]
        for k, cp in enumerate(cps):
            c = S["checkpoints"][cp]
            lr, lj = np.array(c["lanczos_raw"]), np.array(c["lanczos_jacobi"])
            axes[1].plot(np.arange(1, len(lr) + 1), lr / lr[0], color="#2a78d6", linestyle=styles[k % 3])
            axes[1].plot(np.arange(1, len(lj) + 1), lj / lj[0], color="#eb6834", linestyle=styles[k % 3])
        axes[1].set_yscale("log")
        axes[1].set_xlabel("eigenvalue index i")
        axes[1].set_ylabel("λᵢ / λ₁")
        axes[1].set_title("(b) leading spectrum")
        axes[1].set_xlim(1, 50)  # the last Ritz values approximate the near-null end, not the leading spectrum
        lead = np.concatenate([np.array(S["checkpoints"][cp][k])[:50] / S["checkpoints"][cp][k][0]
                               for cp in cps for k in ("lanczos_raw", "lanczos_jacobi")])
        axes[1].set_ylim(lead.min() * 0.6, 1.5)
        hs = [Line2D([], [], color="#2a78d6"), Line2D([], [], color="#eb6834")] + \
             [Line2D([], [], color=MUTED, linestyle=styles[k % 3]) for k in range(len(cps))]
        axes[1].legend(hs, ["JᵀJ", "Jacobi"] + [f"it. {cp}" for cp in cps], loc="upper center",
                       bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.8)
        npz = sorted(glob.glob(os.path.join(os.path.dirname(sp[0]), f"diag_samples_it{cps[-1]}.npz")))
        if npz:
            d = np.load(npz[0])
            for gi, g in enumerate(groups):
                m = d["group"] == gi
                rho = last["groups"][g].get("spearman_adam_v_vs_gn_diag")
                axes[2].scatter(d["gn_diag"][m], d["adam_v"][m], s=3, alpha=0.45, color=GROUP_COLOR[gi], lw=0,
                                label=f"{short.get(g, g)} {rho:.2f}" if rho is not None else short.get(g, g),
                                rasterized=True)
            axes[2].set_xscale("log")
            axes[2].set_yscale("log")
            gx, gy = d["gn_diag"], d["adam_v"]
            axes[2].set_xlim(np.percentile(gx[gx > 0], 1) / 3, gx.max() * 3)
            axes[2].set_ylim(np.percentile(gy[gy > 0], 1) / 3, gy.max() * 3)
            axes[2].set_xlabel("diag(JᵀJ)")
            axes[2].set_ylabel("Adam second moment v")
            axes[2].set_title("(c) Adam v vs. diag(JᵀJ)")
            axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, markerscale=3,
                           handletextpad=0.2, columnspacing=0.8, title="Spearman ρ", title_fontsize=6.5)
        save(fig, out, "ia_e3_curvature")
        res["E3"] = {cp: dict(loss=S["checkpoints"][cp]["loss"],
                              top_raw=S["checkpoints"][cp]["lanczos_raw"][0],
                              top_jacobi=S["checkpoints"][cp]["lanczos_jacobi"][0],
                              # lambda_1 / lambda_k over the leading Ritz values; the last few Ritz values of a
                              # 60-step Lanczos run approximate the (near-null) bottom of the spectrum instead
                              k10_raw=S["checkpoints"][cp]["lanczos_raw"][0] / S["checkpoints"][cp]["lanczos_raw"][9],
                              k50_raw=S["checkpoints"][cp]["lanczos_raw"][0] / S["checkpoints"][cp]["lanczos_raw"][49],
                              k10_jacobi=S["checkpoints"][cp]["lanczos_jacobi"][0] / S["checkpoints"][cp]["lanczos_jacobi"][9],
                              k50_jacobi=S["checkpoints"][cp]["lanczos_jacobi"][0] / S["checkpoints"][cp]["lanczos_jacobi"][49],
                              block_spread_raw=max(v["block_lmax"] for v in S["checkpoints"][cp]["groups"].values())
                              / min(v["block_lmax"] for v in S["checkpoints"][cp]["groups"].values()),
                              block_spread_jacobi=max(v["block_lmax_scaled"] for v in S["checkpoints"][cp]["groups"].values())
                              / min(v["block_lmax_scaled"] for v in S["checkpoints"][cp]["groups"].values()),
                              groups=S["checkpoints"][cp]["groups"]) for cp in cps}
        res["E3"]["adam_lr"] = S["lr"]

    # ---- E4: cost scaling
    e4 = summaries(os.path.join(base, "E4", "*", "summary.json"))
    if e4:
        fig, ax = plt.subplots(figsize=(3.4, 2.45), layout="constrained")
        tab = defaultdict(dict)
        e4_label = {"adam": "first-order (SGD, Nesterov,\nRMSProp, Adam: identical)", "lbfgs": LABEL["lbfgs"],
                    "lm_group": LABEL["lm_group"], "lm_noprec": LABEL["lm_noprec"]}
        for sv in SOLVERS:
            rs = sorted([r for r in e4 if r["cfg"]["solver"] == sv], key=lambda r: r["cfg"]["n"])
            if not rs:
                continue
            ns = [r["cfg"]["n"] for r in rs]
            per = [r["time"] / max(r["iters"], 1) for r in rs]
            if sv in e4_label:  # the four first-order solvers cost the same per step; draw them once
                ax.plot(ns, per, color=COLOR[sv], linestyle="-" if sv != "lm_noprec" else DASH[sv], marker="o",
                        ms=3, label=e4_label[sv])
            for r, p in zip(rs, per):
                tab[sv][r["cfg"]["n"]] = dict(sec_per_iter=round(p, 4), passes_per_iter=round(r["passes"] / max(r["iters"], 1), 1),
                                              peak_mem_mb=round(r["peak_mem_mb"], 1))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([500, 1000, 2000, 4000, 8000], ["500", "1k", "2k", "4k", "8k"])
        ax.minorticks_off()
        ax.set_xlabel("number of Gaussians N")
        ax.set_ylabel("seconds per iteration")
        h, lab = ax.get_legend_handles_labels()
        order = [lab.index(e4_label[k]) for k in ["lm_group", "lm_noprec", "lbfgs", "adam"] if e4_label[k] in lab]
        ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left", fontsize=6.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 12)  # room for the legend above the curves
        save(fig, out, "ia_e4_cost")
        res["E4"] = tab

    # ---- E5: constraint handling
    e5 = summaries(os.path.join(base, "E5", "*", "summary.json"))
    if e5:
        res["E5"] = {r["cfg"]["mode"]: dict(psnr=None if r["diverged"] else round(r["final_psnr"], 2),
                                            loss=None if r["diverged"] else round(r["final_loss"], 5),
                                            infeasible=r["infeasible"], iters=r["iters"], time=round(r["time"], 1),
                                            diverged=r["diverged"]) for r in e5}
        fig, ax = plt.subplots(figsize=(3.4, 2.2), layout="constrained")
        mode_c = {"reparam": "#2a78d6", "project": "#eb6834", "barrier": "#1baf7a"}
        mode_d = {"reparam": "-", "project": (0, (4, 1.5)), "barrier": (0, (1, 1.2))}
        names = {"reparam": "reparameterisation", "project": "projection", "barrier": "log barrier (diverged)"}
        for mode in ["reparam", "project", "barrier"]:
            r = next((x for x in e5 if x["cfg"]["mode"] == mode), None)
            if r is None:
                continue
            log = [x for x in jl(os.path.join(r["_dir"], "log.jsonl")) if x.get("loss") is not None]
            ax.plot([x["it"] for x in log], [x["loss"] for x in log], color=mode_c[mode], linestyle=mode_d[mode],
                    label=names[mode])
            if r["diverged"]:
                ax.plot(log[-1]["it"], log[-1]["loss"], marker="x", color=mode_c[mode], ms=5, mew=1.2)
        ax.set_yscale("log")
        ax.set_xlim(0, 3000)
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("training loss")
        loss_ticks(ax)
        ax.legend(loc="upper right")
        save(fig, out, "ia_e5_constraints")


# ------------------------------------------------------------------ Tier B
def tier_b(R, out, res):
    base = os.path.join(R, "tierB")
    runs = summaries(os.path.join(base, "runs", "*", "summary.json"))
    if runs:
        res["E6"] = {}
        for r in runs:
            name = os.path.basename(r["_dir"])
            res["E6"][name] = dict(scene=r["scene"], method=r["method"], tag=r["args"].get("tag", ""),
                                   psnr=round(r["test"]["psnr"], 2), ssim=round(r["test"]["ssim"], 4),
                                   lpips=round(r["test"]["lpips"], 4), time_train=round(r["time_train"], 1),
                                   n_gaussians=r["n_gaussians"], peak_mem_mb=round(r["peak_mem_mb"], 0),
                                   lm_iters=r.get("lm_iters"), lm_accepted=r.get("lm_accepted"),
                                   lm_time=r.get("lm_time"), t_at_switch=r.get("t_at_switch"),
                                   lm_groups=r["args"].get("lm_groups"))
        scenes = sorted({r["scene"] for r in runs}, key=lambda s: ["lego", "chair", "ficus", "garden", "bicycle"].index(s))
        ncol = 3
        nrow = (len(scenes) + 1 + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(COL_W, 1.75 * nrow + 0.1), layout="constrained", squeeze=False)
        flat = axes.ravel()
        mc = {"adam": COLOR["adam"], "rmsprop": COLOR["rmsprop"], "hybrid": COLOR["hybrid"]}
        names6 = {"adam": "Adam", "rmsprop": "RMSProp", "hybrid": "Adam → LM (ours)"}
        for ax, sc in zip(flat, scenes):
            ref = next((r for r in runs if r["scene"] == sc and r["method"] == "adam" and not r["args"].get("tag")), None)
            tmin = 0.75 * ref["t_at_switch"] if ref and ref.get("t_at_switch") else 0.0  # zoom on the last third
            ys = []
            for r in runs:
                if r["scene"] != sc or r["args"].get("tag"):
                    continue
                log = [x for x in jl(os.path.join(r["_dir"], "log.jsonl")) if x.get("test_psnr") is not None]
                if r["method"] == "hybrid":
                    log = [x for x in log if x.get("phase") in ("adam", "lm", "lm_end")]
                log = [x for x in log if x["t"] >= tmin]
                ys += [x["test_psnr"] for x in log]
                ax.plot([x["t"] / 60 for x in log], [x["test_psnr"] for x in log], color=mc[r["method"]],
                        linestyle=DASH.get(r["method"], "-"), lw=1.6 if r["method"] == "hybrid" else 1.1,
                        label=names6[r["method"]], marker="o", ms=2, zorder=4 if r["method"] == "hybrid" else 3)
            if ref and ref.get("t_at_switch"):
                ax.axvline(ref["t_at_switch"] / 60, color=MUTED, lw=0.6, ls=":", zorder=1)
            if ys:
                pad = 0.08 * (max(ys) - min(ys) + 1e-3)
                ax.set_ylim(min(ys) - pad, max(ys) + pad)
            ax.set_title(sc, fontweight="normal")
            ax.locator_params(axis="x", nbins=4)
            ax.locator_params(axis="y", nbins=4)
        for ax in axes[:, 0]:
            ax.set_ylabel("test PSNR (dB)")
        for ax in axes[-1, :]:
            ax.set_xlabel("training time (min)")
        handles = {}
        for ax in flat[:len(scenes)]:
            for h, lab in zip(*ax.get_legend_handles_labels()):
                handles.setdefault(lab, h)
        handles["Adam → LM switch (step 20k)"] = Line2D([], [], color=MUTED, lw=0.6, ls=":")
        for ax in flat[len(scenes):]:
            ax.axis("off")
        flat[-1].legend(handles.values(), handles.keys(), loc="center", fontsize=6.8)
        save(fig, out, "ia_e6_curves")
    pil = [r for r in runs if r["scene"] == "lego" and r["args"].get("tag", "").startswith(("_pilot_", "_fb_"))]
    # the pilots started from the checkpoint of an earlier run of the same lego Adam baseline (tag _pilotbase)
    adam = next((r for r in runs if r["scene"] == "lego" and r["method"] == "adam" and r["args"].get("tag") == "_pilotbase"),
                None)
    if pil and adam:
        fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.4), sharey=False, layout="constrained")
        A = [x for x in jl(os.path.join(adam["_dir"], "log.jsonl")) if x.get("test_psnr") is not None and x["step"] >= 20000]
        t0 = A[0]["t"]
        spec = {"_pilot_all": ("every attribute, 4 views", "#2a78d6", (0, (1, 1.2))),
                "_pilot_color": ("colour only, 4 views", "#eb6834", (0, (1, 1.2))),
                "_fb_all": ("every attribute, all views", "#2a78d6", "-"),
                "_fb_appearance": ("colour + opacity, all views", "#1baf7a", "-"),
                "_fb_color": ("colour only, all views", "#eb6834", "-")}
        for ax, keys in zip(axes, [["_pilot_all", "_pilot_color", "_fb_all", "_fb_appearance", "_fb_color"],
                                   ["_fb_all", "_fb_appearance", "_fb_color"]]):
            ax.plot([x["t"] - t0 for x in A], [x["test_psnr"] for x in A], color=COLOR["adam"], lw=1.4,
                    label="Adam, steps 20k → 30k", zorder=4)
            for r in pil:
                tag = r["args"]["tag"]
                if tag not in keys:
                    continue
                L = [x for x in jl(os.path.join(r["_dir"], "log.jsonl")) if x.get("test_psnr") is not None]
                lab, c, ls = spec[tag]
                ax.plot([x["t"] - t0 for x in L], [x["test_psnr"] for x in L], color=c, linestyle=ls, label=lab)
            ax.set_xlabel("seconds after the 20k checkpoint")
            ax.set_xlim(0, 310)
        axes[0].set_ylabel("test PSNR (dB)")
        axes[0].set_title("(a) all variants")
        axes[1].set_title("(b) full batch, zoomed")
        axes[1].set_ylim(35.62, 36.15)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=3)
        save(fig, out, "ia_e6_lm_pilots")
    fd = glob.glob(os.path.join(base, "fd_check", "*.json"))
    if fd:
        F = json.load(open(fd[0]))
        groups = ["means", "scales", "quats", "opacities", "sh0", "shN"]
        kinds = [("image", "rendered image", "#2a78d6"), ("residual", "residuals, float32", "#eb6834"),
                 ("residual_fp64", "residuals, float64", "#1baf7a")]
        kinds = [k for k in kinds if f"{k[0]}/means/h=0.001" in F]
        fig, ax = plt.subplots(figsize=(COL_W * 0.72, 2.2), layout="constrained")
        x = np.arange(len(groups))
        w = 0.8 / len(kinds)
        for i, (k, lab, c) in enumerate(kinds):
            ax.bar(x + (i - (len(kinds) - 1) / 2) * w, [F[f"{k}/{g}/h=0.001"] for g in groups], w * 0.9, color=c,
                   label=lab, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 2.0)
        ax.set_xticks(x, ["mean", "scale", "rotation", "opacity", "SH dc", "SH rest"])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("relative adjoint mismatch")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        save(fig, out, "ia_fd_check")
        res["fd_check"] = F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default=None, help="run-log folder (default: paths.runs(); the report's logs: results)")
    a = ap.parse_args()
    R = a.runs or paths.runs()
    out = a.out or os.path.join(R, "figures")
    os.makedirs(out, exist_ok=True)
    res = {}
    tier_a(R, out, res)
    tier_b(R, out, res)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", out, sorted(res))


if __name__ == "__main__":
    main()
