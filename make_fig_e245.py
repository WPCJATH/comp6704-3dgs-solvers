"""One combined figure for the report's stability / cost / constraints paragraph (Section 5.4):
(a) E2 step-size sensitivity, (b) E4 cost per iteration, (c) E5 constraint handling.
Same run logs, colours and print size as make_figures.py.

    python make_fig_e245.py [--runs results] [--out DIR]     -> ia_e245_combined.png / .pdf
"""
import argparse
import os

import make_figures as mf  # sets rcParams; main() is guarded
from make_figures import COLOR, DASH, INK, LABEL, MUTED, jl, plt, np, summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = os.path.join(a.runs, "tierA")
    out = a.out or os.path.join(a.runs, "figures")
    os.makedirs(out, exist_ok=True)

    fig = plt.figure(figsize=(mf.COL_W, 2.05), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) E2: PSNR after 1500 iterations at five step sizes around each solver's best order of magnitude
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cmap.set_bad("#eeeeee")
    e2 = summaries(os.path.join(base, "E2", "*", "summary.json"))
    rows = ["sgd", "momentum", "rmsprop", "adam"]
    short = {"sgd": "SGD", "momentum": "Nesterov", "rmsprop": "RMSProp", "adam": "Adam"}
    M = np.full((4, 5), np.nan)
    for i, s in enumerate(rows):
        rs = sorted([r for r in e2 if r["cfg"]["solver"] == s], key=lambda r: r["cfg"]["lr"])[:5]
        for j, r in enumerate(rs):
            if not r["diverged"]:
                M[i, j] = r["final_psnr"]
            dark = False
            if not r["diverged"]:
                cr, cg, cb, _ = cmap(min(max((r["final_psnr"] - 10) / 23, 0), 1))
                dark = 0.299 * cr + 0.587 * cg + 0.114 * cb < 0.55
            axA.text(j, i, "div." if r["diverged"] else f"{r['final_psnr']:.1f}", ha="center", va="center",
                     fontsize=5.4, fontweight="bold", color="white" if dark else INK)
    axA.imshow(M, cmap=cmap, aspect="auto", vmin=10, vmax=33)
    axA.set_yticks(range(4), [short[s] for s in rows], fontsize=6.2)
    axA.set_xticks(range(5), ["0.1", "0.32", "1", "3.2", "10"])
    axA.set_xlabel("step size (× best order)")
    axA.tick_params(length=0)
    for sp in axA.spines.values():
        sp.set_visible(False)
    axA.grid(False)
    axA.set_title("(a) Step size: PSNR (dB)")

    # (b) E4: seconds per iteration against the number of Gaussians
    e4 = summaries(os.path.join(base, "E4", "*", "summary.json"))
    lab = {"adam": "first-order (all four)", "lbfgs": "L-BFGS", "lm_noprec": "LM, no scaling", "lm_group": "LM, per-attribute"}
    for sv in ["lm_group", "lm_noprec", "lbfgs", "adam"]:
        rs = sorted([r for r in e4 if r["cfg"]["solver"] == sv], key=lambda r: r["cfg"]["n"])
        if rs:
            axB.plot([r["cfg"]["n"] for r in rs], [r["time"] / max(r["iters"], 1) for r in rs], color=COLOR[sv],
                     linestyle="-" if sv != "lm_noprec" else DASH[sv], marker="o", ms=2.5, label=lab[sv])
    axB.set_xscale("log"); axB.set_yscale("log"); axB.minorticks_off()
    axB.set_xticks([500, 2000, 8000], ["500", "2k", "8k"])
    axB.set_xlabel("number of Gaussians N")
    axB.set_ylabel("seconds per iteration")
    axB.set_ylim(top=axB.get_ylim()[1] * 15)
    axB.legend(loc="upper left", fontsize=5.6, handlelength=1.8)
    axB.set_title("(b) Cost per iteration")

    # (c) E5: Adam's training loss with three ways of handling the constraints
    e5 = summaries(os.path.join(base, "E5", "*", "summary.json"))
    mode_c = {"reparam": "#2a78d6", "project": "#eb6834", "barrier": "#1baf7a"}
    mode_d = {"reparam": "-", "project": (0, (4, 1.5)), "barrier": (0, (1, 1.2))}
    names = {"reparam": "reparameterisation", "project": "projection", "barrier": "log barrier (diverged)"}
    for mode in ["reparam", "project", "barrier"]:
        r = next((x for x in e5 if x["cfg"]["mode"] == mode), None)
        if r is None:
            continue
        log = [x for x in jl(os.path.join(r["_dir"], "log.jsonl")) if x.get("loss") is not None]
        axC.plot([x["it"] for x in log], [x["loss"] for x in log], color=mode_c[mode], linestyle=mode_d[mode], label=names[mode])
        if r["diverged"]:
            axC.plot(log[-1]["it"], log[-1]["loss"], marker="x", color=mode_c[mode], ms=4, mew=1.1)
    axC.set_yscale("log"); axC.set_xlim(0, 3000)
    axC.set_xlabel("Adam iteration"); axC.set_ylabel("training loss")
    mf.loss_ticks(axC)
    axC.legend(loc="upper right", fontsize=5.6, handlelength=1.8)
    axC.set_title("(c) Constraints")

    mf.save(fig, out, "ia_e245_combined")
    print("wrote", os.path.join(out, "ia_e245_combined.png"))


if __name__ == "__main__":
    main()
