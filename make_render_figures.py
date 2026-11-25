"""Qualitative figures: rendered images next to the ground truth.

    python make_render_figures.py [--runs DIR] [--out DIR]     (default: <runs>/figures)

Needs the datasets (ground-truth images). The Tier B test renders are large and are not shipped in
results/, so the Tier B figure needs your own runs (or the shipped figure in results/figures).

Tier A  (needs `python tierA/grid.py --exp renders`):
    ia_renders_2d.png      ground truth, the random initialisation and each solver's final image after 600 s
Tier B  (uses the two test views every train3d.py run saves under runs/tierB/runs/<run>/renders):
    ia_renders_3d.png      per scene: ground truth, Adam, Adam -> LM (ours), RMSProp where it was run,
                           with a zoomed crop and the |Adam - hybrid| difference
    ia_renders_3d_<scene>.png        one row per scene
    ia_renders_3d_<scene>_page.png   two rows per scene (full frames, then enlarged crops), for a web page
PSNR printed on each panel is computed on that image alone, so it differs from the test-set means.
"""
import argparse
import glob
import json
import os
import re
import sys

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ia3dgs import paths  # noqa: E402
from ia3dgs.data3d import SCENES  # noqa: E402

COL_W = 5.5
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
                     "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
LABEL = {"sgd": "SGD", "momentum": "Nesterov", "rmsprop": "RMSProp", "adam": "Adam", "lbfgs": "L-BFGS",
         "lm_group": "LM (per-attr. Jacobi)", "hybrid": "Adam → LM (ours)"}
CROP_COLOR = "#e34948"


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return -10 * np.log10(max(mse, 1e-12))


def area_resize(img, factor):
    if factor == 1:
        return img
    h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
    x = img[:h, :w].astype(np.float64).reshape(h // factor, factor, w // factor, factor, -1).mean((1, 3))
    return x.round().clip(0, 255).astype(np.uint8)


def gt_view(scene, idx):
    """The test image exactly as ia3dgs.data3d feeds it to the trainer."""
    sub, kind = SCENES[scene]
    root = os.path.join(paths.datasets(), sub, scene)
    if kind == "blender":
        meta = json.load(open(os.path.join(root, "transforms_test.json")))
        rgba = imageio.imread(os.path.join(root, meta["frames"][idx]["file_path"].lstrip("./") + ".png"))
        rgba = rgba.astype(np.float32) / 255
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        return area_resize((rgb * 255).round().astype(np.uint8), 2)
    names = sorted(os.listdir(os.path.join(root, "images_4")))
    test = [n for i, n in enumerate(names) if i % 8 == 0]
    return imageio.imread(os.path.join(root, "images_4", test[idx]))[..., :3]


def worst_crop(a, b, size):
    """Top-left corner of the size x size window where |a - b| is largest (coarse search)."""
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)
    h, w = d.shape
    ii = np.cumsum(np.cumsum(np.pad(d, ((1, 0), (1, 0))), 0), 1)
    best, arg = -1, (0, 0)
    step = max(4, size // 8)
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            s = ii[y + size, x + size] - ii[y, x + size] - ii[y + size, x] + ii[y, x]
            if s > best:
                best, arg = s, (y, x)
    return arg


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- Tier A
def tier_a(out):
    R = os.path.join(paths.runs(), "tierA", "renders")
    runs = {}
    for f in glob.glob(os.path.join(R, "*_s0", "summary.json")):
        s = json.load(open(f))
        if os.path.exists(os.path.join(os.path.dirname(f), "final.png")):
            runs[s["cfg"]["solver"]] = (os.path.dirname(f), s)
    if not runs:
        print("tier A: no renders yet (python tierA/grid.py --exp renders)")
        return
    from ia3dgs.model2d import load_image
    any_cfg = next(iter(runs.values()))[1]["cfg"]
    gt = load_image(os.path.join(paths.datasets(), "kodak", any_cfg["image"] + ".png"), any_cfg["size"], "cpu")
    gt = (gt.numpy() * 255).round().astype(np.uint8)
    order = [k for k in ["sgd", "momentum", "rmsprop", "adam", "lbfgs", "lm_group", "hybrid"] if k in runs]
    init = imageio.imread(os.path.join(runs[order[0]][0], "init.png"))
    panels = [("Target (kodim23)", gt, None), (f"Random init · {psnr(init, gt):.1f} dB", init, None)]
    short = {**LABEL, "lm_group": "LM", "hybrid": "Adam → LM (ours)"}
    for k in order:
        img = imageio.imread(os.path.join(runs[k][0], "final.png"))
        panels.append((f"{short[k]} · {psnr(img, gt):.2f} dB", img, k))
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL_W * 1.25, 1.42 * nrow), layout="constrained", squeeze=False)
    for ax in axes.flat:
        off(ax)
    for ax, (t, img, k) in zip(axes.flat, panels):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(t, fontweight="bold" if k == "hybrid" else "normal", fontsize=6.8)
    if "adam" in runs and "hybrid" in runs and len(panels) < nrow * ncol:
        a = imageio.imread(os.path.join(runs["adam"][0], "final.png")).astype(np.float32)
        h = imageio.imread(os.path.join(runs["hybrid"][0], "final.png")).astype(np.float32)
        d = np.abs(a - h).mean(-1)
        ax = axes.flat[len(panels)]
        ax.imshow(d, cmap="magma", vmin=0, vmax=max(8.0, float(np.percentile(d, 99.9))), interpolation="nearest")
        ax.set_title("|Adam − ours|, 0–255", fontsize=6.8)
    fig.suptitle("Tier A · 2000 Gaussians, 600 s on an RTX 3090, seed 0 · LM = per-attribute Jacobi",
                 fontsize=7.5, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"ia_renders_2d.{ext}"))
    plt.close(fig)
    print("wrote ia_renders_2d")


# ---------------------------------------------------------------- Tier B
def tier_b(out):
    R = os.path.join(paths.runs(), "tierB", "runs")
    rows = []
    for scene in ["lego", "chair", "ficus", "garden", "bicycle"]:
        files = {}
        for m in ["adam", "hybrid", "rmsprop"]:
            d = os.path.join(R, f"{scene}_{m}_s0", "renders")
            if os.path.isdir(d):
                files[m] = sorted(glob.glob(os.path.join(d, "*_test*.png")))
        if "adam" not in files or "hybrid" not in files:
            continue
        # the second saved view is the middle of the evaluation curve's views
        path_a = files["adam"][-1]
        idx = int(re.search(r"_test(\d+)\.png$", path_a).group(1))
        imgs = {m: imageio.imread([p for p in fs if p.endswith(f"_test{idx:03d}.png")][0])[..., :3]
                for m, fs in files.items()}
        gt = gt_view(scene, idx)
        rows.append((scene, idx, gt, imgs))
    if not rows:
        print("tier B: no renders found")
        return
    methods = ["adam", "hybrid", "rmsprop"]
    names = {"adam": "Adam (30k)", "hybrid": "Adam → LM (ours)", "rmsprop": "RMSProp (30k)"}

    def draw(axrow, scene, idx, gt, imgs, big=False, fixed=False):
        """fixed=True keeps one column per method (blank if not run) so the difference map is always last."""
        size = max(48, min(gt.shape[:2]) // 4)
        y, x = worst_crop(imgs["adam"], gt, size)
        h, w = gt.shape[:2]
        # put the zoomed inset in the corner farthest from the crop so it never hides it
        ix = 0.0 if x + size / 2 > w / 2 else 0.64
        iy = 0.64 if y + size / 2 > h / 2 else 0.0  # inset_axes y runs upwards
        cols = [("Ground truth", gt)] + [(names[m], imgs.get(m)) for m in methods if fixed or m in imgs]
        for ax in axrow:
            off(ax)
        for j, (t, img) in enumerate(cols):
            ax = axrow[j]
            if img is None:
                ax.text(0.5, 0.5, f"{t}\nnot run on this scene", ha="center", va="center", fontsize=6, color="#777777",
                        transform=ax.transAxes)
                continue
            ax.imshow(img)
            ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, ec=CROP_COLOR, lw=0.8))
            sub = "" if j == 0 else f" · {psnr(img, gt):.2f} dB"
            ax.set_title(t + sub, fontsize=7 if big else 6.2)
            ins = ax.inset_axes([ix, iy, 0.36, 0.36])
            ins.imshow(img[y:y + size, x:x + size], interpolation="nearest")
            off(ins)
            for s in ins.spines.values():
                s.set_visible(True)
                s.set_edgecolor(CROP_COLOR)
                s.set_linewidth(0.8)
        ax = axrow[len(cols)]
        diff = np.abs(imgs["adam"].astype(np.float32) - imgs["hybrid"].astype(np.float32)).mean(-1)
        im = ax.imshow(diff, cmap="magma", vmin=0, vmax=max(8.0, float(np.percentile(diff, 99.9))))
        ax.set_title("|Adam − ours|, 0–255", fontsize=7 if big else 6.2)
        axrow[0].set_ylabel(f"{scene} · test view {idx}", fontsize=6.8)
        axrow[0].yaxis.set_visible(True)
        return im

    ncol = 5
    fig, axes = plt.subplots(len(rows), ncol, figsize=(COL_W * 1.3, 1.45 * len(rows) + 0.1), layout="constrained",
                             squeeze=False)
    for axrow, (scene, idx, gt, imgs) in zip(axes, rows):
        draw(axrow, scene, idx, gt, imgs, fixed=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"ia_renders_3d.{ext}"))
    plt.close(fig)
    for scene, idx, gt, imgs in rows:
        n = 2 + sum(m in imgs for m in methods)
        fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.5 * gt.shape[0] / gt.shape[1] + 0.3), layout="constrained")
        draw(axes, scene, idx, gt, imgs, big=True)
        fig.savefig(os.path.join(out, f"ia_renders_3d_{scene}.png"), dpi=200)
        plt.close(fig)
        page_panel(out, scene, idx, gt, imgs, methods, names)
    print("wrote ia_renders_3d (+ per scene):", ", ".join(r[0] for r in rows))


def page_panel(out, scene, idx, gt, imgs, methods, names):
    """Two rows for a web page: full frames on top, the zoomed crop of each below at full panel size."""
    size = max(48, min(gt.shape[:2]) // 4)
    y, x = worst_crop(imgs["adam"], gt, size)
    cols = [("Ground truth", gt)] + [(names[m], imgs[m]) for m in methods if m in imgs]
    diff = np.abs(imgs["adam"].astype(np.float32) - imgs["hybrid"].astype(np.float32)).mean(-1)
    n = len(cols) + 1
    h, w = gt.shape[:2]
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 2.2 * h / w + 2.2 + 0.4), layout="constrained",
                             gridspec_kw={"height_ratios": [h / w, 1]})
    for ax in axes.flat:
        off(ax)
    vmax = max(8.0, float(np.percentile(diff, 99.9)))
    for j, (t, img) in enumerate(cols):
        axes[0, j].imshow(img)
        axes[0, j].add_patch(plt.Rectangle((x, y), size, size, fill=False, ec=CROP_COLOR, lw=1.0))
        axes[0, j].set_title(t if j == 0 else f"{t} · {psnr(img, gt):.2f} dB", fontsize=8)
        axes[1, j].imshow(img[y:y + size, x:x + size], interpolation="lanczos")
    axes[0, -1].imshow(diff, cmap="magma", vmin=0, vmax=vmax)
    axes[0, -1].set_title("|Adam − ours|, 0–255", fontsize=8)
    axes[1, -1].imshow(diff[y:y + size, x:x + size], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[1, 0].set_ylabel("zoom (red box)", fontsize=7)
    axes[1, 0].yaxis.set_visible(True)
    fig.savefig(os.path.join(out, f"ia_renders_3d_{scene}_page.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs", default=None, help="run-log folder (default: paths.runs())")
    a = ap.parse_args()
    if a.runs:
        os.environ["IA3DGS_RUNS"] = os.path.abspath(a.runs)
    out = a.out or os.path.join(paths.runs(), "figures")
    os.makedirs(out, exist_ok=True)
    tier_b(out)
    tier_a(out)


if __name__ == "__main__":
    main()
