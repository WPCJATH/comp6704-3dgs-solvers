"""Scene loaders for Tier B: NeRF-Synthetic (Blender JSON) and Mip-NeRF 360 (COLMAP binary).

Images are decoded once and kept on the GPU as uint8. Cameras use the OpenCV convention
(x right, y down, z forward); viewmats are world-to-camera 4x4 matrices.
"""
import json
import os
import struct
from dataclasses import dataclass

import imageio.v2 as imageio
import numpy as np
import torch


@dataclass
class Split:
    images: torch.Tensor   # [N, H, W, 3] uint8
    viewmats: torch.Tensor  # [N, 4, 4]
    Ks: torch.Tensor        # [N, 3, 3]
    names: list


@dataclass
class Scene:
    train: Split
    test: Split
    init_xyz: torch.Tensor   # [M, 3]
    init_rgb: torch.Tensor   # [M, 3] in [0, 1]
    scene_scale: float
    background: torch.Tensor  # [3]
    random_init: bool


def _scene_scale(c2w):
    centers = c2w[:, :3, 3]
    return float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)


def _resize(img, factor):
    if factor == 1:
        return img
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
    t = torch.nn.functional.interpolate(t, scale_factor=1 / factor, mode="area")
    return t[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()


def load_blender(root, factor=2, device="cuda", n_init=100_000, seed=0):
    def split(name):
        meta = json.load(open(os.path.join(root, f"transforms_{name}.json")))
        imgs, c2ws, names = [], [], []
        for fr in meta["frames"]:
            p = os.path.join(root, fr["file_path"].lstrip("./") + ".png")
            rgba = imageio.imread(p).astype(np.float32) / 255
            rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])  # white background
            imgs.append(_resize((rgb * 255).round().astype(np.uint8), factor))
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            c2w[:3, 1:3] *= -1  # OpenGL -> OpenCV
            c2ws.append(c2w)
            names.append(os.path.basename(p))
        h, w = imgs[0].shape[:2]
        f = 0.5 * w / np.tan(0.5 * meta["camera_angle_x"])
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
        c2ws = np.stack(c2ws)
        return Split(torch.from_numpy(np.stack(imgs)).to(device),
                     torch.from_numpy(np.linalg.inv(c2ws)).float().to(device),
                     torch.from_numpy(np.repeat(K[None], len(imgs), 0)).to(device), names), c2ws

    train, c2w_train = split("train")
    test, _ = split("test")
    g = np.random.default_rng(seed)
    xyz = g.random((n_init, 3)) * 2.6 - 1.3  # 3DGS's random init for synthetic scenes
    rgb = g.random((n_init, 3))
    return Scene(train, test, torch.from_numpy(xyz).float().to(device), torch.from_numpy(rgb).float().to(device),
                 _scene_scale(c2w_train), torch.ones(3, device=device), True)


# ---------------- COLMAP binary ----------------
_CAM_NPARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}


def _read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            cid, model, w, h = _read(f, "iiQQ")
            params = _read(f, "d" * _CAM_NPARAMS[model])
            cams[cid] = (model, w, h, params)
    return cams


def read_images(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = _read(f, "Q")
            f.seek(24 * npts, 1)
            imgs[iid] = (name.decode(), np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid)
    return imgs


def read_points3d(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        (n,) = _read(f, "Q")
        for _ in range(n):
            _, x, y, z, r, g, b, _, tl = _read(f, "QdddBBBdQ")
            f.seek(8 * tl, 1)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255


def _qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def load_colmap(root, factor=4, device="cuda", llffhold=8):
    """Load a Mip-NeRF 360 scene from its COLMAP model, with images downsampled by `factor`;
    every `llffhold`-th view is a test view (the standard split)."""
    sp = os.path.join(root, "sparse", "0")
    cams = read_cameras(os.path.join(sp, "cameras.bin"))
    ims = sorted(read_images(os.path.join(sp, "images.bin")).values(), key=lambda t: t[0])
    imgdir = os.path.join(root, f"images_{factor}")
    data = {"train": ([], [], [], []), "test": ([], [], [], [])}
    c2w_train = []
    for i, (name, q, t, cid) in enumerate(ims):
        model, w0, h0, params = cams[cid]
        img = imageio.imread(os.path.join(imgdir, name))[..., :3]
        h, w = img.shape[:2]
        sx, sy = w / w0, h / h0
        if model == 0 or model == 2:  # SIMPLE_PINHOLE / SIMPLE_RADIAL (distortion ignored; images are undistorted)
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:  # PINHOLE / OPENCV ...
            fx, fy, cx, cy = params[:4]
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=np.float32)
        w2c = np.eye(4)
        w2c[:3, :3] = _qvec2rot(q)
        w2c[:3, 3] = t
        key = "test" if i % llffhold == 0 else "train"
        for lst, v in zip(data[key], (img, w2c, K, name)):
            lst.append(v)
        if key == "train":
            c2w_train.append(np.linalg.inv(w2c))
    xyz, rgb = read_points3d(os.path.join(sp, "points3D.bin"))

    def mk(d):
        return Split(torch.from_numpy(np.stack(d[0])).to(device), torch.from_numpy(np.stack(d[1])).float().to(device),
                     torch.from_numpy(np.stack(d[2])).to(device), d[3])

    return Scene(mk(data["train"]), mk(data["test"]), torch.from_numpy(xyz).to(device),
                 torch.from_numpy(rgb).to(device), _scene_scale(np.stack(c2w_train)), torch.zeros(3, device=device), False)


SCENES = {
    "lego": ("nerf_synthetic", "blender"), "chair": ("nerf_synthetic", "blender"), "ficus": ("nerf_synthetic", "blender"),
    "garden": ("mipnerf360", "colmap"), "bicycle": ("mipnerf360", "colmap"),
}


def load_scene(datasets_root, name, device="cuda"):
    """Load a scene by name (lego, chair, ficus, garden, bicycle) from the datasets folder."""
    sub, kind = SCENES[name]
    root = os.path.join(datasets_root, sub, name)
    return load_blender(root, device=device) if kind == "blender" else load_colmap(root, device=device)
