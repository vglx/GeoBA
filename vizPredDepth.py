#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read two PNG depth maps and visualize as point clouds (shared pose), with
STRICT size alignment before any projection (avoid intrinsic-size mismatch).

- png_um: depth stored in micrometers (µm)  -> converted to mm (float32).
- png_unit01: depth stored as uint16 with fixed unit_mm per LSB (default 0.1 mm/LSB) -> mm.
- Both clouds are projected using the SAME intrinsics (fx, fy, cx, cy) that correspond
  to the FINAL aligned size used for BOTH maps.
- Option --align_to controls which one is the reference size ("um" or "unit").
- Visualization and PLY export both use the EXACT aligned arrays (not the originals).

Units:
- Depth arrays are in millimeters (mm).
- Back-projected points are in millimeters (mm).
- If you supply T_wc, its translation must also be in millimeters.

Extras:
- Optional export: quantize the µm depth to a 16-bit PNG at unit_mm (e.g., 0.1 mm/LSB).
- RMSE/MAE are computed on overlapping valid pixels AFTER alignment.
"""

import argparse
import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt


# ---------- IO helpers ----------

def load_png16(path: str) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image: {path}, got shape={img.shape}")
    if img.dtype != np.uint16:
        img = img.astype(np.uint16)
    return img


def depth_um_png_to_mm(path: str) -> np.ndarray:
    """PNG whose stored integer is micrometers (µm) → mm float32. 0/neg/nonfinite -> NaN."""
    img = imageio.imread(path)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image: {path}, got shape={img.shape}")
    depth_um = img.astype(np.float32)
    depth_mm = depth_um * 1e-3
    depth_mm = np.where(np.isfinite(depth_mm) & (depth_mm > 0), depth_mm, np.nan).astype(np.float32)
    return depth_mm


def depth_unit_png_to_mm(path: str, unit_mm: float = 0.1) -> np.ndarray:
    """PNG stored as uint16 with fixed mm-per-LSB (default 0.1 mm). 0 -> NaN."""
    img = load_png16(path)
    depth_mm = img.astype(np.float32) * float(unit_mm)
    depth_mm = np.where(img == 0, np.nan, depth_mm).astype(np.float32)
    return depth_mm


def depth_mm_to_png16(depth_mm: np.ndarray, unit_mm: float = 0.1) -> np.ndarray:
    """mm float -> uint16 with given unit_mm per LSB; 0 encodes invalid."""
    if depth_mm.dtype not in (np.float32, np.float64):
        depth_mm = depth_mm.astype(np.float32)
    invalid = ~np.isfinite(depth_mm) | (depth_mm <= 0)
    q = np.rint(depth_mm / float(unit_mm)).astype(np.int64)
    q[invalid] = 0
    q = np.clip(q, 0, 65535).astype(np.uint16)
    return q


# ---------- nearest-neighbor resize (keeps NaN) ----------

def resize_nn(img: np.ndarray, out_hw):
    """Nearest-neighbor resize for 2D float arrays (keeps NaNs)."""
    H, W = img.shape
    Ho, Wo = int(out_hw[0]), int(out_hw[1])
    ys = (np.arange(Ho) * (H / Ho)).astype(np.int64)
    xs = (np.arange(Wo) * (W / Wo)).astype(np.int64)
    ys = np.clip(ys, 0, H - 1)
    xs = np.clip(xs, 0, W - 1)
    return img[ys[:, None], xs[None, :]]


# ---------- metrics ----------

def compute_rmse_mae(a_mm: np.ndarray, b_mm: np.ndarray):
    """Compute RMSE/MAE on overlapping valid pixels (finite & >0)."""
    m = np.isfinite(a_mm) & np.isfinite(b_mm) & (a_mm > 0) & (b_mm > 0)
    n = int(m.sum())
    if n == 0:
        return np.nan, np.nan, 0
    d = a_mm[m] - b_mm[m]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    return rmse, mae, n


# ---------- geometry ----------

def backproject_mm(depth_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 1):
    """Back-project depth (mm) into point cloud (mm)."""
    H, W = depth_mm.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_mm[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), np.float32), valid
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    pc = np.stack([x, y, z], axis=-1).astype(np.float32)
    pc = pc[valid]
    return pc, valid


def load_pose_txt(path: str) -> np.ndarray:
    """Return 4x4 T_wc (camera->world). Translation must be in millimeters."""
    if path is None or not os.path.isfile(path):
        return np.eye(4, dtype=np.float32)
    M = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [float(x) for x in line.strip().split() if x.strip()]
            if len(parts) == 0:
                continue
            M.append(parts)
    M = np.array(M, dtype=np.float32)
    if M.shape != (4, 4):
        raise ValueError(f"Pose file must be 4x4, got {M.shape}")
    return M


def transform_points(T_wc: np.ndarray, pts_c: np.ndarray) -> np.ndarray:
    if pts_c.shape[0] == 0:
        return pts_c
    R = T_wc[:3, :3].astype(np.float32)
    t = T_wc[:3, 3].astype(np.float32)
    return (pts_c @ R.T) + t


def save_ply(path: str, pts: np.ndarray, rgb: np.ndarray = None):
    n = pts.shape[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
        else:
            for p, c in zip(pts, rgb):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ---------- viz helpers ----------

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        "Read two PNG depth maps (µm and unit-mm/LSB), align SIZE first, then project with SAME intrinsics."
    )
    ap.add_argument("--png_um", required=True, help="Depth PNG in micrometers (µm).")
    ap.add_argument("--png_unit01", required=True, help="Depth PNG with fixed mm/LSB unit (e.g., 0.1).")
    ap.add_argument("--unit_mm", type=float, default=0.1, help="mm per LSB for --png_unit01 (default: 0.1).")

    # Intrinsics correspond to the FINAL aligned size used for both maps
    ap.add_argument("--fx", type=float, required=True, help="Focal length fx (pixels).")
    ap.add_argument("--fy", type=float, required=True, help="Focal length fy (pixels).")
    ap.add_argument("--cx", type=float, required=True, help="Principal point cx (pixels).")
    ap.add_argument("--cy", type=float, required=True, help="Principal point cy (pixels).")

    ap.add_argument("--pose_txt", default=None, help="4x4 T_wc (camera->world), translation in mm. If omitted, identity.")
    ap.add_argument("--stride", type=int, default=2, help="Subsample stride for visualization.")
    ap.add_argument("--save_ply", default=None, help="Optional dir to save um.ply and unit.ply")
    ap.add_argument("--export_from_um", action="store_true", help="Also export the µm PNG as 16-bit PNG at unit_mm.")
    ap.add_argument("--align_to", choices=["um", "unit"], default="um",
                    help="Resize the OTHER depth to match this one's size (NN). All projection uses the aligned size.")

    args = ap.parse_args()

    # Load raw depths (mm)
    depth_um_mm_raw   = depth_um_png_to_mm(args.png_um)
    depth_unit_mm_raw = depth_unit_png_to_mm(args.png_unit01, unit_mm=args.unit_mm)

    # Optional export: write quantized PNG from the µm input
    if args.export_from_um:
        out_png = os.path.splitext(args.png_um)[0] + f"_quant_{args.unit_mm:.3f}mm.png"
        imageio.imwrite(out_png, depth_mm_to_png16(depth_um_mm_raw, unit_mm=args.unit_mm))
        print(f"[INFO] Saved quantized PNG from µm input -> {out_png}")

    # Strict size alignment BEFORE any projection
    H_um,   W_um   = depth_um_mm_raw.shape
    H_unit, W_unit = depth_unit_mm_raw.shape

    if args.align_to == "um":
        ref_H, ref_W = H_um, W_um
        depth_um_mm   = depth_um_mm_raw
        if (H_unit, W_unit) != (ref_H, ref_W):
            print(f"[INFO] Resizing UNIT to UM size: unit {depth_unit_mm_raw.shape} -> {ref_H, ref_W} (NN)")
            depth_unit_mm = resize_nn(depth_unit_mm_raw, (ref_H, ref_W))
        else:
            depth_unit_mm = depth_unit_mm_raw
    else:  # align_to == "unit"
        ref_H, ref_W = H_unit, W_unit
        depth_unit_mm = depth_unit_mm_raw
        if (H_um, W_um) != (ref_H, ref_W):
            print(f"[INFO] Resizing UM to UNIT size: um {depth_um_mm_raw.shape} -> {ref_H, ref_W} (NN)")
            depth_um_mm = resize_nn(depth_um_mm_raw, (ref_H, ref_W))
        else:
            depth_um_mm = depth_um_mm_raw

    # Sanity: intrinsics must correspond to (ref_W, ref_H)
    # If你的 fx,fy,cx,cy 是针对原始 UM/UNIT 尺寸，请先按比例换算到 (ref_W,ref_H) 后再传入。
    H_ref, W_ref = ref_H, ref_W

    # Metrics on aligned grids
    rmse, mae, n = compute_rmse_mae(depth_um_mm, depth_unit_mm)
    print(f"[METRIC] RMSE={rmse:.3f} mm | MAE={mae:.3f} mm | n={n} overlapping valid pixels | size={H_ref}x{W_ref}")

    # Back-project BOTH using SAME intrinsics on the SAME aligned size
    pc_um_c,  _ = backproject_mm(depth_um_mm,  args.fx, args.fy, args.cx, args.cy, stride=args.stride)
    pc_u01_c, _ = backproject_mm(depth_unit_mm, args.fx, args.fy, args.cx, args.cy, stride=args.stride)

    # Pose (translation must be in mm)
    T_wc = load_pose_txt(args.pose_txt)
    pc_um_w  = transform_points(T_wc, pc_um_c)
    pc_u01_w = transform_points(T_wc, pc_u01_c)

    # Optional PLY
    if args.save_ply:
        os.makedirs(args.save_ply, exist_ok=True)
        save_ply(os.path.join(args.save_ply, "cloud_um.ply"),   pc_um_w)
        save_ply(os.path.join(args.save_ply, "cloud_unit.ply"), pc_u01_w)
        print(f"[INFO] Saved PLYs to: {args.save_ply}")

    # Visualization
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    if pc_um_w.shape[0] > 0:
        ax.scatter(pc_um_w[:, 0],  pc_um_w[:, 1],  pc_um_w[:, 2],  s=1, alpha=0.6, label="PNG µm → mm (aligned)")
    if pc_u01_w.shape[0] > 0:
        ax.scatter(pc_u01_w[:, 0], pc_u01_w[:, 1], pc_u01_w[:, 2], s=1, alpha=0.6, label=f"PNG {args.unit_mm}mm/LSB → mm (aligned)")
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.legend(loc="best"); ax.set_title("Point Clouds (shared pose, size-aligned before projection)")
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()