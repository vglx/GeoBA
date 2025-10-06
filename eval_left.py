#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, math, numpy as np
import imageio.v2 as imageio
import open3d as o3d

# ===================== 配置 =====================
dataset     = "10"
depth_dir   = f"./data/halfDef/{dataset}/depth_left"          # ✅ 使用左目深度
mesh_dir    = f"./data/halfDef/{dataset}/results/PLYs"        # 变形后网格（deformed_mesh_f1.ply ...）
poses_txt   = f"./data/halfDef/{dataset}/poses_gt.txt"        # ✅ 读入中心相机的位姿（每行 12 或 16）
pose_format = "tcw"                                           # "tcw" 或 "twc"（文件里中心相机的格式）
mesh_unit   = "mm"                                            # PLY 单位："mm" 或 "m"

# ✅ 模板（初始）网格路径（obj/ply 都可）
template_mesh = f"./data/halfDef/{dataset}/mesh.obj"

# 相机内参（像素）——左目与深度同一相机
tx_mm = 4.5                  # ✅ rig 总基线（毫米）
fx, fy = 155.0030, 155.0030
cx, cy = 160.0, 120.0

# 光栅化与速度选项
FAST_MODE = False
STRIDE    = 1
Z_NEAR    = 1e-6
# =================================================


def _load_sidecar_json(png_path):
    base, _ = os.path.splitext(png_path)
    sidecar = base + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_depth_um_png(png_path, return_unit="m"):
    depth_u16 = imageio.imread(png_path).astype(np.uint16)
    depth = depth_u16.astype(np.float32)

    meta = _load_sidecar_json(png_path)
    if meta:
        unit = str(meta.get("unit", "um")).lower()
        scale = float(meta.get("scale", 1.0))
        depth *= scale
        if unit.startswith("um"):
            depth_m = depth * 1e-6
        elif unit.startswith("mm"):
            depth_m = depth * 1e-3
        elif unit in ("m", "meter", "meters"):
            depth_m = depth
        else:
            depth_m = depth * 1e-6
    else:
        depth_m = depth * 1e-6

    mask = np.isfinite(depth_m) & (depth_m > 0)
    depth_m = np.where(mask, depth_m, np.nan).astype(np.float32)
    if return_unit == "mm":
        return depth_m * 1000.0, mask, meta
    return depth_m, mask, meta


def load_poses_multi(path, pose_format="tcw"):
    """
    从文本读取中心相机位姿，返回统一成 T_wc 的列表。
    - pose_format="tcw"：文件里是 T_cw -> 会转成 T_wc
    - pose_format="twc"：文件里已是 T_wc
    注意：行 0 对应模板（与深度 0.png 对齐），1..N 对应各帧。
    """
    try:
        raw = np.loadtxt(path, delimiter=",")
    except Exception:
        raw = np.loadtxt(path)
    raw = np.atleast_2d(raw)
    if raw.shape[1] not in (12, 16):
        raise RuntimeError(f"poses_txt expects 12 or 16 numbers per line, got shape {raw.shape}")
    Ts = []
    for i in range(raw.shape[0]):
        row = raw[i]
        if row.size == 12:
            T = np.eye(4, dtype=np.float64); T[:3, :4] = row.reshape(3, 4)
        else:
            T = row.reshape(4, 4)
        if pose_format.lower() == "tcw":
            T = np.linalg.inv(T)  # 统一成 T_wc（世界<-相机 的逆）
        Ts.append(T.astype(np.float64))
    return Ts


def derive_left_Twcs_from_center(Twcs_center, baseline_mm):
    """根据中心相机的 T_wc 列表，推导左目相机的 T_wc^L。
    约定：rig X 轴向右；左目在 -baseline/2，右目在 +baseline/2（单位：米）。
    T_wc^L = [Rwc, twc + Rwc * (-b/2, 0, 0)^T]
    旋转与中心一致（平行双目）。
    """
    half_b_m = (baseline_mm * 0.5) * 1e-3
    tC_L = np.array([-half_b_m, 0.0, 0.0], dtype=np.float64)  # 左负右正
    Twcs_left = []
    for Twc in Twcs_center:
        Rwc = Twc[:3, :3]
        twc = Twc[:3, 3]
        TLwc = np.eye(4, dtype=np.float64)
        TLwc[:3, :3] = Rwc
        TLwc[:3, 3]  = twc + Rwc @ tC_L
        Twcs_left.append(TLwc)
    return Twcs_left


def world_to_cam(verts_w, T_cw):
    V = verts_w.shape[0]
    verts_h = np.concatenate([verts_w, np.ones((V, 1), dtype=verts_w.dtype)], axis=1)
    verts_c = (T_cw @ verts_h.T).T[:, :3]
    return verts_c


def project(verts_c, fx, fy, cx, cy):
    z = verts_c[:, 2]
    valid = z > Z_NEAR
    x = np.empty_like(z, dtype=np.float32); x.fill(np.nan)
    y = np.empty_like(z, dtype=np.float32); y.fill(np.nan)
    x[valid] = (verts_c[valid, 0] * fx / z[valid]) + cx
    y[valid] = (verts_c[valid, 1] * fy / z[valid]) + cy
    return np.stack([x, y, z.astype(np.float32)], axis=1), valid


def _edge_fn(ax, ay, bx, by, px, py):
    return (py - ay) * (bx - ax) - (px - ax) * (by - ay)


def rasterize_triangles(verts_c, faces, h, w, stride=1):
    depth = np.full((h, w), np.inf, dtype=np.float32)
    proj_, valid = project(verts_c, fx, fy, cx, cy)

    for f in faces:
        i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        x0, y0, z0 = proj_[i0]; x1, y1, z1 = proj_[i1]; x2, y2, z2 = proj_[i2]
        xmin = int(max(0, math.floor(min(x0, x1, x2))))
        xmax = int(min(w - 1, math.ceil(max(x0, x1, x2))))
        ymin = int(max(0, math.floor(min(y0, y1, y2))))
        ymax = int(min(h - 1, math.ceil(max(y0, y1, y2))))
        if xmax < xmin or ymax < ymin:
            continue
        area = _edge_fn(x0, y0, x1, y1, x2, y2)
        if area == 0:
            continue
        for py in range(ymin, ymax + 1, stride):
            xs = np.arange(xmin, xmax + 1, stride, dtype=np.float32)
            pys = np.full_like(xs, py, dtype=np.float32)
            w0 = _edge_fn(x1, y1, x2, y2, xs, pys)
            w1 = _edge_fn(x2, y2, x0, y0, xs, pys)
            w2 = _edge_fn(x0, y0, x1, y1, xs, pys)
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) if area > 0 else (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
            if not np.any(inside):
                continue
            w0i, w1i, w2i = w0[inside], w1[inside], w2[inside]
            denom = (w0i + w1i + w2i)
            a0 = w0i / denom; a1 = w1i / denom; a2 = w2i / denom
            zi = a0 * z0 + a1 * z1 + a2 * z2
            xs_i = xs[inside].astype(np.int32)
            ys_i = np.full(xs_i.shape, py, dtype=np.int32)
            old = depth[ys_i, xs_i]
            upd = zi < old
            if np.any(upd):
                depth[ys_i[upd], xs_i[upd]] = zi[upd]
    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0] = np.nan
    return depth


def rasterize_points_nearest(verts_c, h, w):
    depth = np.full((h, w), np.inf, dtype=np.float32)
    proj_, valid = project(verts_c, fx, fy, cx, cy)
    pts = proj_[valid]
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    zs = pts[:, 2]
    m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, zs = xs[m], ys[m], zs[m]
    for x, y, z in zip(xs, ys, zs):
        if z < depth[y, x]:
            depth[y, x] = z
    depth[~np.isfinite(depth)] = np.nan
    depth[depth <= 0] = np.nan
    return depth


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan, 0, 0.0
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d * d))), int(m.sum()), float(np.sum(d * d))


# ---------- 批处理工具 ----------
_num_re = re.compile(r"(\d+)")

def _index_from_name(path):
    m = _num_re.findall(os.path.basename(path))
    return int(m[0]) if m else None

def _sorted_by_index(filepaths):
    pairs = []
    for p in filepaths:
        idx = _index_from_name(p)
        if idx is not None:
            pairs.append((idx, p))
    pairs.sort(key=lambda x: x[0])
    return pairs

def _list_files(dir_, exts):
    exts = tuple(exts)
    return [os.path.join(dir_, f) for f in os.listdir(dir_) if f.lower().endswith(exts)]


def load_mesh_as_meters(mesh_path, unit):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None, None
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    if unit.lower() == "mm":
        V_m = V * 1e-3
    elif unit.lower() == "m":
        V_m = V
    else:
        raise ValueError("mesh_unit must be 'm' or 'mm'")
    return V_m, F


def render_depth_from_mesh(V_m, F, T_wc, H, W):
    T_cw = np.linalg.inv(T_wc)
    V_c = world_to_cam(V_m.astype(np.float64), T_cw)
    if FAST_MODE:
        return rasterize_points_nearest(V_c, H, W)
    else:
        return rasterize_triangles(V_c, F, H, W, stride=STRIDE)


if __name__ == "__main__":
    # 列出深度图与网格，按文件名中的数字排序（index 从 0 开始）
    depth_files = _sorted_by_index(_list_files(depth_dir, (".png", ".tiff", ".tif")))
    mesh_files  = _sorted_by_index(_list_files(mesh_dir, (".ply", ".obj")))
    if len(depth_files) == 0 or len(mesh_files) == 0:
        raise RuntimeError("深度图或网格文件未找到，请检查 depth_dir / mesh_dir。")

    # 读取中心相机位姿（返回 T_wc 列表，行号与深度图索引一致）
    T_wcs_center = load_poses_multi(poses_txt, pose_format=pose_format)
    # ✅ 从中心位姿推导左目位姿（左负右正）
    T_wcs_left   = derive_left_Twcs_from_center(T_wcs_center, baseline_mm=tx_mm)

    # 载入模板网格
    if not os.path.exists(template_mesh):
        raise RuntimeError(f"找不到模板网格：{template_mesh}")
    V_tmpl_m, F_tmpl = load_mesh_as_meters(template_mesh, mesh_unit)
    if V_tmpl_m is None:
        raise RuntimeError(f"模板网格为空：{template_mesh}")

    # 全局累积（用于加权 RMSE）
    total_px_deformed = 0
    total_sse_deformed = 0.0
    total_px_template = 0
    total_sse_template = 0.0

    # 建立索引->路径映射，遍历匹配 index>=1（按你的约定跳过 0：模板帧）
    depth_map = dict(depth_files)
    mesh_map  = dict(mesh_files)
    max_idx = min(max(depth_map.keys()), max(mesh_map.keys()), len(T_wcs_left)-1)

    print(f"[INFO] depth imgs: {len(depth_files)}, meshes: {len(mesh_files)}, poses(center): {len(T_wcs_center)} (max usable idx={max_idx})")
    print("[INFO] 评估使用：左目深度 + 左目位姿 (由中心位姿 + 基线 4.5mm 推导)")
    print("[INFO] 模板网格:", os.path.basename(template_mesh))

    per_frame = []  # (idx, rmse_def_mm, rmse_tmpl_mm, npx_def, npx_tmpl)
    for idx in range(1, max_idx + 1):
        if (idx not in depth_map) or (idx not in mesh_map):
            print(f"[WARN] 缺少文件：idx={idx} depth/mesh 之一不存在，跳过")
            continue

        depth_path = depth_map[idx]
        mesh_path  = mesh_map[idx]
        T_wc_L     = T_wcs_left[idx]  # ✅ 与左目深度同一相机的位姿

        # 读取深度（米）——左目
        depth_m, _, _ = load_depth_um_png(depth_path, return_unit="m")
        H, W = depth_m.shape

        # ----- Deformed vs Depth（左目位姿）-----
        V_def_m, F_def = load_mesh_as_meters(mesh_path, mesh_unit)
        if V_def_m is None:
            print(f"[WARN] 空网格：{mesh_path}，跳过 deformed 对比")
            rmse_def_m, npx_def, sse_def = np.nan, 0, 0.0
        else:
            depth_def_m = render_depth_from_mesh(V_def_m, F_def, T_wc_L, H, W)
            rmse_def_m, npx_def, sse_def = rmse(depth_def_m, depth_m)

        # ----- Template vs Depth（左目位姿）-----
        depth_tmpl_m = render_depth_from_mesh(V_tmpl_m, F_tmpl, T_wc_L, H, W)
        rmse_tmpl_m, npx_tmpl, sse_tmpl = rmse(depth_tmpl_m, depth_m)

        # 统计
        if np.isfinite(rmse_def_m):
            total_px_deformed  += npx_def
            total_sse_deformed += sse_def
        if np.isfinite(rmse_tmpl_m):
            total_px_template  += npx_tmpl
            total_sse_template += sse_tmpl

        rmse_def_mm  = rmse_def_m  * 1000.0 if np.isfinite(rmse_def_m)  else np.nan
        rmse_tmpl_mm = rmse_tmpl_m * 1000.0 if np.isfinite(rmse_tmpl_m) else np.nan
        per_frame.append((idx, rmse_def_mm, rmse_tmpl_mm, npx_def, npx_tmpl))

        # 帧级打印：deformed / template / 改进
        if np.isfinite(rmse_def_mm) and np.isfinite(rmse_tmpl_mm):
            delta_mm = rmse_tmpl_mm - rmse_def_mm
            perc = (delta_mm / rmse_tmpl_mm * 100.0) if rmse_tmpl_mm > 0 else 0.0
            print(f"[FRAME {idx:03d}] def={rmse_def_mm:.3f} mm | tmpl={rmse_tmpl_mm:.3f} mm | Δ={delta_mm:.3f} mm ({perc:.1f}%) | depth='{os.path.basename(depth_path)}'")
        else:
            print(f"[FRAME {idx:03d}] def={rmse_def_mm} mm | tmpl={rmse_tmpl_mm} mm | depth='{os.path.basename(depth_path)}'")

    # 全局加权 RMSE（按像素数加权）
    if total_px_deformed > 0:
        global_def_m  = math.sqrt(total_sse_deformed / total_px_deformed)
        global_def_mm = global_def_m * 1000.0
    else:
        global_def_mm = float("nan")

    if total_px_template > 0:
        global_tmpl_m  = math.sqrt(total_sse_template / total_px_template)
        global_tmpl_mm = global_tmpl_m * 1000.0
    else:
        global_tmpl_mm = float("nan")

    print("\n[GLOBAL] Weighted RMSE (idx>=1)")
    print(f"         Deformed : {global_def_mm:.3f} mm over {total_px_deformed} px")
    print(f"         Template : {global_tmpl_mm:.3f} mm over {total_px_template} px")
    if np.isfinite(global_def_mm) and np.isfinite(global_tmpl_mm):
        delta_g_mm = global_tmpl_mm - global_def_mm
        perc_g = (delta_g_mm / global_tmpl_mm * 100.0) if global_tmpl_mm > 0 else 0.0
        print(f"         Δ (tmpl-def): {delta_g_mm:.3f} mm  ({perc_g:.1f}% improvement)")
    else:
        print("         Δ (tmpl-def): N/A")