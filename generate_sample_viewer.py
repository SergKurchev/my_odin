#!/usr/bin/env python3
"""
generate_sample_viewer.py
=========================
Generates a self-contained interactive HTML viewer for one multi-view sample.

Reads all 20 frames of:
  - RGB images      (rgb/00000.png ... 00019.png)
  - Depth maps      (depth/00000.npy ... 00019.npy)   [float32 metres]
  - Segmentation    (masks/00000.png ... 00019.png)
  - Camera params   (cameras.json)
  - Color map       (color_map.json)

Projects all pixels with valid depth into world-space, merges them,
and writes a standalone visualization.html with:
  - Orbit / Pan / Zoom
  - Mode toggle: RGB | Segmentation
  - Per-frame camera frustum visualisation (toggle-able)
  - Instance legend
  - Stats bar

Usage:
    python generate_sample_viewer.py <sample_path> [--max-points 800000] [--stride 2]

Example:
    python generate_sample_viewer.py multiview_dataset/sample_00000
    python generate_sample_viewer.py multiview_dataset/sample_00000 --stride 3 --max-points 600000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_NAMES = {0: "Ripe", 1: "Unripe", 2: "Half-ripe"}
SEG_PALETTE = {
    0:  (232,  68,  68),   # ripe      – red
    1:  ( 72, 199,  72),   # unripe    – green
    2:  (240, 160,  40),   # half-ripe – orange
   -1:  ( 80,  80,  80),   # background / peduncle
}

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_rotmat(qx, qy, qz, qw):
    """Unit quaternion → 3×3 rotation matrix (float64)."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def unproject_frame(depth: np.ndarray, rgb: np.ndarray, mask: np.ndarray,
                    fx, fy, cx, cy,
                    R: np.ndarray, t: np.ndarray,
                    color_to_info: dict,
                    stride: int = 1,
                    max_depth: float = 5.0,
                    mode: str = "plant"):
    """
    Project one frame into world-space points.

    mode:
      'all'   - all valid depth pixels (room walls included)
      'plant' - exclude pure-white walls and pure-black floor by RGB color
      'bush'  - only pixels that appear in the segmentation mask (berries only)

    Unity-specific notes:
    - ReadPixels on Windows/DX stores rows bottom-up in Texture2D.
      The .npy is written in this order so row[0] = BOTTOM.  We flip vertically.
    - Unity camera Y is UP; image v increases downward → negate Y_cam.
    - Depth shader outputs LINEAR normalized depth (after our shader fix).

    Returns ndarray (N, 8): x, y, z, r, g, b, instance_id, category_id
    """
    H, W = depth.shape

    # Fix 1: flip depth vertically (Unity ReadPixels bottom-up)
    depth = depth[::-1, :]

    u = np.arange(0, W, stride, dtype=np.float32)
    v = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth[::stride, ::stride].astype(np.float32)
    valid = (Z > 0.001) & (Z < max_depth)

    # Fix 2: Unity camera Y-up → negate Y_cam
    X_cam =  (uu - cx) * Z / fx
    Y_cam = -(vv - cy) * Z / fy
    Z_cam =   Z

    pts_cam   = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
    pts_world = pts_cam @ R.T + t

    rgb_s  = rgb[::stride, ::stride]
    mask_s = mask[::stride, ::stride]

    # Segmentation labels
    inst_img = np.full(Z.shape, -1, dtype=np.int32)
    cat_img  = np.full(Z.shape, -1, dtype=np.int32)
    mr, mg, mb = mask_s[:,:,0], mask_s[:,:,1], mask_s[:,:,2]
    straw = mr > 0
    for color, info in color_to_info.items():
        px = straw & (mr == color[0]) & (mg == color[1]) & (mb == color[2])
        inst_img[px] = info["instance_id"]
        cat_img[px]  = info["category_id"]

    # Mode filter
    if mode == "bush":
        # Only segmented strawberry pixels
        valid = valid & straw
    elif mode == "plant":
        # Exclude room background:
        #   white walls: R,G,B all > 220
        #   black floor: R,G,B all < 20
        rr = rgb_s[:,:,0]; gg = rgb_s[:,:,1]; bb = rgb_s[:,:,2]
        is_white = (rr > 220) & (gg > 220) & (bb > 220)
        is_black = (rr <  20) & (gg <  20) & (bb <  20)
        valid = valid & ~is_white & ~is_black
    # mode == "all" → no extra filter

    fv   = valid.ravel()
    pts  = pts_world.reshape(-1, 3)[fv]
    r    = rgb_s[:,:,0].ravel()[fv].astype(np.float32)
    g    = rgb_s[:,:,1].ravel()[fv].astype(np.float32)
    b    = rgb_s[:,:,2].ravel()[fv].astype(np.float32)
    inst = inst_img.ravel()[fv].astype(np.float32)
    cat  = cat_img.ravel()[fv].astype(np.float32)

    return np.column_stack([pts, r, g, b, inst, cat]).astype(np.float32)



# ─────────────────────────────────────────────────────────────────────────────
# Build point cloud from all frames
# ─────────────────────────────────────────────────────────────────────────────


def build_pointcloud(sample_path: Path, cameras: list, color_map: dict,
                     stride: int, max_points: int,
                     mode: str = "plant") -> np.ndarray:

    color_to_info = {tuple(v["color"]): v for v in color_map.values()}

    chunks = []
    for cam in cameras:
        fi   = cam["frame_index"]
        name = f"{fi:05d}"

        depth_p = sample_path / "depth" / (name + ".npy")
        rgb_p   = sample_path / "rgb"   / (name + ".png")
        mask_p  = sample_path / "masks" / (name + ".png")

        if not depth_p.exists() or not rgb_p.exists():
            print(f"  [skip] frame {fi}: missing files")
            continue

        depth = np.load(str(depth_p))
        rgb   = np.array(Image.open(rgb_p).convert("RGB"))
        mask  = np.array(Image.open(mask_p).convert("RGB")) \
                if mask_p.exists() else np.zeros_like(rgb)

        intr  = cam["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        R = quat_to_rotmat(*cam["rotation"])
        t = np.array(cam["position"], dtype=np.float64)

        chunk = unproject_frame(depth, rgb, mask, fx, fy, cx, cy, R, t,
                                color_to_info, stride=stride, mode=mode)
        chunks.append(chunk)
        print(f"  Frame {fi:2d}: {len(chunk):>8,} pts")

    if not chunks:
        return np.zeros((0, 8), dtype=np.float32)

    pts = np.concatenate(chunks, axis=0)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        print(f"  Downsampled to {max_points:,}")

    return pts


    color_to_info = {tuple(v["color"]): v for v in color_map.values()}

    chunks = []
    for cam in cameras:
        fi   = cam["frame_index"]
        name = f"{fi:05d}"

        depth_p = sample_path / "depth" / (name + ".npy")
        rgb_p   = sample_path / "rgb"   / (name + ".png")
        mask_p  = sample_path / "masks" / (name + ".png")

        if not depth_p.exists() or not rgb_p.exists():
            print(f"  [skip] frame {fi}: missing files")
            continue

        depth = np.load(str(depth_p))                              # (H,W) float32
        rgb   = np.array(Image.open(rgb_p).convert("RGB"))        # (H,W,3) uint8
        mask  = np.array(Image.open(mask_p).convert("RGB")) \
                if mask_p.exists() else np.zeros_like(rgb)

        intr  = cam["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        R = quat_to_rotmat(*cam["rotation"])
        t = np.array(cam["position"], dtype=np.float64)

        chunk = unproject_frame(depth, rgb, mask, fx, fy, cx, cy, R, t,
                                color_to_info, stride=stride,
                                only_bush=only_bush)
        chunks.append(chunk)
        print(f"  Frame {fi:2d}: {len(chunk):>8,} pts")

    if not chunks:
        return np.zeros((0, 8), dtype=np.float32)

    pts = np.concatenate(chunks, axis=0)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        print(f"  Downsampled to {max_points:,}")

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Camera frustum helper
# ─────────────────────────────────────────────────────────────────────────────

def frustum_lines(cam: dict, size: float = 0.06):
    """Return list of [x1,y1,z1, x2,y2,z2] line segments for a frustum."""
    intr = cam["intrinsics"]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    W = cx * 2; H = cy * 2
    half_w = W / 2 / fx * size
    half_h = H / 2 / fy * size

    R = quat_to_rotmat(*cam["rotation"])
    t = np.array(cam["position"], dtype=np.float64)

    # Four corners at depth=size in camera space
    corners_cam = np.array([
        [-half_w,  half_h, size],
        [ half_w,  half_h, size],
        [ half_w, -half_h, size],
        [-half_w, -half_h, size],
    ])
    corners_w = corners_cam @ R.T + t

    origin = t
    lines = []
    for c in corners_w:
        lines.append([*origin, *c])
    for i in range(4):
        lines.append([*corners_w[i], *corners_w[(i+1) % 4]])
    return np.array(lines, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def compact_float_array(arr: np.ndarray, precision: int = 4) -> str:
    """Serialize 1-D float array as compact JS TypedArray literal."""
    return "new Float32Array([" + ",".join(f"{v:.{precision}f}" for v in arr) + "])"


def build_html(pts: np.ndarray, cameras: list, color_map: dict,
               sample_name: str) -> str:

    N = len(pts)
    print(f"  Building HTML for {N:,} points …")

    # ── Camera frustums ────────────────────────────────────────────────────
    all_frustum_lines = []
    for cam in cameras:
        all_frustum_lines.append(frustum_lines(cam))
    frustum_arr = np.concatenate(all_frustum_lines, axis=0)  # (M, 6)

    # ── Build instance info for legend ─────────────────────────────────────
    instance_info = {}
    for k, v in color_map.items():
        cat = v["category_id"]
        col = SEG_PALETTE.get(cat, SEG_PALETTE[-1])
        instance_info[str(v["instance_id"])] = {
            "ripeness":    v["ripeness"],
            "category_id": cat,
            "hex": "#{:02x}{:02x}{:02x}".format(*col),
        }

    # Count per-category strawberry pixels in point cloud
    cat_counts = {}
    if N > 0:
        for cat in [0, 1, 2]:
            cat_counts[cat] = int(np.sum(pts[:, 7] == cat))

    # ── Encode point cloud as flat typed arrays ────────────────────────────
    has_pred = (pts.shape[1] >= 10)
    
    if N > 0:
        xs   = pts[:, 0].astype(np.float32)
        ys   = pts[:, 1].astype(np.float32)
        zs   = pts[:, 2].astype(np.float32)
        rs   = np.clip(pts[:, 3], 0, 255).astype(np.uint8)
        gs   = np.clip(pts[:, 4], 0, 255).astype(np.uint8)
        bs   = np.clip(pts[:, 5], 0, 255).astype(np.uint8)
        insts = pts[:, 6].astype(np.int32)
        cats = pts[:, 7].astype(np.int32)
        if has_pred:
            insts_pred = pts[:, 8].astype(np.int32)
            cats_pred = pts[:, 9].astype(np.int32)
    else:
        xs = ys = zs = np.zeros(0, np.float32)
        rs = gs = bs = np.zeros(0, np.uint8)
        insts = np.zeros(0, np.int32)
        cats = np.zeros(0, np.int32)
        if has_pred:
            insts_pred = np.zeros(0, np.int32)
            cats_pred = np.zeros(0, np.int32)

    js_xs   = compact_float_array(xs)
    js_ys   = compact_float_array(ys)
    js_zs   = compact_float_array(zs)
    js_rs   = "new Uint8Array([" + ",".join(str(v) for v in rs)   + "])"
    js_gs   = "new Uint8Array([" + ",".join(str(v) for v in gs)   + "])"
    js_bs   = "new Uint8Array([" + ",".join(str(v) for v in bs)   + "])"
    js_insts= "new Int32Array([" + ",".join(str(v) for v in insts)+ "])"
    js_cats = "new Int32Array([" + ",".join(str(v) for v in cats) + "])"
    js_insts_pred = "new Int32Array([" + ",".join(str(v) for v in insts_pred)+ "])" if has_pred else "null"
    js_cats_pred = "new Int32Array([" + ",".join(str(v) for v in cats_pred)+ "])" if has_pred else "null"

    # Frustum line endpoints
    fl = frustum_arr.ravel()
    js_frustum = compact_float_array(fl)

    # ── Camera positions for axes ──────────────────────────────────────────
    cam_pos_js = json.dumps([[c["position"][0], c["position"][1], c["position"][2]]
                              for c in cameras])

    seg_palette_js = json.dumps({str(k): list(v) for k, v in SEG_PALETTE.items()})
    instance_info_js = json.dumps(instance_info, indent=2)

    num_cameras    = len(cameras)
    num_frustum_lines = len(frustum_arr)
    bg_color       = "0x0d1117"

    # Stats
    total_strawberry = sum(cat_counts.values())
    stats_html = f"{N:,} points · {num_cameras} cameras"
    for cat, cnt in sorted(cat_counts.items()):
        if cnt:
            stats_html += f" · {CATEGORY_NAMES[cat]}: {cnt:,}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RGBD Viewer — {sample_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{
  background:#0d1117;color:#e6edf3;
  font-family:'Segoe UI',system-ui,sans-serif;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}}
#header{{
  padding:10px 18px;background:#161b22;
  border-bottom:1px solid #30363d;
  display:flex;align-items:center;gap:10px;flex-shrink:0;
}}
#header h1{{font-size:15px;font-weight:600;flex:1;color:#f0f6fc}}
#header .meta{{font-size:11px;color:#8b949e}}
#toolbar{{
  display:flex;align-items:center;gap:6px;flex-wrap:wrap;
  padding:7px 18px;background:#161b22;
  border-bottom:1px solid #30363d;flex-shrink:0;
}}
.btn{{
  padding:5px 13px;border-radius:5px;
  border:1px solid #30363d;background:#21262d;
  color:#e6edf3;font-size:12px;cursor:pointer;
  transition:background .12s,border-color .12s;
  user-select:none;
}}
.btn:hover{{background:#30363d}}
.btn.active{{background:#1f6feb;border-color:#388bfd;color:#fff}}
.sep{{width:1px;height:20px;background:#30363d;margin:0 2px}}
#legend{{
  display:none;flex-wrap:wrap;gap:8px;align-items:center;
  padding:6px 18px;background:#161b22;
  border-bottom:1px solid #30363d;flex-shrink:0;font-size:12px;
}}
.leg-item{{display:flex;align-items:center;gap:5px;color:#8b949e}}
.leg-dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
#main{{flex:1;display:flex;min-height:0}}
#canvas-container{{flex:1;position:relative;overflow:hidden}}
canvas{{width:100%!important;height:100%!important;display:block}}
#sidebar{{
  width:220px;flex-shrink:0;background:#161b22;
  border-left:1px solid #30363d;overflow-y:auto;
  font-size:12px;display:flex;flex-direction:column;
}}
#sidebar-title{{
  padding:10px 12px;font-weight:600;font-size:12px;
  color:#8b949e;border-bottom:1px solid #30363d;
  letter-spacing:.05em;text-transform:uppercase;
}}
.frame-btn{{
  padding:8px 12px;border-bottom:1px solid #21262d;
  cursor:pointer;transition:background .1s;
  display:flex;align-items:center;gap:8px;color:#c9d1d9;
}}
.frame-btn:hover{{background:#21262d}}
.frame-btn.active{{background:#1f6feb22;color:#58a6ff}}
.frame-num{{font-weight:600;min-width:20px;font-size:11px}}
.frame-pos{{font-size:10px;color:#484f58;font-family:monospace}}
#info{{
  position:absolute;bottom:10px;left:14px;
  font-size:10px;color:#484f58;pointer-events:none;line-height:1.6;
}}
#loading{{
  position:absolute;inset:0;background:#0d1117;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:12px;font-size:14px;color:#8b949e;z-index:10;
}}
.spinner{{
  width:36px;height:36px;border:3px solid #30363d;
  border-top-color:#1f6feb;border-radius:50%;
  animation:spin .8s linear infinite;
}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>

<div id="header">
  <h1>🍓 RGBD Point Cloud — {sample_name}</h1>
  <span class="meta">{stats_html}</span>
</div>

<div id="toolbar">
  <button class="btn active" id="btnRGB"  onclick="setMode('rgb')">RGB</button>
  <div class="sep"></div>
  <span style="font-size:12px;color:#8b949e">Ground Truth:</span>
  <button class="btn"        id="btnSeg"  onclick="setMode('seg')">Category</button>
  <button class="btn"        id="btnInst" onclick="setMode('inst')">Instances</button>
  """ + (f"""<div class="sep"></div>
  <span style="font-size:12px;color:#8b949e">Prediction:</span>
  <button class="btn"        id="btnSegPred"  onclick="setMode('seg_pred')">Category</button>
  <button class="btn"        id="btnInstPred" onclick="setMode('inst_pred')">Instances</button>
  """ if has_pred else "") + f"""
  <div class="sep"></div>
  <button class="btn active" id="btnFrustums" onclick="toggleFrustums()">Cameras</button>
  <button class="btn active" id="btnCamDots"  onclick="toggleCamDots()">Cam Dots</button>
  <div class="sep"></div>
  <button class="btn" onclick="resetCamera()">Reset View</button>
  <button class="btn" onclick="toggleSidebar()">Frames ▾</button>
</div>

<div id="legend">
  <div class="leg-item"><div class="leg-dot" style="background:rgb(232,68,68)"></div>Ripe</div>
  <div class="leg-item"><div class="leg-dot" style="background:rgb(72,199,72)"></div>Unripe</div>
  <div class="leg-item"><div class="leg-dot" style="background:rgb(240,160,40)"></div>Half-ripe</div>
  <div class="leg-item"><div class="leg-dot" style="background:rgb(80,80,80)"></div>Background</div>
</div>

<div id="main">
  <div id="canvas-container">
    <div id="loading">
      <div class="spinner"></div>
      <span>Building point cloud…</span>
    </div>
    <div id="info">
      Scroll: zoom &nbsp;·&nbsp; Left drag: orbit &nbsp;·&nbsp; Right drag / Middle: pan
    </div>
  </div>

  <div id="sidebar">
    <div id="sidebar-title">Frames ({num_cameras})</div>
    {''.join(
        f'<div class="frame-btn" id="fb{c["frame_index"]}" onclick="highlightCamera({c["frame_index"]})">'
        f'<span class="frame-num">#{c["frame_index"]:02d}</span>'
        f'<span class="frame-pos">'
        f'{c["position"][0]:+.2f} {c["position"][1]:+.2f} {c["position"][2]:+.2f}'
        f'</span></div>'
        for c in cameras
    )}
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";

// ── Data ──────────────────────────────────────────────────────────────────────
const XS    = {js_xs};
const YS    = {js_ys};
const ZS    = {js_zs};
const RS    = {js_rs};
const GS    = {js_gs};
const BS    = {js_bs};
const INSTS = {js_insts};
const CATS  = {js_cats};
const INSTS_PRED = {js_insts_pred};
const CATS_PRED  = {js_cats_pred};
const N     = XS.length;

const FRUSTUM_SEGS = {js_frustum}; // flat [x1,y1,z1,x2,y2,z2, ...]
const CAM_POSITIONS = {cam_pos_js};
const SEG_PALETTE   = {seg_palette_js};
const INSTANCE_INFO = {instance_info_js};

// ── Three.js setup ────────────────────────────────────────────────────────────
const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({{antialias:false}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color({bg_color});

const camera = new THREE.PerspectiveCamera(60,
    container.clientWidth / container.clientHeight, 0.001, 50);
camera.position.set(0, 0.5, 1.5);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = true;

// ── Point cloud geometry ──────────────────────────────────────────────────────
const geo = new THREE.BufferGeometry();
const posArr = new Float32Array(N * 3);
const colArr = new Float32Array(N * 3);

for (let i = 0; i < N; i++) {{
    posArr[i*3]   = XS[i]; posArr[i*3+1] = YS[i]; posArr[i*3+2] = ZS[i];
}}

geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
geo.setAttribute('color',    new THREE.BufferAttribute(colArr, 3));

let currentMode = 'rgb';

function applyColors(mode) {{
    for (let i = 0; i < N; i++) {{
        let r, g, b;
        if (mode === 'rgb') {{
            r = RS[i]/255; g = GS[i]/255; b = BS[i]/255;
        }} else if (mode === 'seg' || mode === 'seg_pred') {{
            const val = mode === 'seg' ? CATS[i] : (CATS_PRED ? CATS_PRED[i] : -1);
            const key = String(val);
            const col = SEG_PALETTE[key] || SEG_PALETTE['-1'];
            r = col[0]/255; g = col[1]/255; b = col[2]/255;
        }} else if (mode === 'inst' || mode === 'inst_pred') {{
            const inst = mode === 'inst' ? INSTS[i] : (INSTS_PRED ? INSTS_PRED[i] : -1);
            if (inst < 0) {{
                // background
                const col = SEG_PALETTE['-1'];
                r = col[0]/255; g = col[1]/255; b = col[2]/255;
            }} else {{
                // pseudo-random color
                const hash = (inst * 2654435761) >>> 0; 
                r = ((hash >>> 16) & 0xFF) / 255;
                g = ((hash >>> 8) & 0xFF) / 255;
                b = (hash & 0xFF) / 255;
            }}
        }}
        colArr[i*3]=r; colArr[i*3+1]=g; colArr[i*3+2]=b;
    }}
    geo.attributes.color.needsUpdate = true;
}}

applyColors('rgb');

const ptsMat = new THREE.PointsMaterial({{size:0.003,vertexColors:true,sizeAttenuation:true}});
const ptsObj = new THREE.Points(geo, ptsMat);
scene.add(ptsObj);

// ── Camera frustums ───────────────────────────────────────────────────────────
const frustumGeo = new THREE.BufferGeometry();
const numSegs = FRUSTUM_SEGS.length / 6;
const fposArr = new Float32Array(numSegs * 6);
for (let i = 0; i < FRUSTUM_SEGS.length; i++) fposArr[i] = FRUSTUM_SEGS[i];
frustumGeo.setAttribute('position', new THREE.BufferAttribute(fposArr, 3));
const frustumMat = new THREE.LineBasicMaterial({{color:0xffa040, opacity:0.6, transparent:true}});
const frustumObj = new THREE.LineSegments(frustumGeo, frustumMat);
scene.add(frustumObj);

// ── Camera position dots ──────────────────────────────────────────────────────
const camGeo = new THREE.BufferGeometry();
const camPosFlat = new Float32Array(CAM_POSITIONS.flat());
camGeo.setAttribute('position', new THREE.BufferAttribute(camPosFlat, 3));
const camMat = new THREE.PointsMaterial({{size:0.018, color:0xffa040, sizeAttenuation:true}});
const camDots = new THREE.Points(camGeo, camMat);
scene.add(camDots);

// ── Highlight sphere (selected camera) ───────────────────────────────────────
const hlGeo  = new THREE.SphereGeometry(0.015, 8, 8);
const hlMat  = new THREE.MeshBasicMaterial({{color:0xffffff}});
const hlMesh = new THREE.Mesh(hlGeo, hlMat);
hlMesh.visible = false;
scene.add(hlMesh);

// ── Grid helper ───────────────────────────────────────────────────────────────
const grid = new THREE.GridHelper(1, 10, 0x222222, 0x1a1a1a);
scene.add(grid);

// ── Auto-centre on point cloud ────────────────────────────────────────────────
geo.computeBoundingBox();
const bb  = geo.boundingBox;
const ctr = new THREE.Vector3(); bb.getCenter(ctr);
const sz  = new THREE.Vector3(); bb.getSize(sz);
const maxDim = Math.max(sz.x, sz.y, sz.z);
camera.position.set(ctr.x, ctr.y + maxDim*0.6, ctr.z + maxDim*1.8);
controls.target.copy(ctr);
controls.update();
const defaultCamPos   = camera.position.clone();
const defaultCamTarget = controls.target.clone();

// ── Hide loading ──────────────────────────────────────────────────────────────
setTimeout(() => {{
    document.getElementById('loading').style.display = 'none';
}}, 50);

// ── UI helpers ────────────────────────────────────────────────────────────────
function setMode(mode) {{
    currentMode = mode;
    document.getElementById('btnRGB').classList.toggle('active', mode==='rgb');
    document.getElementById('btnSeg').classList.toggle('active', mode==='seg');
    document.getElementById('btnInst').classList.toggle('active', mode==='inst');
    if (document.getElementById('btnSegPred')) document.getElementById('btnSegPred').classList.toggle('active', mode==='seg_pred');
    if (document.getElementById('btnInstPred')) document.getElementById('btnInstPred').classList.toggle('active', mode==='inst_pred');
    document.getElementById('legend').style.display = (mode.startsWith('seg')) ? 'flex' : 'none';
    applyColors(mode);
}}

let frustumsVisible = true;
function toggleFrustums() {{
    frustumsVisible = !frustumsVisible;
    frustumObj.visible = frustumsVisible;
    document.getElementById('btnFrustums').classList.toggle('active', frustumsVisible);
}}

let camDotsVisible = true;
function toggleCamDots() {{
    camDotsVisible = !camDotsVisible;
    camDots.visible = camDotsVisible;
    document.getElementById('btnCamDots').classList.toggle('active', camDotsVisible);
}}

function resetCamera() {{
    camera.position.copy(defaultCamPos);
    controls.target.copy(defaultCamTarget);
    controls.update();
    hlMesh.visible = false;
    document.querySelectorAll('.frame-btn').forEach(b=>b.classList.remove('active'));
}}

let sidebarHidden = false;
function toggleSidebar() {{
    const sb = document.getElementById('sidebar');
    sidebarHidden = !sidebarHidden;
    sb.style.display = sidebarHidden ? 'none' : '';
    document.querySelector('[onclick="toggleSidebar()"]').textContent =
        sidebarHidden ? 'Frames ▸' : 'Frames ▾';
}}

function highlightCamera(idx) {{
    const p = CAM_POSITIONS[idx];
    hlMesh.position.set(p[0], p[1], p[2]);
    hlMesh.visible = true;

    // Move orbit target toward camera position gently
    controls.target.set(ctr.x, ctr.y, ctr.z);
    camera.position.set(p[0]*1.8, p[1]*1.8+0.15, p[2]*1.8);
    controls.update();

    document.querySelectorAll('.frame-btn').forEach(b=>b.classList.remove('active'));
    const fb = document.getElementById('fb'+idx);
    if (fb) fb.classList.add('active');
}}

// ── Resize ────────────────────────────────────────────────────────────────────
const ro = new ResizeObserver(() => {{
    const w=container.clientWidth, h=container.clientHeight;
    camera.aspect = w/h;
    camera.updateProjectionMatrix();
    renderer.setSize(w,h);
}});
ro.observe(container);

// ── Render loop ───────────────────────────────────────────────────────────────
(function animate(){{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}})();
</script>
</body>
</html>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate RGBD HTML viewer for one multi-view sample.")
    ap.add_argument("sample_path",      help="Path to sample_NNNNN folder")
    ap.add_argument("--max-points", type=int, default=800_000,
                    help="Max total points after downsampling (default 800000)")
    ap.add_argument("--stride",     type=int, default=2,
                    help="Stride for pixel sampling (default 2 = every other pixel)")
    ap.add_argument("--mode", default="plant",
                    choices=["all", "plant", "bush"],
                    help="Point filter: 'plant'=full bush excluding white/black bg (default), "
                         "'bush'=berries only from seg mask, 'all'=everything")
    args = ap.parse_args()


    sample_path = Path(args.sample_path).resolve()
    if not sample_path.is_dir():
        print(f"[ERROR] Not a directory: {sample_path}"); sys.exit(1)

    sample_name = sample_path.name
    print(f"\n[Viewer] Building RGBD viewer for {sample_name}")
    print(f"  stride={args.stride}  max_points={args.max_points:,}\n")

    print(f"  mode={args.mode}")

    cameras   = json.loads((sample_path / "cameras.json").read_text())
    color_map = json.loads((sample_path / "color_map.json").read_text())

    pts = build_pointcloud(sample_path, cameras, color_map,
                           stride=args.stride, max_points=args.max_points,
                           mode=args.mode)


    print(f"\n  Total points: {len(pts):,}")
    print("  Generating HTML ...")

    html = build_html(pts, cameras, color_map, sample_name)

    out_path = sample_path / "visualization.html"
    out_path.write_text(html, encoding="utf-8")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone -> {out_path}  ({size_mb:.1f} MB)")
    print("Open in browser: double-click the file or drag into Chrome/Firefox\n")


if __name__ == "__main__":
    main()
