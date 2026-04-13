import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_sample_viewer import build_html

def rotmat_to_quat(R):
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]

def quat_to_rotmat(qx, qy, qz, qw):
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)

def main():
    dataset_dir = r"C:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\projects\StrawPick\NBV_article\SegPointNetsTest\multiview_dataset"
    sample_id = "sample_00000"
    s_dir = Path(dataset_dir) / sample_id
    
    with open(s_dir / "cameras.json", "r") as f:
        cameras_data = json.load(f)
    with open(s_dir / "color_map.json", "r") as f:
        color_map = json.load(f)

    color_to_info = {tuple(v["color"]): v for v in color_map.values()}

    # Ограничиваем сэмпл 5 кадрами (CHUNK_SIZE = 5), как это теперь делает get_strawberry_dataset_dicts
    CHUNK_SIZE = 5
    cameras_data = cameras_data[:CHUNK_SIZE]

    cameras = []
    chunks = []
    stride = 2
    
    for cam in cameras_data:
        fi = cam["frame_index"]
        name = f"{fi:05d}"
        
        rgb_p = s_dir / "rgb" / f"{name}.png"
        depth_p = s_dir / "depth" / f"{name}.npy"
        mask_p = s_dir / "masks" / f"{name}.png"
        
        if not rgb_p.exists() or not depth_p.exists() or not mask_p.exists():
            continue
            
        img = np.asarray(Image.open(rgb_p))
        Z_full = np.load(depth_p)[::-1, :].copy() # emulate mapper flip
        
        # Load mask and create mock instances/categories
        mask_img = np.asarray(Image.open(mask_p))
        mask_s = mask_img[::stride, ::stride]
        mask_r, mask_g, mask_b = mask_s[:,:,0], mask_s[:,:,1], mask_s[:,:,2]
        
        # Intrinsics
        intr = cam["intrinsics"]
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        
        # Pose
        R_mat = quat_to_rotmat(*cam["rotation"])
        t_vec = np.array(cam["position"], dtype=np.float32)
        
        H, W = Z_full.shape
        u = np.arange(0, W, stride, dtype=np.float32)
        v = np.arange(0, H, stride, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        
        Z_s = Z_full[::stride, ::stride]
        valid = (Z_s > 0.001) & (Z_s < 5.0)
        
        X_cam =  (uu - cx) * Z_s / fx
        Y_cam = -(vv - cy) * Z_s / fy
        Z_cam =   Z_s
        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=-1)
        
        pts_world = pts_cam @ R_mat.T + t_vec
        
        fv = valid.ravel()
        pts_chunk = pts_world.reshape(-1, 3)[fv]
        
        rgb_s = img[::stride, ::stride]
        r = rgb_s[:,:,0].ravel()[fv]
        g = rgb_s[:,:,1].ravel()[fv]
        b = rgb_s[:,:,2].ravel()[fv]
        
        # GT & PRED instances/categories
        inst_img = np.full(Z_s.shape, -1, dtype=np.float32)
        cat_img  = np.full(Z_s.shape, -1, dtype=np.float32)
        pred_inst_img = np.full(Z_s.shape, -1, dtype=np.float32)
        pred_cat_img  = np.full(Z_s.shape, -1, dtype=np.float32)
        
        straw = mask_r > 0
        for color, info in color_to_info.items():
            px = straw & (mask_r == color[0]) & (mask_g == color[1]) & (mask_b == color[2])
            
            inst_img[px] = info["instance_id"]
            cat_img[px] = info["category_id"]
            
            # Делаем ошибку детерминированной для данного инстанса (чтобы во всех кадрах была одинаковой)
            np.random.seed(info["instance_id"])
            
            # "Slightly wrong" predictions: сдвигаем ID на 0/1/2
            fake_inst = info["instance_id"] + np.random.randint(0, 3) 
            
            fake_cat = info["category_id"]
            if np.random.rand() < 0.3: # 30% error
                fake_cat = (fake_cat + 1) % 3
                
            pred_inst_img[px] = fake_inst
            pred_cat_img[px] = fake_cat
            
        gt_inst = inst_img.ravel()[fv]
        gt_cat = cat_img.ravel()[fv]
        pred_inst = pred_inst_img.ravel()[fv]
        pred_cat = pred_cat_img.ravel()[fv]
        
        chunk = np.column_stack([pts_chunk, r, g, b, gt_inst, gt_cat, pred_inst, pred_cat]).astype(np.float32)
        chunks.append(chunk)
        
        cam_dict = {
            "position": t_vec.tolist(),
            "rotation": rotmat_to_quat(R_mat),
            "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
            "frame_index": fi
        }
        cameras.append(cam_dict)

    if len(chunks) > 0:
        pts = np.concatenate(chunks, axis=0)
        is_white = (pts[:, 3] > 220) & (pts[:, 4] > 220) & (pts[:, 5] > 220)
        is_black = (pts[:, 3] < 20) & (pts[:, 4] < 20) & (pts[:, 5] < 20)
        pts = pts[~is_white & ~is_black]

        MAX_POINTS = 800000
        if len(pts) > MAX_POINTS:
            idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
            pts = pts[idx]

        html = build_html(pts, cameras, color_map, sample_name=sample_id + "_MOCK_PRED")
        out_html_path = "test_vis.html"
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Сгенерирован файл: {out_html_path}")

if __name__ == "__main__":
    main()
