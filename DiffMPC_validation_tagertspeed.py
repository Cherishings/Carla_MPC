# -*- coding: utf-8 -*-
"""
CARLA + MPC（完全对齐训练端 wp2）：
- 误差态 x=[e_y, e_psi, e_v]，e_v=tgt-cur（不裁剪）
- x_ref ≡ 0
- 仅 MPC 内部约束（throttle∈[0,1], steer∈[-0.5,0.5]）
- 预测域 N=25
- A/B/Q 从重训导出目录加载（A_final.npy, B_final.npy, Q_final.npy）

注意：
- 模型 speed 输出的单位需与训练保持一致；如果你的模型输出为 km/h，请在下方把速度除以 3.6 转为 m/s。
"""

import os, sys, math, time
import numpy as np
import pandas as pd
import cv2
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import carla
import cvxpy as cp  # 确保安装 cvxpy 与 osqp

# ============================= 需要的路径 =============================
CKPT_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Swin_WP5_TP_mps_20250820_133420/swin_wp5_best.pth" #权重


# ============================= 相机/UI 参数 =============================
CAM_W, CAM_H = 1024, 512
FOV_DEG = 90.0
WIN_W, WIN_H = 960, 720
TILE_W, TILE_H = 480, 360
RGB_DISP_W, RGB_DISP_H = 960, 360
Z_OFFSET = 0.1

# ============================= 模型/数据设定（训练一致） =============================
INPUT_W, INPUT_H = 224, 224
NUM_CLASSES = 7
NUM_DEPTH_BINS = 8
NUM_WAYPOINTS = 5
TARGET_POINT = (20.0, 0.0)        # 训练时的引导点（自车坐标系，m）
USE_AMP = False

# ============================= 控制（完全对齐训练） =============================
AUTO_DRIVE = True
STEER_LIMIT = 0.5                 # 与训练一致
LOOKAHEAD_INDEX = 1               # wp2（lookahead_idx=1）
MAX_THROTTLE = 1.0                # 与训练一致
THROTTLE_GAIN = 1.0               # 不改变幅度

# ============================= 设备/预处理 =============================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

transform = A.Compose([
    A.Resize(INPUT_H, INPUT_W),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def apply_training_crop(img):
    H, W = img.shape[:2]
    ch, cw = 384, 1024
    if H < ch or W < cw:
        raise ValueError(f"Camera frame {W}x{H} is smaller than training crop {cw}x{ch}.")
    side = (W - cw) // 2
    return img[:ch, side:side+cw, :]

# ============================= 模型定义（与你现有一致） =============================
class UNetDecoderWithSkip(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.num_stages = len(decoder_channels)
        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for i in range(self.num_stages):
            self.up_convs.append(nn.ConvTranspose2d(in_ch, decoder_channels[i], 2, 2))
            skip_ch = encoder_channels[-(i+2)] if (i+2) <= len(encoder_channels) else 0
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(decoder_channels[i] + skip_ch, decoder_channels[i], 3, padding=1),
                nn.ReLU(inplace=True)
            ))
            in_ch = decoder_channels[i]

    def forward(self, feats):
        x = feats[-1]
        for i in range(self.num_stages):
            x = self.up_convs[i](x)
            if i + 2 <= len(feats):
                skip = feats[-(i+2)]
                if x.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = self.fuse_convs[i](x)
        return x

def build_2d_sincos_pos_embed(d_model, h, w, temperature=10000.0, device=None):
    if device is None:
        device = torch.device("cpu")
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing='ij'
    )
    xx = xx.reshape(-1).float()
    yy = yy.reshape(-1).float()
    half_dim = d_model // 4
    if half_dim < 1:
        raise ValueError("d_model should be >= 4 for 2D sincos pos embed.")
    omega = torch.arange(half_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (temperature ** (omega / max(1, half_dim - 1)))
    x_sin = torch.sin(xx[:, None] * omega[None, :])
    x_cos = torch.cos(xx[:, None] * omega[None, :])
    y_sin = torch.sin(yy[:, None] * omega[None, :])
    y_cos = torch.cos(yy[:, None] * omega[None, :])
    pos = torch.cat([x_sin, x_cos, y_sin, y_cos], dim=1)
    if pos.shape[1] < d_model:
        pos = torch.cat([pos, torch.zeros((pos.shape[0], d_model - pos.shape[1]), device=device)], dim=1)
    elif pos.shape[1] > d_model:
        pos = pos[:, :d_model]
    return pos

class SwinMultiTaskModel(nn.Module):
    def __init__(self, swin_name="swin_tiny_patch4_window7_224",
                 num_classes=NUM_CLASSES, num_depth_bins=NUM_DEPTH_BINS,
                 num_waypoints=NUM_WAYPOINTS, d_model=128, nhead=8, num_layers=3, has_depth=True):
        super().__init__()
        self.has_depth = has_depth
        self.num_waypoints = num_waypoints
        self.d_model = d_model

        self.encoder = timm.create_model(swin_name, pretrained=False, features_only=True)
        enc_channels = [f["num_chs"] for f in self.encoder.feature_info]
        enc_out_ch = enc_channels[-1]

        self.decoder = UNetDecoderWithSkip(encoder_channels=enc_channels, decoder_channels=[256,128,64])
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        if self.has_depth:
            self.depth_head = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.Conv2d(64,num_depth_bins,1))

        self.input_proj = nn.Conv2d(enc_out_ch, d_model, 1)
        self.visual_fc  = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(enc_out_ch, d_model), nn.ReLU())
        self.meta_fc    = nn.Sequential(nn.Linear(3, d_model), nn.ReLU())

        self.query_embed = nn.Parameter(torch.randn(num_waypoints, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.tf_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.wp_head = nn.Linear(d_model, 2)

        self.speed_head = nn.Linear(d_model, 1)    # 目标速度
        self.brake_head = nn.Linear(d_model, 1)    # logits（本脚本对齐模式不使用）
    def forward(self, x, target_point, current_speed):
        feats = self.encoder(x)
        feats = [f.permute(0,3,1,2).contiguous() for f in feats]

        dec_out = self.decoder(feats)
        seg_out = self.seg_head(dec_out)
        seg_out = F.interpolate(seg_out, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)

        if self.has_depth:
            depth_out = self.depth_head(dec_out)
            depth_out = F.interpolate(depth_out, size=(INPUT_H, INPUT_W), mode='bilinear', align_corners=False)
        else:
            depth_out = None

        last = feats[-1]
        H, W = last.shape[2], last.shape[3]
        mem = self.input_proj(last).flatten(2).permute(0,2,1)
        pos2d = build_2d_sincos_pos_embed(self.d_model, H, W, device=mem.device)
        mem = mem + pos2d.unsqueeze(0)

        q = self.query_embed.unsqueeze(0).expand(x.size(0), -1, -1)
        q_idx = torch.arange(self.num_waypoints, device=x.device, dtype=torch.float32)
        q_pos = torch.stack([torch.sin(q_idx/10.0), torch.cos(q_idx/10.0)], dim=1)
        if q_pos.shape[1] < self.d_model:
            q_pos = F.pad(q_pos, (0, self.d_model - q_pos.shape[1]))
        q = q + q_pos.unsqueeze(0)

        vis_g = self.visual_fc(last)
        meta  = self.meta_fc(torch.cat([target_point, current_speed.unsqueeze(1)], dim=1))
        control_feat = vis_g + meta

        dec = self.tf_decoder(tgt=q, memory=mem)
        waypoints = self.wp_head(dec)          # [B,5,2] (x,y) in ego
        speed = self.speed_head(control_feat)  # 目标速度（单位保持与训练一致）
        brake_logits = self.brake_head(control_feat)
        return seg_out, depth_out, waypoints, speed, brake_logits

def load_model(ckpt_path):
    assert os.path.exists(ckpt_path), f"CKPT_PATH not found: {ckpt_path}"
    print(f"[Info] Loading weights from: {ckpt_path}")
    model = SwinMultiTaskModel(has_depth=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[Info] Weights loaded (strict=False)")
    if missing:   print("[Missing]:", missing)
    if unexpected:print("[Unexpected]:", unexpected)
    model.eval()
    return model

# ============================= 工具函数 =============================
def to_surface(arr_rgb_uint8):
    return pygame.surfarray.make_surface(np.flipud(np.rot90(arr_rgb_uint8)))

def ego_to_world(vehicle, rel_pts, z_offset=Z_OFFSET):
    tf = vehicle.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by, bz = tf.location.x, tf.location.y, tf.location.z
    out = []
    for x, y in rel_pts:
        X = bx + x * c - y * s
        Y = by + x * s + y * c
        Z = bz + z_offset
        out.append((X, Y, Z))
    return out

def get_camera_K(camera):
    w = int(camera.attributes["image_size_x"])
    h = int(camera.attributes["image_size_y"])
    fov = float(camera.attributes["fov"])
    f = w / (2.0 * math.tan(fov * math.pi / 360.0))
    K = np.array([[f, 0, w/2.0],
                  [0, f, h/2.0],
                  [0, 0, 1]])
    return K

def world_to_camera_matrix(camera):
    return np.array(camera.get_transform().get_inverse_matrix())

def project_world_to_image(K, world_to_cam, pts_world):
    out = []
    for X, Y, Z in pts_world:
        pw = np.array([X, Y, Z, 1.0])
        pc = world_to_cam @ pw
        x, y, z = pc[0], pc[1], pc[2]
        if x <= 0:
            out.append(None)
            continue
        u = K[0,0] * (y / x) + K[0,2]
        v = K[1,1] * (-z / x) + K[1,2]
        out.append((int(u), int(v)))
    return out

class EMAFilter:
    def __init__(self, decay=0.2):
        self.decay = decay
        self.state = None
    def update(self, pts_world):
        if self.state is None:
            self.state = list(pts_world)
        else:
            d = self.decay
            self.state = [
                (d*px + (1-d)*x, d*py + (1-d)*y, d*pz + (1-d)*z)
                for (px,py,pz),(x,y,z) in zip(self.state, pts_world)
            ]
        return self.state

def catmull_rom_spline(pts2d, samples_per_seg=24):
    if len(pts2d) < 4:
        return np.array(pts2d, dtype=np.int32)
    pts = np.array(pts2d, dtype=np.float32)
    out = []
    for i in range(1, len(pts)-2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        for t in np.linspace(0, 1, samples_per_seg, endpoint=False):
            t2, t3 = t*t, t*t*t
            a = 2*p1
            b = -p0 + p2
            c = 2*p0 - 5*p1 + 4*p2 - p3
            d = -p0 + 3*p1 - 3*p2 + p3
            pt = 0.5*(a + b*t + c*t2 + d*t3)
            out.append(pt)
    out.append(pts[-2])
    return np.array(out, dtype=np.int32)

def draw_waypoints_on_rgb(rgb_vis, pix_pts, draw_ids=True, smooth=True, color=(0,255,0)):
    h, w, _ = rgb_vis.shape
    in_pts = []
    for i, p in enumerate(pix_pts):
        if p is None:
            continue
        u, v = p
        if 0 <= u < w and 0 <= v < h:
            in_pts.append((u, v, i))
    for (u, v, i) in in_pts:
        cv2.rectangle(rgb_vis, (u-4, v-4), (u+4, v+4), color, thickness=-1)
        if draw_ids:
            cv2.putText(rgb_vis, f"W{i+1}", (u+6, v-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if len(in_pts) >= 2:
        pts2d = [(u, v) for (u, v, _) in in_pts]
        if smooth and len(in_pts) >= 4:
            curve = catmull_rom_spline(pts2d, samples_per_seg=24)
            cv2.polylines(rgb_vis, [curve.reshape(-1,1,2)], False, color, 2, cv2.LINE_AA)
        else:
            arr = np.array(pts2d, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(rgb_vis, [arr], False, color, 2, cv2.LINE_AA)

# ============================= MPC（加载 A/B/Q） =============================

A = np.array([
    [ 0.94031286,  0.18832965,  0.06623337],
    [ 0.03944254,  0.22143374, -0.13703150],
    [ 0.02276975,  0.04958092,  0.87748770]
], dtype=np.float32)

B = np.array([
    [-0.04344966,  0.11081975],
    [ 0.11215162,  0.13459045],
    [-0.11678395, -0.09145724]
], dtype=np.float32)

# Q 对应 z = [e_y, e_psi, e_v, throttle, steer]
Q = np.array([
    [ 3.4961864e-01, -1.3913208e+00, -2.5043568e-01, -5.2455328e-03, -2.3252256e-02],
    [-1.3913208e+00,  9.2326937e+00,  1.1432242e+00, -5.4794043e-01,  2.8989999e+00],
    [-2.5043568e-01,  1.1432242e+00,  5.7583618e-01,  2.9375270e-01, -1.3445494e-01],
    [-5.2455328e-03, -5.4794043e-01,  2.9375270e-01,  3.3911356e-01, -5.0487196e-01],
    [-2.3252256e-02,  2.8989999e+00, -1.3445494e-01, -5.0487196e-01,  1.7973270e+01]
], dtype=np.float32)

nx = A.shape[0]
nu = B.shape[1]
assert nx == 3 and nu == 2, f"Unexpected shapes: A{A.shape}, B{B.shape}"
assert Q.shape == (nx + nu, nx + nu), f"Unexpected Q shape: {Q.shape}"

Q_xx = Q[:nx, :nx]
N = 25  # 与训练 T 对齐

def mpc_control(x0, x_refs):
    x = cp.Variable((nx, N + 1))
    u = cp.Variable((nu, N))
    cost = 0
    constr = [x[:, 0] == x0]

    H = Q  # 直接用你手写/训练得到的 Q (5x5)

    for k in range(N):
        constr += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]

        z_k     = cp.hstack([x[:, k], u[:, k]])           # (5,)
        z_ref_k = cp.hstack([x_refs[:, k], cp.Constant(np.zeros(nu))])  # u_ref=0
        # 0.5 z^T (2H) z + (-2H z_ref)^T z  —— 与训练完全一致
        cost += 0.5 * cp.quad_form(z_k, 2*H) + cp.sum(cp.multiply((-2 * (H @ z_ref_k)), z_k))

        # 控制约束
        constr += [u[0, k] <= MAX_THROTTLE, u[0, k] >= 0.0]
        constr += [u[1, k] <=  STEER_LIMIT, u[1, k] >= -STEER_LIMIT]

    # 末端代价（只对 x，用 H 的左上角块）
    x_errN  = x[:, N] - x_refs[:, N]
    cost   += 0.5 * cp.quad_form(x_errN, Q_xx)

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[MPC] infeasible: {prob.status}")
        return None

    return np.asarray(u.value[:, 0]).reshape(-1)

def build_x0_from_wp_idx(wps_ego, cur_speed_mps, tgt_speed_mps, idx=1):
    """
    通用版本：根据给定路点索引 idx 构造误差状态 x=[e_y, e_psi, e_v]
    """
    if wps_ego is None or len(wps_ego) <= idx:
        return np.zeros(3, dtype=np.float32)

    x_ref, y_ref = float(wps_ego[idx, 0]), float(wps_ego[idx, 1])  # wpN

    e_y   = float(np.clip(y_ref, -5.0, 5.0))
    e_psi = float(np.clip(math.atan2(y_ref, max(1e-3, x_ref)), -np.pi/2, np.pi/2))
    e_v   = float(tgt_speed_mps - cur_speed_mps)  # 不裁剪
    #e_v   = float(-cur_speed_mps)

    return np.array([e_y, e_psi, e_v], dtype=np.float32)

# def build_x0_from_wp_idx(wps_ego, cur_speed_mps, tgt_speed_mps, idx=1):
#     """
#     通用版本：根据给定路点索引 idx 构造误差状态 x=[e_y, e_psi, e_v]
#     """
#     if wps_ego is None or len(wps_ego) <= idx - 1:
#         return np.zeros(3, dtype=np.float32)

#     x_ref, y_ref = float(wps_ego[idx -1, 0]), float(wps_ego[idx - 1, 1])  # wpN

#     e_y   = float(np.clip(y_ref, -5.0, 5.0))
#     e_psi = float(np.clip(math.atan2(y_ref, max(1e-3, x_ref)), -np.pi/2, np.pi/2))
#     e_v   = float(tgt_speed_mps - cur_speed_mps)  # 不裁剪
#     #e_v   = float(-cur_speed_mps)

    return np.array([e_y, e_psi, e_v], dtype=np.float32)

def build_mpc_refs_zoh_idx(wps_ego, tgt_speed_mps, cur_speed_mps, idx=1, horizon=25):
    """
    根据指定 lookahead 索引 idx 构造 MPC 参考（ZOH）
    """
    if wps_ego is None or len(wps_ego) <= idx:
        return np.zeros((3, horizon+1), dtype=np.float32)

    xN, yN = float(wps_ego[idx, 0]), float(wps_ego[idx, 1])
    ey_ref   = float(np.clip(yN, -5.0, 5.0))
    epsi_ref = float(np.clip(math.atan2(yN, max(1e-3, xN)), -np.pi/2, np.pi/2))
    ev_ref   = float(tgt_speed_mps - cur_speed_mps)
    #ev_ref   = float(-cur_speed_mps)

    ey   = np.full(horizon+1, ey_ref,   dtype=np.float32)
    epsi = np.full(horizon+1, epsi_ref, dtype=np.float32)
    ev   = np.full(horizon+1, ev_ref,   dtype=np.float32)
    return np.vstack([ey, epsi, ev])

def apply_vehicle_control(vehicle, u):
    """完全对齐训练：仅按 MPC 约束裁剪；无斜率/限速/刹车保底。"""
    thr = float(np.clip(u[0], 0.0, MAX_THROTTLE)) * THROTTLE_GAIN
    steer = float(np.clip(u[1], -STEER_LIMIT, STEER_LIMIT))
    brake = 0.0
    vehicle.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brake))
    return thr, steer, brake

# ============================= 主流程 =============================
def main():
    # ---- pygame ----
    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RGB | Segmentation | Depth | MPC (Aligned)")
    clock = pygame.time.Clock()
    font_big = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 28)

    # 先加载模型，避免路径错误
    model = load_model(CKPT_PATH)

    # ---- CARLA ----
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    TARGET_TOWN = "Town04"
    world = client.get_world()
    cur = world.get_map().name
    if TARGET_TOWN not in cur:
        world = client.load_world(TARGET_TOWN)
        world.wait_for_tick()
        print(f"[Info] Loaded map: {world.get_map().name}")
    bp = world.get_blueprint_library()

    veh_bp = bp.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(veh_bp, spawn_point) or world.spawn_actor(veh_bp, spawn_point)

    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", str(FOV_DEG))
    cam_transform = carla.Transform(carla.Location(x=-1.5, z=2.0))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)

    shared = {"rgb": None}
    def on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
        shared["rgb"] = rgb
    camera.listen(on_image)

    ema = EMAFilter(decay=0.7)

    color_map = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [0, 255, 0],
        3: [255, 0, 0],
        4: [255, 255, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
    }

    log = []
    try:
        while True:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            frame = shared["rgb"]
            if frame is None:
                pygame.display.flip()
                continue

            # 当前速度
            vel = vehicle.get_velocity()
            speed_mps = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            speed_kmh = 3.6 * speed_mps

            # 图像预处理
            frame_crop = apply_training_crop(frame)
            aug = transform(image=frame_crop)
            inp = aug["image"].unsqueeze(0).to(device)

            tgt_pt = torch.tensor([TARGET_POINT], dtype=torch.float32, device=device)
            # 保持与你训练时代码一致：meta 里传当前速度（单位如果训练用 km/h 就传 km/h）
            cur_sp_for_meta = torch.tensor([speed_mps], dtype=torch.float32, device=device)

            # 推理
            with torch.no_grad():
                if USE_AMP:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        seg_logits, depth_logits, wp_pred, spd_pred, _ = model(inp, tgt_pt, cur_sp_for_meta)
                else:
                    seg_logits, depth_logits, wp_pred, spd_pred, _ = model(inp, tgt_pt, cur_sp_for_meta)

            # 语义/深度可视化
            seg_lbl = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()
            seg_rgb = np.zeros((seg_lbl.shape[0], seg_lbl.shape[1], 3), dtype=np.uint8)
            for k, c in color_map.items():
                seg_rgb[seg_lbl == k] = c
            seg_img = cv2.resize(seg_rgb, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)

            if depth_logits is not None:
                depth_cls = torch.argmax(depth_logits.squeeze(0), dim=0).cpu().numpy()
                depth_255 = (depth_cls.astype(np.float32) / (NUM_DEPTH_BINS - 1) * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_255, cv2.COLORMAP_PLASMA)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
                depth_vis = cv2.resize(depth_vis, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
            else:
                depth_vis = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

            # 顶部 RGB 叠加路点
            rgb_vis = frame.copy()
            wps_ego = wp_pred.squeeze(0).detach().cpu().numpy()  # (5,2)
            wps_world_raw = ego_to_world(vehicle, [(float(x), float(y)) for x, y in wps_ego], z_offset=Z_OFFSET)
            wps_world_smooth = ema.update(wps_world_raw)
            K = get_camera_K(camera)
            W2C = world_to_camera_matrix(camera)
            pix = project_world_to_image(K, W2C, wps_world_smooth)
            draw_waypoints_on_rgb(rgb_vis, pix, draw_ids=True, smooth=True, color=(0,255,0))

            # === 把平滑后的世界坐标重新转回自车坐标系，用于 MPC ===
            # （因为模型原本输出的 wps_ego 是自车系的，这里要让MPC跟随平滑后的路径）
            tf = vehicle.get_transform()
            yaw = math.radians(tf.rotation.yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            bx, by, bz = tf.location.x, tf.location.y, tf.location.z

            wps_ego_smooth = []
            for X, Y, Z in wps_world_smooth:
                x_rel =  (X - bx) * c + (Y - by) * s
                y_rel = -(X - bx) * s + (Y - by) * c
                wps_ego_smooth.append((x_rel, y_rel))
            wps_ego_smooth = np.array(wps_ego_smooth, dtype=np.float32)

            # ===== Debug: 打印EMA前后差异 =====
            if np.random.rand() < 0.05:  # 随机抽样打印，防止刷屏（约每20帧一次）
                print("\n[DEBUG] 原始 wps_ego (模型输出，自车坐标):")
                print(np.round(wps_ego, 3))
                print("[DEBUG] 平滑后 wps_ego_smooth (EMA→world→ego):")
                print(np.round(wps_ego_smooth, 3))
                diffs = wps_ego_smooth - wps_ego
                print("[DEBUG] 平滑差值 Δ (smooth - raw):")
                print(np.round(diffs, 3))

            rgb_vis_disp = cv2.resize(rgb_vis, (RGB_DISP_W, RGB_DISP_H), interpolation=cv2.INTER_AREA)

            # ====== MPC 控制（完全对齐训练） ======
            # 目标速度（确保单位与训练一致；若你的模型输出为 km/h，请改为 /3.6）
            tgt_speed_mps = float(spd_pred.item())         # 如需换算： float(spd_pred.item()) / 3.6
            #tgt_speed_mps = 30.0/3.6

            x0 = build_x0_from_wp_idx(wps_ego_smooth, cur_speed_mps=speed_mps, tgt_speed_mps=tgt_speed_mps, idx=LOOKAHEAD_INDEX)
            xrefs = build_mpc_refs_zoh_idx(wps_ego_smooth, tgt_speed_mps=tgt_speed_mps, cur_speed_mps=speed_mps, idx=LOOKAHEAD_INDEX, horizon=N)

            try:
                u0    = mpc_control(x0, xrefs)  # [throttle, steer]
                if u0 is None:
                    raise RuntimeError("MPC returned None")
            except Exception as e:
                print(f"[Warn] MPC failed: {e}")
                # 兜底（很少触发）：简单 LQR 风格
                e_y, e_psi, e_v = x0.tolist()
                u0 = np.array([
                    np.clip(0.05 + 0.08*e_v, 0.0, MAX_THROTTLE),
                    np.clip(0.8*e_psi, -STEER_LIMIT, STEER_LIMIT)
                ], dtype=np.float32)

            throttle_cmd, steer_cmd, brake_cmd = apply_vehicle_control(vehicle, u0)
            print(f"speed={speed_mps:.2f} m/s, tgt={tgt_speed_mps:.2f} m/s, u0={u0}, thr={throttle_cmd:.2f}, steer={steer_cmd:.2f}")

            # # ====== 日志 ======
            # tf = vehicle.get_transform()
            # log.append({
            #     "car_x": tf.location.x,
            #     "car_y": tf.location.y,
            #     "car_yaw": tf.rotation.yaw,
            #     "speed_kmh": speed_kmh,
            #     "throttle": throttle_cmd,
            #     "steer": steer_cmd,
            #     "wp2_rel_x": float(wps_ego[1,0]) if wps_ego.shape[0] >= 2 else np.nan,
            #     "wp2_rel_y": float(wps_ego[1,1]) if wps_ego.shape[0] >= 2 else np.nan,
            #     "tgt_speed_mps": tgt_speed_mps,
            # })

            # ====== 日志（同时记录平滑前 & 平滑后 & 相对坐标） ======
            tf = vehicle.get_transform()

            # 选择 wp2 的索引（容错）
            wp_idx = min(LOOKAHEAD_INDEX, len(wps_world_smooth) - 1) if len(wps_world_smooth) > 0 else 0

            # 世界坐标（平滑后，MPC 实际用的）
            if len(wps_world_smooth) > 0:
                wp2_world_smooth = wps_world_smooth[wp_idx]
                wp2_x_world_smooth = float(wp2_world_smooth[0])
                wp2_y_world_smooth = float(wp2_world_smooth[1])
            else:
                wp2_x_world_smooth = np.nan
                wp2_y_world_smooth = np.nan

            # 世界坐标（平滑前，模型原始输出）
            if len(wps_world_raw) > 0:
                wp2_world_raw = wps_world_raw[wp_idx]
                wp2_x_world_raw = float(wp2_world_raw[0])
                wp2_y_world_raw = float(wp2_world_raw[1])
            else:
                wp2_x_world_raw = np.nan
                wp2_y_world_raw = np.nan

            # 自车坐标（模型原始输出）
            if wps_ego.shape[0] > wp_idx:
                wp2_rel_x = float(wps_ego[wp_idx, 0])
                wp2_rel_y = float(wps_ego[wp_idx, 1])
            else:
                wp2_rel_x = np.nan
                wp2_rel_y = np.nan
            # === 实际误差（MPC输入前的 e）===
            e_y   = float(np.clip(wps_ego_smooth[LOOKAHEAD_INDEX, 1], -5.0, 5.0))
            e_psi = float(np.clip(math.atan2(wps_ego_smooth[LOOKAHEAD_INDEX, 1],
                                            max(1e-3, wps_ego_smooth[LOOKAHEAD_INDEX, 0])),
                                -np.pi/2, np.pi/2))
            e_v   = float(tgt_speed_mps - speed_mps)

            log.append({
                "car_x": tf.location.x,
                "car_y": tf.location.y,
                "car_yaw": tf.rotation.yaw,
                "speed_kmh": speed_kmh,
                "throttle": throttle_cmd,
                "steer": steer_cmd,
                "brake": brake_cmd if 'brake_cmd' in locals() else 0.0,
                "tgt_speed_mps": tgt_speed_mps,

                # === 实际误差 ===
                "e_y": e_y,
                "e_psi": e_psi,
                "e_v": e_v,

                # === MPC参考状态 ===
                "ref_e_y": float(xrefs[0, 0]),
                "ref_e_psi": float(xrefs[1, 0]),
                "ref_e_v": float(xrefs[2, 0]),

                # === 控制参考 (u_ref = 0) ===
                "ref_throttle": 0.0,
                "ref_steer": 0.0,

                # === 路点（平滑前后） ===
                f"wp{LOOKAHEAD_INDEX+1}_x_smooth": wp2_x_world_smooth,
                f"wp{LOOKAHEAD_INDEX+1}_y_smooth": wp2_y_world_smooth,
                f"wp{LOOKAHEAD_INDEX+1}_x_raw": wp2_x_world_raw,
                f"wp{LOOKAHEAD_INDEX+1}_y_raw": wp2_y_world_raw,

                # === 模型输出（自车系） ===
                f"wp{LOOKAHEAD_INDEX+1}_rel_x": wp2_rel_x,
                f"wp{LOOKAHEAD_INDEX+1}_rel_y": wp2_rel_y,
            })

            # ---------- Pygame 显示 ----------
            rgb_surface   = to_surface(rgb_vis_disp)
            seg_surface   = to_surface(seg_img)
            depth_surface = to_surface(depth_vis)

            display.blit(rgb_surface, (0, 0))
            display.blit(seg_surface, (0, 360))
            display.blit(depth_surface, (480, 360))

            display.blit(font_big.render(f"Speed: {speed_kmh:.1f} km/h", True, (255,255,255)), (20, 10))
            display.blit(font_small.render(f"Tgt (m/s): {tgt_speed_mps:.2f}", True, (255,255,255)), (20, 60))
            display.blit(font_small.render(f"MPC Throttle: {throttle_cmd:.2f}", True, (0,255,0)), (20, 90))
            display.blit(font_small.render(f"MPC Steer: {steer_cmd:.2f}", True, (0,255,0)), (20, 115))

            pygame.display.flip()

    finally:
        try: camera.stop()
        except: pass
        try: vehicle.destroy()
        except: pass
        pd.DataFrame(log).to_csv("true_trajectory_log.csv", index=False)
        print("[Info] Saved true_trajectory_log.csv")
        pygame.quit()

if __name__ == "__main__":
    main()





