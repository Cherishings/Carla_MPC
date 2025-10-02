# -*- coding: utf-8 -*-
"""
CARLA 可视化 + MPC 自动驾驶（含油门限幅/斜率、目标速度缓升、30±2 km/h 速度限制）：
顶部整幅 RGB（叠加 5 个预测路点与顺滑曲线）
底部左：语义分割；底部右：离散深度

说明：
- 只“换模型权重”，其余验证逻辑与您现有脚本保持一致（裁剪、MPC、可视化、限速、EMA 等均不改）。
- 为避免上次因权重路径错误导致的传感器崩溃，本版在创建相机前先校验并加载权重。
- 刹车分支按训练一致使用 logits；推理时对 logits 做 sigmoid 得到 brake_prob（阈值判断逻辑不变）。
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
import cvxpy as cp  # MPC 依赖：确保已安装 cvxpy 和 osqp

# ============================= 可调参数 =============================
CKPT_PATH = "/home/ali_carla/Desktop/Ali/Leader_Board_2/CARLA_Leaderboard_2.0/CARLA_Leaderboard_20/PythonAPI/examples/Swin_WP5_TP_mps_20250820_133420/swin_wp5_best.pth" #权重

# —— 相机与窗口 ——（相机≥裁剪尺寸；UI 可随意缩放，不影响模型输入）
CAM_W, CAM_H = 1024, 512         # 相机原始分辨率（≥ 1024×384）
FOV_DEG = 90.0
WIN_W, WIN_H = 960, 720          # 窗口总大小（上 960×360；下两块 480×360）
TILE_W, TILE_H = 480, 360        # 底部左右两块面板的尺寸
RGB_DISP_W, RGB_DISP_H = 960, 360  # 顶部展示尺寸（仅显示，不参与模型）

# —— 路点绘制与平滑 —— 
Z_OFFSET = 0.1
EMA_DECAY = 0.85
SAMPLES_PER_SEG = 24
USE_AMP = False

# —— 模型/数据设定（与训练一致） —— 
INPUT_W, INPUT_H = 224, 224
NUM_CLASSES = 7
NUM_DEPTH_BINS = 8
NUM_WAYPOINTS = 5
TARGET_POINT = (20.0, 0.0)       # 训练时的引导点（自车坐标系，m）

# —— 自动驾驶控制 —— 
AUTO_DRIVE = True
STEER_LIMIT = 0.5
BRAKE_THRESH = 0.95
LOOKAHEAD_INDEX = 1

# === 全局油门与斜率限制 ===
MAX_THROTTLE = 0.5
THROTTLE_GAIN = 1.5
THROTTLE_SLEW = 0.05

# === 目标速度“缓升” + 30±2 km/h 限速带 ===
SPEED_RAMP_A = 1.0
USE_MODEL_SPEED = False
SPEED_LIMIT_KMH = 30.0
SPEED_BAND_KMH  = 2.0
SPEED_MIN_KMH   = SPEED_LIMIT_KMH - SPEED_BAND_KMH
SPEED_MAX_KMH   = SPEED_LIMIT_KMH + SPEED_BAND_KMH
SPEED_SET_KMH   = SPEED_LIMIT_KMH
SPEED_MIN_MPS   = SPEED_MIN_KMH / 3.6
SPEED_MAX_MPS   = SPEED_MAX_KMH / 3.6
SPEED_SET_MPS   = SPEED_SET_KMH / 3.6

# ============================= 基础设置 =============================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {device}")

# —— 和训练一致的预处理 —— #
transform = A.Compose([
    A.Resize(INPUT_H, INPUT_W),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

def apply_training_crop(img):
    """
    训练时用到的裁剪：取“上部 384 × 水平居中 1024”。
    img: H×W×3 (RGB)
    """
    H, W = img.shape[:2]
    ch, cw = 384, 1024
    if H < ch or W < cw:
        raise ValueError(f"Camera frame {W}x{H} is smaller than training crop {cw}x{ch}. "
                         "Please set camera to >= 1024x384.")
    side = (W - cw) // 2
    return img[:ch, side:side+cw, :]

# ============================= 模型（与训练一致） =============================
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

        # 训练时用 pretrained=True; 这里会被 checkpoint 覆盖
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

        self.speed_head = nn.Linear(d_model, 1)
        # 与训练一致：输出 logits（推理时再做 sigmoid）
        self.brake_head = nn.Linear(d_model, 1)

    def forward(self, x, target_point, current_speed):
        feats = self.encoder(x)                           # timm==1.0.16 返回 NHWC
        feats = [f.permute(0,3,1,2).contiguous() for f in feats]  # -> NCHW

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
        speed = self.speed_head(control_feat)  # 目标速度（训练里是什么单位就是什么单位）
        brake_logits = self.brake_head(control_feat)  # logits
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
# === 在自车前方 20 m 生成静止 NPC（基于道路 waypoint，朝向/车道对齐）===
def spawn_npc_ahead(world, ego, dist_list=(20.0,), hold_still=True):
    """
    dist_list: 优先尝试的前向距离（米），可给多个备选；这里默认只试 20 m
    hold_still: True 则把 NPC 钉住不动（静态前车）
    返回：npc actor 或 None
    """
    lib = world.get_blueprint_library()
    # 任选一台车（优先 model3）
    bp_list = lib.filter("vehicle.tesla.model3")
    npc_bp = bp_list[0] if len(bp_list) > 0 else lib.find("vehicle.audi.tt")
    npc_bp.set_attribute("role_name", "npc_ahead")

    m = world.get_map()
    ego_tf = ego.get_transform()

    # 从自车所在道路中心开始，沿车头方向找前方 waypoint
    w0 = m.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if w0 is None:
        print("[NPC] Ego 不在可驾驶车道上，放弃生成。")
        return None

    for d in dist_list:
        w_list = w0.next(float(d))
        if not w_list:
            continue
        tf = w_list[0].transform
        tf.location.z += 0.3  # 略抬高，避免底盘卡地

        npc = world.try_spawn_actor(npc_bp, tf)
        if npc is not None:
            if hold_still:
                # 静止：完全不动（更稳），或者改成 npc.apply_control(brake=1.0)
                npc.set_autopilot(False)
                npc.set_simulate_physics(False)
            print(f"[NPC] 前方 {d:.1f} m 生成成功，id={npc.id}")

            # 画一条调试线，便于从画面里确认位置
            world.debug.draw_line(
                ego_tf.location + carla.Location(z=1.5),
                npc.get_transform().location + carla.Location(z=1.5),
                thickness=0.1, color=carla.Color(0,128,0), life_time=1.0
            )
            return npc

    # 若 waypoint 方案都失败（比如位置被占），兜底：按车头 forward vector 推 20 m 试一次
    try:
        fwd = ego_tf.get_forward_vector()
        loc = ego_tf.location + carla.Location(x=fwd.x * 20.0, y=fwd.y * 20.0, z=0.3)
        tf  = carla.Transform(loc, ego_tf.rotation)
        npc = world.try_spawn_actor(npc_bp, tf)
        if npc is not None:
            if hold_still:
                npc.set_autopilot(False)
                npc.set_simulate_physics(False)
            print("[NPC] Waypoint 失败，采用 forward-vector 兜底生成成功。")
            world.debug.draw_line(
                ego_tf.location + carla.Location(z=1.5),
                npc.get_transform().location + carla.Location(z=1.5),
                thickness=0.1, color=carla.Color(255,165,0), life_time=5.0
            )
            return npc
    except Exception as e:
        print(f"[NPC] 兜底生成异常：{e}")

    print("[NPC] 生成失败（位置被占或不合法）。")
    return None



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
    def __init__(self, decay=EMA_DECAY):
        self.decay = decay
        self.state = None  # [(x,y,z), ...] 长度=5

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

def catmull_rom_spline(pts2d, samples_per_seg=SAMPLES_PER_SEG):
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
            curve = catmull_rom_spline(pts2d, samples_per_seg=SAMPLES_PER_SEG)
            cv2.polylines(rgb_vis, [curve.reshape(-1,1,2)], False, color, 2, cv2.LINE_AA)
        else:
            arr = np.array(pts2d, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(rgb_vis, [arr], False, color, 2, cv2.LINE_AA)

# ============================= MPC（与你原来一致） =============================
A = np.array([
    [ 0.6800279,   0.4065171,   0.00412744],
    [ 0.00738894,  0.6344572,  -0.01294562],
    [ 0.01396624, -0.03162604,  0.57808036]
])

B = np.array([
    [-0.03505046, -0.3509874 ],
    [ 0.02046476, -0.00288475],
    [ 0.00736699, -0.07007284]
])

Q = np.array([
    [ 9.0583766e-01,  6.4924471e-02, -1.1931288e+00,  1.0546644e-01, -1.1077980e+00],
    [ 6.4924471e-02,  1.1044665e-02, -3.0973318e-01,  4.3207701e-02, -1.5907243e-02],
    [-1.1931288e+00, -3.0973318e-01,  1.5252293e+01, -2.5793946e+00,  7.5840425e-01],
    [ 1.0546644e-01,  4.3207701e-02, -2.5793946e+00,  4.7711173e-01, -5.4839104e-01],
    [-1.1077980e+00, -1.5907243e-02,  7.5840425e-01, -5.4839104e-01,  1.1870452e+01]
])

p = np.array([-0.17293791,  0.02827822,  0.6870698,  -0.11809217,  0.16456106])

nx = A.shape[0]
nu = B.shape[1]
Q_xx = Q[:nx, :nx]
p_x  = p[:nx]
N = 10

def mpc_control(x0):
    x = cp.Variable((nx, N + 1))
    u = cp.Variable((nu, N))
    cost = 0
    constr = [x[:, 0] == x0]
    for k in range(N):
        constr += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
        xu = cp.hstack([x[:, k], u[:, k]])
        cost += 0.5 * cp.quad_form(xu, Q) + p @ xu
        constr += [u[0, k] <= MAX_THROTTLE, u[0, k] >= 0.0]
        constr += [u[1, k] <=  STEER_LIMIT, u[1, k] >= -STEER_LIMIT]
    cost += 0.5 * cp.quad_form(x[:, N], Q_xx) + p_x @ x[:, N]
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"MPC failed: {prob.status}")
    u0 = u[:, 0].value
    return np.asarray(u0).reshape(-1)

def build_mpc_state(wps_ego, cur_speed_mps, tgt_speed_mps):
    if wps_ego is None or len(wps_ego) == 0:
        return np.zeros(3, dtype=np.float32)
    idx = min(max(LOOKAHEAD_INDEX, 0), len(wps_ego)-1)
    x_look, y_look = float(wps_ego[idx, 0]), float(wps_ego[idx, 1])
    if x_look < 1.0 and len(wps_ego) >= 2:
        x_look, y_look = float(wps_ego[-1, 0]), float(wps_ego[-1, 1])
    e_y = float(np.clip(y_look, -5.0, 5.0))
    e_psi = float(np.clip(math.atan2(y_look, max(1e-3, x_look)), -np.pi/2, np.pi/2))
    e_v = float(np.clip(tgt_speed_mps - cur_speed_mps, -10.0, 10.0))
    return np.array([e_y, e_psi, e_v], dtype=np.float32)

def apply_vehicle_control(vehicle, u, brake_prob, cur_speed_mps, prev_throttle):
    thr = float(np.clip(u[0], 0.0, MAX_THROTTLE)) * THROTTLE_GAIN
    thr = float(np.clip(thr, 0.0, MAX_THROTTLE))
    steer = float(np.clip(u[1], -STEER_LIMIT, STEER_LIMIT))
    brake = 0.0
    if brake_prob > BRAKE_THRESH and cur_speed_mps > 0.2:
        brake = max(brake, 0.5); thr = 0.0
    if cur_speed_mps > SPEED_MAX_MPS:
        overshoot = cur_speed_mps - SPEED_MAX_MPS
        brake = max(brake, float(np.clip(0.2 + 0.3*overshoot, 0.3, 1.0))); thr = 0.0
    d = np.clip(thr - prev_throttle, -THROTTLE_SLEW, THROTTLE_SLEW)
    thr = float(np.clip(prev_throttle + d, 0.0, MAX_THROTTLE))
    prev_throttle = thr
    vehicle.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brake))
    return thr, steer, brake, prev_throttle

# ============================= 主流程 =============================
def main():
    # ---- pygame ----
    pygame.init()
    display = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("RGB | Segmentation | Depth | MPC")
    clock = pygame.time.Clock()
    font_big = pygame.font.Font(None, 48)
    font_small = pygame.font.Font(None, 28)

    # 先校验并加载模型，避免权重错误导致传感器遗留
    model = load_model(CKPT_PATH)

    # ---- CARLA ----
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    TARGET_TOWN = "Town05"
    world = client.get_world()
    cur = world.get_map().name
    if TARGET_TOWN not in cur:
        world = client.load_world(TARGET_TOWN)
        world.wait_for_tick()
        print(f"[Info] Loaded map: {world.get_map().name}")
    bp = world.get_blueprint_library()
    
    cam_bp = bp.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAM_W))
    cam_bp.set_attribute("image_size_y", str(CAM_H))
    cam_bp.set_attribute("fov", str(FOV_DEG))

    # —— 关闭后处理/泛光/镜头光晕（若你的 CARLA 版本支持）——
    for k, v in [
        ("enable_postprocess_effects", "false"),
        ("motion_blur_intensity", "0.0"),
        ("lens_flare_intensity", "0.0"),
        ("chromatic_aberration_intensity", "0.0"),
        ("gamma_correction", "2.2"),  # 可选，让色调更稳定
    ]:
        try:
            if cam_bp.has_attribute(k):
                cam_bp.set_attribute(k, v)
        except Exception:
            pass
    # 车辆
    veh_bp = bp.find("vehicle.lincoln.mkz_2017")
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(veh_bp, spawn_point) or world.spawn_actor(veh_bp, spawn_point)

    # 在自车正前方 20 m 生成静止 NPC
    npc = spawn_npc_ahead(world, vehicle, dist_list=(15.0,), hold_still=True)


    # 相机（分辨率≥裁剪尺寸）
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

    # EMA
    ema = EMAFilter(decay=EMA_DECAY)

    # 颜色表（语义可视化）
    color_map = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [0, 255, 0],
        3: [255, 0, 0],
        4: [255, 255, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
    }

    # 目标速度缓升状态
    prev_t = time.time()
    vel0 = vehicle.get_velocity()
    speed0_kmh = 3.6 * (vel0.x**2 + vel0.y**2 + vel0.z**2) ** 0.5
    tgt_speed_filtered = float(np.clip(speed0_kmh/3.6, SPEED_MIN_MPS, SPEED_MAX_MPS))  # m/s

    prev_throttle = 0.0
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
            speed_kmh = 3.6 * (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            speed_mps = speed_kmh / 3.6

            # ====== 与训练一致的裁剪 + 预处理 ======
            frame_crop = apply_training_crop(frame)                  # 先裁成 384×1024
            aug = transform(image=frame_crop)                        # 再 Resize(224) + Normalize
            inp = aug["image"].unsqueeze(0).to(device)
            tgt_pt = torch.tensor([TARGET_POINT], dtype=torch.float32, device=device)
            # 注意：这里按你“老脚本”的做法，把 km/h 作为 meta 速度输入，保持完全一致
            cur_sp_for_meta = torch.tensor([speed_kmh], dtype=torch.float32, device=device)

            # 推理
            with torch.no_grad():
                if USE_AMP:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        seg_logits, depth_logits, wp_pred, spd_pred, brk_logits = model(inp, tgt_pt, cur_sp_for_meta)
                else:
                    seg_logits, depth_logits, wp_pred, spd_pred, brk_logits = model(inp, tgt_pt, cur_sp_for_meta)

            # 把 logits 转成概率，以便与 BRAKE_THRESH 配合（外部逻辑保持不变）
            brk_prob = torch.sigmoid(brk_logits)

            # 语义可视化
            seg_lbl = torch.argmax(seg_logits.squeeze(0), dim=0).cpu().numpy()
            seg_rgb = np.zeros((seg_lbl.shape[0], seg_lbl.shape[1], 3), dtype=np.uint8)
            for k, c in color_map.items():
                seg_rgb[seg_lbl == k] = c
            seg_img = cv2.resize(seg_rgb, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)

            # 深度可视化（若有）
            if depth_logits is not None:
                depth_cls = torch.argmax(depth_logits.squeeze(0), dim=0).cpu().numpy()
                depth_255 = (depth_cls.astype(np.float32) / (NUM_DEPTH_BINS - 1) * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_255, cv2.COLORMAP_PLASMA)
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
                depth_vis = cv2.resize(depth_vis, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
            else:
                depth_vis = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)

            # 顶部 RGB 叠加路点（在原始相机帧尺寸下投影，再整体缩放展示）
            rgb_vis = frame.copy()   # 1024×512
            wps_ego = wp_pred.squeeze(0).detach().cpu().numpy()  # (5,2)
            wps_world_raw = ego_to_world(vehicle, [(float(x), float(y)) for x, y in wps_ego], z_offset=Z_OFFSET)
            wps_world_smooth = ema.update(wps_world_raw)
            K = get_camera_K(camera)
            W2C = world_to_camera_matrix(camera)
            pix = project_world_to_image(K, W2C, wps_world_smooth)
            draw_waypoints_on_rgb(rgb_vis, pix, draw_ids=True, smooth=True, color=(0,255,0))
            # 显示前缩放到 UI 尺寸
            rgb_vis_disp = cv2.resize(rgb_vis, (RGB_DISP_W, RGB_DISP_H), interpolation=cv2.INTER_AREA)

            # =========== 控制：MPC or 手动 ===========
            throttle_cmd = 0.0
            steer_cmd = 0.0
            brake_cmd = 0.0

            if AUTO_DRIVE:
                now = time.time()
                dt = max(1e-3, now - prev_t)
                prev_t = now

                # 模型速度（单位必须与训练一致；若训练是 km/h 而这里想用 m/s，请在此处统一转换）
                model_tgt_mps = float(spd_pred.item())  # 若模型输出为 km/h，请改为 /3.6

                desired_tgt = (float(np.clip(model_tgt_mps, SPEED_MIN_MPS, SPEED_MAX_MPS))
                               if USE_MODEL_SPEED else SPEED_SET_MPS)
                max_step = SPEED_RAMP_A * dt
                tgt_speed_filtered += float(np.clip(desired_tgt - tgt_speed_filtered, -max_step, max_step))
                tgt_speed_filtered = float(np.clip(tgt_speed_filtered, SPEED_MIN_MPS, SPEED_MAX_MPS))

                brake_prob = float(brk_prob.item())

                x0 = build_mpc_state(wps_ego, cur_speed_mps=speed_mps, tgt_speed_mps=tgt_speed_filtered)

                try:
                    u0 = mpc_control(x0)  # [throttle, steer]
                except Exception as e:
                    print(f"[Warn] MPC failed: {e}. Use fallback.")
                    e_y, e_psi, e_v = x0.tolist()
                    u0 = np.array([
                        np.clip(0.05 + 0.08*e_v, 0.0, MAX_THROTTLE),
                        np.clip(0.8*e_psi, -STEER_LIMIT, STEER_LIMIT)
                    ], dtype=np.float32)

                throttle_cmd, steer_cmd, brake_cmd, prev_throttle = apply_vehicle_control(
                    vehicle, u0, brake_prob=brake_prob, cur_speed_mps=speed_mps, prev_throttle=prev_throttle
                )
            else:
                keys = pygame.key.get_pressed()
                ctrl = carla.VehicleControl()
                ctrl.throttle = 0.5 + (0.5 if keys[pygame.K_w] else 0.0)
                ctrl.brake = 1.0 if keys[pygame.K_s] else 0.0
                ctrl.steer = (-0.3 if keys[pygame.K_a] else (0.3 if keys[pygame.K_d] else 0.0))
                ctrl.hand_brake = keys[pygame.K_SPACE]
                vehicle.apply_control(ctrl)
                throttle_cmd, steer_cmd, brake_cmd = ctrl.throttle, ctrl.steer, ctrl.brake

            # ====== 从这里开始加日志记录 ======
            tf = vehicle.get_transform()
            log.append({
                "car_x": tf.location.x,
                "car_y": tf.location.y,
                "car_yaw": tf.rotation.yaw,
                "speed_kmh": speed_kmh,
                "throttle": throttle_cmd,
                "steer": steer_cmd,
                "brake": brake_cmd,
                "wp2_x": wps_world_raw[min(LOOKAHEAD_INDEX, len(wps_world_raw)-1)][0],  # 世界坐标
                "wp2_y": wps_world_raw[min(LOOKAHEAD_INDEX, len(wps_world_raw)-1)][1],  # 世界坐标
            })

            # ---------- Pygame 显示 ----------
            rgb_surface   = to_surface(rgb_vis_disp)      # 顶部整幅（已缩放到 960×360）
            seg_surface   = to_surface(seg_img)           # 左下 480×360
            depth_surface = to_surface(depth_vis)         # 右下 480×360

            display.blit(rgb_surface, (0, 0))
            display.blit(seg_surface, (0, 360))
            display.blit(depth_surface, (480, 360))

            # 文本信息
            display.blit(font_big.render(f"Speed: {speed_kmh:.1f} km/h", True, (255,255,255)), (20, 10))
            display.blit(font_small.render(f"Speed limit: {SPEED_MIN_KMH:.0f}~{SPEED_MAX_KMH:.0f} km/h", True, (255,255,255)), (20, 60))
            display.blit(font_small.render(f"Desired (m/s): {tgt_speed_filtered:.2f}", True, (255,255,255)), (20, 90))
            display.blit(font_small.render(f"Brake prob: {brake_prob:.2f}", True, (255,255,255)), (20, 120))
            if AUTO_DRIVE:
                display.blit(font_small.render(f"MPC Throttle: {throttle_cmd:.2f}", True, (0,255,0)), (20, 150))
                display.blit(font_small.render(f"MPC Steer: {steer_cmd:.2f}", True, (0,255,0)), (20, 175))
                display.blit(font_small.render(f"MPC Brake: {brake_cmd:.2f}", True, (0,255,0)), (20, 200))
            else:
                display.blit(font_small.render("Manual: WASD/SPACE", True, (255,255,0)), (20, 150))

            pygame.display.flip()

    finally:
        try: camera.stop()
        except: pass
        try:
            if npc is not None: npc.destroy()
        except: pass
        try: vehicle.destroy()
        except: pass
        pd.DataFrame(log).to_csv("trajectory_log.csv", index=False)
        print("[Info] Saved trajectory_log.csv")
        pygame.quit()

if __name__ == "__main__":
    main()