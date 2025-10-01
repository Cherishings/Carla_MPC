#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods

import os
import shutil
import argparse
import setproctitle
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_state', type=int, default=3)   # e_y, e_psi, e_v
    parser.add_argument('--n_ctrl', type=int, default=2)    # throttle, steer
    parser.add_argument('--T', type=int, default=25)        # horizon
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    t = '.'.join([f"{x}={getattr(args, x)}" for x in ['n_state', 'n_ctrl', 'T']])
    setproctitle.setproctitle('bamos.lqr.' + t + f'.{args.seed}')
    if args.save is None:
        args.save = os.path.join(args.work, t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = 'cuda' if args.cuda else 'cpu'
    torch.manual_seed(args.seed)

    n_state, n_ctrl = args.n_state, args.n_ctrl
    n_sc = n_state + n_ctrl

    # === Load initial A,B ===
    A_np = np.load('A.npy')
    B_np = np.load('B.npy')
    A = torch.tensor(A_np, dtype=torch.float32, device=device, requires_grad=True)
    B = torch.tensor(B_np, dtype=torch.float32, device=device, requires_grad=True)

    # === Load dataset ===
    x_true = np.load('x_true.npy').astype(np.float32)
    u_true = np.load('u_true.npy').astype(np.float32)
    assert x_true.shape[0] == u_true.shape[0], "x_true/u_true length mismatch"

    # === Load route_ids ===
    route_ids = np.loadtxt('route_ids.txt', dtype=str)
    assert len(route_ids) == x_true.shape[0], "route_ids length mismatch"

    # === Learnable cost: Q = L L^T (PSD) ===
    L_raw = torch.nn.Parameter(torch.randn(n_sc, n_sc, device=device))
    def get_Q():
        L = torch.tril(L_raw)
        return L @ L.T

    # === Control bounds ===
    u_lower_base = torch.tensor([0.0, -0.5], device=device).view(1, 1, -1)
    u_upper_base = torch.tensor([1.0,  0.5], device=device).view(1, 1, -1)

    # === Training hyperparams ===
    n_batch = 64
    lr = 1e-2
    lam_model = 0.1
    weight_decay_A = 1e-4
    weight_decay_B = 1e-6
    max_grad_norm = 5.0
    iters = 500

    opt = optim.Adam([A, B, L_raw], lr=lr)

    # === logging ===
    fname = os.path.join(args.save, 'losses.csv')
    with open(fname, 'w') as f:
        f.write('total,model,u,pos,speed\n')

    # ==================================================
    # 预处理 route 索引
    # ==================================================
    def preprocess_routes(route_ids):
        routes = {}
        for i, rid in enumerate(route_ids):
            if rid not in routes:
                routes[rid] = []
            routes[rid].append(i)
        for k in routes:
            routes[k] = np.array(routes[k], dtype=int)
        return routes

    routes_dict = preprocess_routes(route_ids)

    # ==================================================
    # 生成 batch
    # ==================================================
    def make_batches(x_trues, u_trues, routes_dict, T, n_batch_, device_):
        x_batch, u_batch, x_init_batch = [], [], []
        route_keys = list(routes_dict.keys())

        for _ in range(n_batch_):
            rid = np.random.choice(route_keys)
            idxs = routes_dict[rid]

            if len(idxs) <= T:
                continue

            start_pos = np.random.randint(0, len(idxs) - T)
            chosen_idxs = idxs[start_pos:start_pos + T]

            x_seq = x_trues[chosen_idxs]
            u_seq = u_trues[chosen_idxs]

            x_batch.append(x_seq)
            u_batch.append(u_seq)
            x_init_batch.append(x_seq[0])

        x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32, device=device_)
        u_batch = torch.tensor(np.array(u_batch), dtype=torch.float32, device=device_)
        x_init_batch = torch.tensor(np.array(x_init_batch), dtype=torch.float32, device=device_)

        return x_init_batch, x_batch.transpose(0, 1), u_batch.transpose(0, 1)

    # ==================================================
    # Model loss
    # ==================================================
    def compute_model_loss(A_, B_, x_arr, u_arr):
        x_t = torch.tensor(x_arr[:-1], dtype=torch.float32, device=A_.device)
        u_t = torch.tensor(u_arr[:-1], dtype=torch.float32, device=A_.device)
        x_tp1 = torch.tensor(x_arr[1:], dtype=torch.float32, device=A_.device)
        x_next_pred = x_t @ A_.T + u_t @ B_.T
        return torch.mean((x_tp1 - x_next_pred) ** 2)

    # ==================================================
    # MPC solver
    # ==================================================
    def solve_mpc(x_init, u_lower_rep, u_upper_rep, F, Qmat, pvec, u_warm=None, Bsz=1):
        try:
            x_pred, u_pred, _ = mpc.MPC(
                n_state, n_ctrl, args.T,
                u_lower=u_lower_rep, u_upper=u_upper_rep, u_init=u_warm,
                lqr_iter=100, verbose=-1,
                exit_unconverged=False,
                detach_unconverged=True,
                grad_method=GradMethods.AUTO_DIFF,
                n_batch=Bsz,
            )(x_init, QuadCost(Qmat, pvec), LinDx(F))
            return x_pred, u_pred
        except AssertionError:
            x_pred, u_pred, _ = mpc.MPC(
                n_state, n_ctrl, args.T,
                u_lower=u_lower_rep, u_upper=u_upper_rep, u_init=None,
                lqr_iter=100, verbose=-1,
                exit_unconverged=False,
                detach_unconverged=True,
                grad_method=GradMethods.AUTO_DIFF,
                n_batch=Bsz,
            )(x_init, QuadCost(Qmat, pvec), LinDx(F))
            return x_pred, u_pred

    # ==================================================
    # Loss function with reference tracking
    # ==================================================
    def get_loss(x_init, x_true_seq, u_true_seq, A_, B_):
        Bsz = x_true_seq.shape[1]
        F = torch.cat((A_, B_), dim=1).unsqueeze(0).unsqueeze(0).repeat(args.T, Bsz, 1, 1)
        u_lower_rep = u_lower_base.repeat(args.T, Bsz, 1)
        u_upper_rep = u_upper_base.repeat(args.T, Bsz, 1)
        Qmat = get_Q()
        z_ref = torch.cat([x_true_seq, u_true_seq], dim=-1)   # (T, B, n_sc)
        pvec = -2 * torch.matmul(z_ref, Qmat.T)               # (T, B, n_sc)

        x_pred, u_pred = solve_mpc(
            x_init, u_lower_rep, u_upper_rep, F, Qmat, pvec,
            u_warm=u_true_seq.detach(), Bsz=Bsz
        )

        pos_loss = torch.mean((x_true_seq[:, :, :2] - x_pred[:, :, :2]) ** 2)
        speed_loss = torch.mean((x_true_seq[:, :, 2] - x_pred[:, :, 2]) ** 2)
        u_true_clipped = torch.max(torch.min(u_true_seq, u_upper_rep), u_lower_rep)
        u_loss = torch.mean((u_true_clipped - u_pred) ** 2)
        traj_loss = u_loss + pos_loss
        return traj_loss, u_loss, pos_loss, speed_loss

    # ==================================================
    # Training loop
    # ==================================================
    for i in range(iters):
        x_init_b, x_seq_b, u_seq_b = make_batches(x_true, u_true, routes_dict, args.T, n_batch, device)
        traj_loss, u_loss, pos_loss, speed_loss = get_loss(x_init_b, x_seq_b, u_seq_b, A, B)
        model_loss = compute_model_loss(A, B, x_true, u_true)

        reg = weight_decay_A * (A**2).sum() + weight_decay_B * (B**2).sum()
        total_loss = traj_loss + lam_model * model_loss + reg

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([A, B, L_raw], max_grad_norm)
        opt.step()

        with open(fname, 'a') as f:
            f.write(f"{total_loss.item():.6f},{model_loss.item():.6f},"
                    f"{u_loss.item():.6f},{pos_loss.item():.6f},{speed_loss.item():.6f}\n")

        if (i % 10) == 0:
            print(f"{i:04d}: Total={total_loss.item():.4f} | U={u_loss.item():.4f} "
                  f"| Pos={pos_loss.item():.4f} | Spd={speed_loss.item():.4f} | Model={model_loss.item():.4f}")

    # ==================================================
    # Save final results
    # ==================================================
    A_final = A.detach().cpu().numpy()
    B_final = B.detach().cpu().numpy()
    Q_final = get_Q().detach().cpu().numpy()
    L_final = L_raw.detach().cpu().numpy()

    np.save(os.path.join(args.save, "A_final.npy"), A_final)
    np.save(os.path.join(args.save, "B_final.npy"), B_final)
    np.save(os.path.join(args.save, "Q_final.npy"), Q_final)
    np.save(os.path.join(args.save, "L_final.npy"), L_final)

    print("\n=== Training finished. Results saved to:", args.save)
    print("A =\n", A_final)
    print("B =\n", B_final)
    print("Q =\n", Q_final)


if __name__ == '__main__':
    main()