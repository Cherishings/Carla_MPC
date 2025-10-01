#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def smooth_curve(y, N=25):
    """简单滑动平均"""
    return np.convolve(y, np.full(N, 1.0 / N), mode="valid")

save_dir = "work/n_state=3.n_ctrl=2.T=25/0"   # 比如 seed=0 的结果

# 加载
A = np.load(os.path.join(save_dir, "A_final.npy"))
B = np.load(os.path.join(save_dir, "B_final.npy"))
Q = np.load(os.path.join(save_dir, "Q_final.npy"))

# 打印结果
print("=== Learned A ===")
print(A)

print("\n=== Learned B ===")
print(B)

print("\n=== Learned Q ===")
print(Q)

def main():
    # === 配置 ===
    exp_dir = os.path.join(SCRIPT_DIR, 'work', 'n_state=3.n_ctrl=2.T=25')  # 根据T的值修改路径
    use_smoothing = True   # 改成 False 就不会做平滑
    smooth_window = 25     # 平滑窗口大小

    # === 要画的列 ===
    keys = ["total", "model", "u", "pos", "speed"]
    labels = {
        "total": "Total Loss",
        "model": "Model Loss",
        "u": "Control Loss (u)",
        "pos": "Position Loss",
        "speed": "Speed Loss"
    }

    # === 1. 画总览图（所有 loss 在一张图上） ===
    fig, ax = plt.subplots(figsize=(8, 4))

    for seed in os.listdir(exp_dir):
        fname = os.path.join(exp_dir, seed, 'losses.csv')
        if not os.path.exists(fname):
            continue
        df = pd.read_csv(fname)

        for key in keys:
            y = df[key].values
            if use_smoothing:
                y = smooth_curve(y, N=smooth_window)
                x = np.arange(len(y)) + smooth_window
            else:
                x = np.arange(len(y))
            ax.plot(x, y, label=f"{labels[key]} (seed={seed})")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss value")
    ax.set_xlim((0., None))
    ax.set_ylim((0., None))
    ax.legend(fontsize=8)
    ax.set_title("Training losses (all)")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fname = os.path.join(SCRIPT_DIR, f"all_losses.{ext}")
        fig.savefig(fname)
        print(f"Saving to: {fname}")
    plt.close(fig)

    # === 2. 分别画每个 loss 的单独图 ===
    for key in keys:
        fig, ax = plt.subplots(figsize=(6, 3))
        for seed in os.listdir(exp_dir):
            fname = os.path.join(exp_dir, seed, 'losses.csv')
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname)

            y = df[key].values
            if use_smoothing:
                y = smooth_curve(y, N=smooth_window)
                x = np.arange(len(y)) + smooth_window
            else:
                x = np.arange(len(y))
            ax.plot(x, y, label=f"seed={seed}")

        ax.set_xlabel("Iteration")
        ax.set_ylabel(labels[key])
        ax.set_xlim((0., None))
        ax.set_ylim((0., None))
        ax.legend(fontsize=8)
        ax.set_title(f"{labels[key]} over iterations")

        fig.tight_layout()
        for ext in ["png", "pdf"]:
            fname = os.path.join(SCRIPT_DIR, f"{key}_loss.{ext}")
            fig.savefig(fname)
            print(f"Saving to: {fname}")
        plt.close(fig)


if __name__ == "__main__":
    main()