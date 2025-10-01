import pandas as pd
import numpy as np

def load_and_convert(file_path, lookahead_idx=2, clip_ey=5.0, clip_epsi=np.pi/2, eps=1e-3):
    """
    讀取 CSV 並輸出誤差型狀態:
      x = [e_y, e_psi, e_v] ，u = [throttle, steering]

    - lookahead_idx: 使用第幾個路點 (0~4)，預設 2 (即第三個點)
    - clip_ey: 側向誤差裁切範圍 (m)
    - clip_epsi: 航向誤差裁切範圍 (rad)
    """
    df = pd.read_csv(file_path, header=0)

    before = len(df)
    df = df.dropna(subset=["Lookahead_Speed"]).reset_index(drop=True)
    after = len(df)
    print(f"已移除 {before - after} 行包含 NaN 的数据，剩余 {after} 行可用")

    target_speed = df["Lookahead_Speed"].values
    speed        = df["Current_Speed"].values

    wx1, wy1 = df["WX1"].values, df["WY1"].values
    wx2, wy2 = df["WX2"].values, df["WY2"].values
    wx3, wy3 = df["WX3"].values, df["WY3"].values
    wx4, wy4 = df["WX4"].values, df["WY4"].values
    wx5, wy5 = df["WX5"].values, df["WY5"].values

    steering = df["Steering"].values
    throttle = df["Throttle"].values

    wxs = [wx1, wx2, wx3, wx4, wx5]
    wys = [wy1, wy2, wy3, wy4, wy5]
    assert 0 <= lookahead_idx <= 4

    x_look = wxs[lookahead_idx].astype(np.float32)
    y_look = wys[lookahead_idx].astype(np.float32)

    e_y   = np.clip(y_look, -clip_ey, clip_ey)
    e_psi = np.clip(np.arctan2(y_look, np.maximum(x_look, eps)), -clip_epsi, clip_epsi)
    e_v   = (target_speed.astype(np.float32) - speed.astype(np.float32))

    x = np.vstack([e_y, e_psi, e_v]).T.astype(np.float32)
    u = np.vstack([throttle, steering]).T.astype(np.float32)

    x_next = x[1:]
    x_true = x[:-1]
    u_true = u[:-1]

    # === 新增 route_id ===
    route_ids = (df["Scenario"] + "_" + df["Route"]).values[:-1]

    print("x_true shape:", x_true.shape)
    print("u_true shape:", u_true.shape)
    print("x_next shape:", x_next.shape)
    print("route_ids shape:", route_ids.shape)

    return x_true, u_true, x_next, route_ids


if __name__ == '__main__':
    file_path = r'D:/Master/Project/MPC/MPC_Data_final.csv'
    x_true, u_true, x_next, route_ids = load_and_convert(file_path, lookahead_idx=2)

    np.save('x_true.npy', x_true)
    np.save('u_true.npy', u_true)
    np.save('x_next.npy', x_next)

    # 用 txt 存字符串
    np.savetxt('route_ids.txt', route_ids, fmt='%s')