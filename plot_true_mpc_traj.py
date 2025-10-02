
import pandas as pd
import matplotlib.pyplot as plt

# 读取记录文件
df = pd.read_csv("true_trajectory_log.csv")

# 实际轨迹
car_x = df["car_x"].values
car_y = df["car_y"].values

# 目标路点（wp2）
wp2_x = df["wp2_x"].values
wp2_y = df["wp2_y"].values

plt.figure(figsize=(10, 8))
plt.plot(car_x, car_y, label="Ego trajectory (actual)", color="blue", linewidth=2)
plt.plot(wp2_x, wp2_y, label="Target WP2 trajectory", color="red", linestyle="--")

# 开始点和结束点标记
plt.scatter(car_x[0], car_y[0], color="green", marker="o", s=80, label="Start (ego)")
plt.scatter(car_x[-1], car_y[-1], color="black", marker="x", s=80, label="End (ego)")

plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Ego trajectory vs Target WP2")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()