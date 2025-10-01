import numpy as np

def estimate_linear_model(x_true, u_true, x_next):
    """
    用最小二乘拟合线性系统:
        x_{k+1} ≈ A x_k + B u_k
    
    输入:
        x_true: (N, n) 状态序列
        u_true: (N, m) 控制输入序列
        x_next: (N, n) 下一时刻状态
    
    输出:
        A: (n, n)
        B: (n, m)
    """
    N, n = x_true.shape   # N样本数, n状态维度
    _, m = u_true.shape   # m输入维度

    # 拼接成 Z = [x; u]，shape = (N, n+m)
    Z = np.hstack([x_true, u_true])     # (N, n+m)

    # 最小二乘解: theta = (Z^T Z)^-1 Z^T X_next
    theta, residuals, rank, s = np.linalg.lstsq(Z, x_next, rcond=None)

    # 拆分 theta → A, B
    A = theta[:n, :].T   # (n, n)
    B = theta[n:, :].T   # (n, m)

    return A, B


if __name__ == '__main__':
    # 载入数据
    x_true = np.load('x_true.npy')
    u_true = np.load('u_true.npy')
    x_next = np.load('x_next.npy')

    print("数据维度:", x_true.shape, u_true.shape, x_next.shape)

    # 拟合 A, B
    A, B = estimate_linear_model(x_true, u_true, x_next)

    print("\nA 矩阵:")
    print(A)
    print("\nB 矩阵:")
    print(B)

    # 保存结果
    np.save('A.npy', A)
    np.save('B.npy', B)