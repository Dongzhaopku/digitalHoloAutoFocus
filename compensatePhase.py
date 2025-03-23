import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 定义 autoCompensation 函数
def auto_compensation(Pha):
    global M, N
    M, N = Pha.shape

    global WPHI, X, Y, Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9
    global dWPHI_dX, dWPHI_dY
    global dZ1_dX, dZ2_dX, dZ3_dX, dZ4_dX, dZ5_dX, dZ6_dX, dZ7_dX, dZ8_dX, dZ9_dX
    global dZ1_dY, dZ2_dY, dZ3_dY, dZ4_dY, dZ5_dY, dZ6_dY, dZ7_dY, dZ8_dY, dZ9_dY

    # 生成 x, y 网格坐标
    x = np.linspace(-N/2 + 1, N/2, N) / (N / 2)
    y = np.linspace(-M/2 + 1, M/2, M) / (M / 2)
    X, Y = np.meshgrid(x, y)

    # 计算 Z1 到 Z9
    Z1 = 2 * X
    Z2 = 2 * Y
    Z3 = np.sqrt(3) * (-1 + 2 * (X**2 + Y**2))

    Z4 = np.sqrt(6) * 2 * X * Y
    Z5 = np.sqrt(6) * (X**2 - Y**2)

    Z6 = np.sqrt(8) * (-2 * Y + 3 * Y * (X**2 + Y**2))
    Z7 = np.sqrt(8) * (-2 * X + 3 * X * (X**2 + Y**2))
    Z8 = np.sqrt(8) * Y * (3 * X**2 - Y**2)
    Z9 = np.sqrt(8) * X * (-3 * Y**2 + X**2)
    WPHI = Pha

    dWPHI_dX = np.gradient(WPHI, axis=1)
    dWPHI_dY = np.gradient(WPHI, axis=0)
    # 沿 X 方向的导数
    dZ1_dX = np.gradient(Z1, axis=1)
    dZ2_dX = np.gradient(Z2, axis=1)
    dZ3_dX = np.gradient(Z3, axis=1)
    dZ4_dX = np.gradient(Z4, axis=1)
    dZ5_dX = np.gradient(Z5, axis=1)
    dZ6_dX = np.gradient(Z6, axis=1)
    dZ7_dX = np.gradient(Z7, axis=1)
    dZ8_dX = np.gradient(Z8, axis=1)
    dZ9_dX = np.gradient(Z9, axis=1)

    # 沿 Y 方向的导数
    dZ1_dY = np.gradient(Z1, axis=0)
    dZ2_dY = np.gradient(Z2, axis=0)
    dZ3_dY = np.gradient(Z3, axis=0)
    dZ4_dY = np.gradient(Z4, axis=0)
    dZ5_dY = np.gradient(Z5, axis=0)
    dZ6_dY = np.gradient(Z6, axis=0)
    dZ7_dY = np.gradient(Z7, axis=0)
    dZ8_dY = np.gradient(Z8, axis=0)
    dZ9_dY = np.gradient(Z9, axis=0)

    sample_rate = 5
    dWPHI_dX = dWPHI_dX[::sample_rate, ::sample_rate]
    dWPHI_dY = dWPHI_dY[::sample_rate, ::sample_rate]
    dZ1_dX = dZ1_dX[::sample_rate, ::sample_rate]
    dZ2_dX = dZ2_dX[::sample_rate, ::sample_rate]
    dZ3_dX = dZ3_dX[::sample_rate, ::sample_rate]
    dZ4_dX = dZ4_dX[::sample_rate, ::sample_rate]
    dZ5_dX = dZ5_dX[::sample_rate, ::sample_rate]
    dZ6_dX = dZ6_dX[::sample_rate, ::sample_rate]
    dZ7_dX = dZ7_dX[::sample_rate, ::sample_rate]
    dZ8_dX = dZ8_dX[::sample_rate, ::sample_rate]
    dZ9_dX = dZ9_dX[::sample_rate, ::sample_rate]

    dZ1_dY = dZ1_dY[::sample_rate, ::sample_rate]
    dZ2_dY = dZ2_dY[::sample_rate, ::sample_rate]
    dZ3_dY = dZ3_dY[::sample_rate, ::sample_rate]
    dZ4_dY = dZ4_dY[::sample_rate, ::sample_rate]
    dZ5_dY = dZ5_dY[::sample_rate, ::sample_rate]
    dZ6_dY = dZ6_dY[::sample_rate, ::sample_rate]
    dZ7_dY = dZ7_dY[::sample_rate, ::sample_rate]
    dZ8_dY = dZ8_dY[::sample_rate, ::sample_rate]
    dZ9_dY = dZ9_dY[::sample_rate, ::sample_rate]



    # 定义初始参数
    a0 = np.zeros(9)

    # 使用 fminunc 函数进行优化
    result = minimize(VARSampling, a0)
    a = result.x

    # 计算补偿后的相位
    Phi = WPHI + a[0] * Z1 + a[1] * Z2 + a[2] * Z3 + a[3] * Z4 + a[4] * Z5 + a[5] * Z6 + a[6] * Z7 + a[7] * Z8 + a[8] * Z9
    PhiR = np.mod(Phi + np.pi, 2 * np.pi) - np.pi  # wrapToPi equivalent
    return PhiR

# 定义 VARSampling 函数
def VARSampling(a):
    global dWPHI_dX, dWPHI_dY
    global dZ1_dX, dZ2_dX, dZ3_dX, dZ4_dX, dZ5_dX, dZ6_dX, dZ7_dX, dZ8_dX, dZ9_dX
    global dZ1_dY, dZ2_dY, dZ3_dY, dZ4_dY, dZ5_dY, dZ6_dY, dZ7_dY, dZ8_dY, dZ9_dY

    Dx = dWPHI_dX + a[0] * dZ1_dX + a[1] * dZ2_dX + a[2] * dZ3_dX + a[3] * dZ4_dX + a[4] * dZ5_dX + a[5] * dZ6_dX + a[6] * dZ7_dX + a[7] * dZ8_dX + a[8] * dZ9_dX
    Dy = dWPHI_dY + a[0] * dZ1_dY + a[1] * dZ2_dY + a[2] * dZ3_dY + a[3] * dZ4_dY + a[4] * dZ5_dY + a[5] * dZ6_dY + a[6] * dZ7_dY + a[7] * dZ8_dY + a[8] * dZ9_dY

    Dx = np.mod(Dx + np.pi, 2 * np.pi) - np.pi  # wrapToPi equivalent
    Dy = np.mod(Dy + np.pi, 2 * np.pi) - np.pi  # wrapToPi equivalent

    var = np.sum(np.sqrt(Dx**2 + Dy**2))

    return var
