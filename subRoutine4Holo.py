"""
Sub Routines for Holography
Written by Zhao Dong
Nov. 3, 2024 A. D. (Begin to write)
handandong@163.com
"""

import numpy as np
from scipy.optimize import minimize
from compensatePhase import auto_compensation
import matplotlib.pyplot as plt

def auto_recover_holo(img):
    # 读入数字全息图，输出物光复振幅
    # 对img进行傅里叶变换
    FIU = np.fft.fftshift(np.fft.fft2(img))
    plt.figure()
    plt.imshow(np.log(1 + np.abs(FIU)), cmap='gray')
    plt.title('Fourier Transform')
    plt.axis('off')
    plt.show()
    # 将Fourier结果输入到auto_filter(FIU)
    filtered_FIU, filter_mask, posX, posY = auto_filter(FIU)

    plt.figure()
    plt.imshow(np.log(1 + np.abs(filtered_FIU)), cmap='gray')
    plt.title('Fourier Transform')
    plt.axis('off')
    plt.show()
    # 进行逆变换
    filtered_img = np.fft.ifft2(np.fft.ifftshift(filtered_FIU))

    # 进行相位补偿
    recovered_img = auto_phase_compensation(filtered_img)

    return recovered_img

def auto_phase_compensation(img):
    pha = np.angle(img)
    pha_compensated = auto_compensation(pha)
    # pha_compensated = pha
    compensated_img = np.abs(img) * np.exp(1j*pha_compensated)
    return compensated_img

def auto_filter(FIU):
    """
    自动滤波，并将滤出的频谱移至中心。

    参数:
    FIU: 输入频谱图 (二维数组)

    返回:
    FIUnew: 经过滤波后的频谱图
    Filter: 应用的滤波器
    posX: X 轴中心位置
    posY: Y 轴中心位置
    """
    per = 0.9
    M, N = FIU.shape
    Mc, Nc = M // 2, N // 2
    IFIU = np.abs(FIU)

    fx_IU = np.sum(IFIU[0:Mc, :], axis=0)  # 各行求和
    # 确保 per 是一个浮点数
    per = 0.9

    # 使用 int() 确保切片索引为整数
    maxfx1 = np.max(fx_IU[:int(N // 2 * per)])
    posMax1 = np.argmax(fx_IU[:int(N // 2 * per)])

    maxfx2 = np.max(fx_IU[N - 1 - int(N // 2 * per - 1):N])
    posMax2 = np.argmax(fx_IU[N - 1 - int(N // 2 * per - 1):N])

    if maxfx1 >= maxfx2:
        posX = posMax1
        fy_IU = np.sum(IFIU[0:Mc, :posX], axis=1)
        maxfy, posY = np.max(fy_IU), np.argmax(fy_IU)
    else:
        posX = N - posMax2 + 1
        fy_IU = np.sum(IFIU[0:Mc, posX:], axis=1)
        maxfy, posY = np.max(fy_IU), np.argmax(fy_IU)

    Dis = np.sqrt((posX - Nc)**2 + (posY - Mc)**2)
    R = Dis / 3
    Filter = get_circle(FIU, R, posX, posY)
    FIUslt = Filter * FIU
    FIUnew = np.roll(FIUslt, (Mc - posY, Nc - posX), axis=(0, 1))

    return FIUnew, Filter, posX, posY

def get_circle(FIU, R, posX, posY):
    """
    创建一个圆形滤波器。

    参数:
    FIU: 输入频谱图 (二维数组)
    R: 圆的半径
    posX: 圆心 X 轴位置
    posY: 圆心 Y 轴位置

    返回:
    CircleFilter: 圆形滤波器
    """
    Y, X = np.ogrid[:FIU.shape[0], :FIU.shape[1]]
    dist = np.sqrt((X - posX)**2 + (Y - posY)**2)
    CircleFilter = dist <= R
    return CircleFilter.astype(float)


from scipy.fft import dct, idct
def least_squares(wrapped_phase):
    P = wrapped_phase
    M, N = P.shape  # Get the size of the wrapped phase matrix

    # Initialize gradient matrices
    dx = np.zeros((M, N))
    dy = np.zeros((M, N))

    # Calculate gradients in x and y directions
    dx[:-1, :] = P[1:, :] - P[:-1, :]
    dx = dx - np.pi * np.floor(dx / np.pi + 0.5)  # Wrap the gradients to (-pi, pi)

    dy[:, :-1] = P[:, 1:] - P[:, :-1]
    dy = dy - np.pi * np.floor(dy / np.pi + 0.5)  # Wrap the gradients to (-pi, pi)

    # Initialize p matrices
    p = np.zeros((M, N))
    p1 = np.zeros((M, N))
    p2 = np.zeros((M, N))

    # Calculate p1 and p2
    p1[1:, :] = dx[1:, :] - dx[:-1, :]
    p2[:, 1:] = dy[:, 1:] - dy[:, :-1]

    # Calculate rho(x, y)
    p = p1 + p2
    p[0, 0] = dx[0, 0] + dy[0, 0]  # Boundary condition
    p[0, 1:] = dx[0, 1:] + dy[0, 1:] - dy[0, :-1]  # Boundary condition
    p[1:, 0] = dx[1:, 0] - dx[:-1, 0] + dy[1:, 0]  # Boundary condition

    # Perform 2D DCT on p
    pp = dct(dct(p.T, norm='ortho').T, norm='ortho') + np.finfo(float).eps  # Add epsilon to avoid division by zero

    # Initialize unwrapped phase
    P_unwrap = np.zeros((M, N))

    # Compute unwrapped phase using inverse DCT
    for m in range(M):
        for n in range(N):
            P_unwrap[m, n] = pp[m, n] / (2 * np.cos(np.pi * (m) / M) + 2 * np.cos(np.pi * (n) / N) - 4 + np.finfo(float).eps)

    # Handle the first element separately as in the original code
    P_unwrap[0, 0] = pp[0, 0]

    # Perform inverse DCT to get the unwrapped phase
    P_unwrap = idct(idct(P_unwrap.T, norm='ortho').T, norm='ortho')

    # Return the unwrapped phase as the result
    least_squares_result = P_unwrap
    return least_squares_result


# 你需要根据你的实际需求初始化 WPHI 变量


# 记得根据需要定义 DZ1SX, DZ2SX 等等变量


