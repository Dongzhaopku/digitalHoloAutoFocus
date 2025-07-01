"""
High Pass Filtering for the uniform criteria of autofocus
Written by Dong, Zhao
Oct 21, 2024 A.D.
-------------------------------------
Information Optics Lab
Hebei University of Engineering, Handan, Hebei Province 056038, P. R. China
handandong@163.com
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import  matplotlib
import time
from scipy.optimize import minimize, minimize_scalar, dual_annealing
import subRoutine4Holo as srh

# 1. 读取图片并灰度化，赋予振幅（0-1）和相位（0-4pi）,同时图片边缘部分补齐0
def get_complex_amplitude(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = img / 255.0  # 将灰度值归一化到0~1
    img_normalized = img
    h, w = img.shape
    # 取图片的中心部分，裁剪为正方形
    min_dim = min(h, w)
    cropped_img = img[(h - min_dim) // 2:(h + min_dim) // 2, (w - min_dim) // 2:(w + min_dim) // 2]

    # 调整为400x400大小
    resized_img = cv2.resize(img, (512, 512))

    # 扩展为512x512，并在周围填充灰度为0的元素
    padded_img = resized_img

    amplitude = padded_img  # 振幅为归一化后的灰度值
    phase = amplitude * np.pi
    # complex_amplitude = amplitude*np.exp(1j*phase)  # Complex
    complex_amplitude = np.exp(1j * phase)  # Complex
    # complex_amplitude = amplitude  # Complex
    return complex_amplitude, img_normalized


# 2. 角谱衍射计算，lambda=532nm
def angular_spectrum_diffraction(input_field, z, wavelength, pixel_size):
    M, N = input_field.shape
    k = 2 * np.pi / wavelength  # 波数 k
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(M, d=pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)))
    U1 = np.fft.fftshift(np.fft.fft2(input_field))
    U2 = U1 * H
    output_field = np.fft.ifft2(np.fft.ifftshift(U2))
    return output_field

def anglular_spectrumm_diffraction_filter(input_field, z, wavelength, pixel_size):
    M, N = input_field.shape
    k = 2 * np.pi / wavelength  # 波数 k
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(M, d=pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z * np.sqrt(1 - (wavelength ** 2) * (FX ** 2 + FY ** 2)))
    sigma = 0.000007
    Mask = 1 - np.exp(-(wavelength**2)*(FX ** 2 + FY ** 2) / 2 / sigma)
    # Mask = 0
    U1 = np.fft.fftshift(np.fft.fft2(input_field))
    U2 = U1 * H * Mask
    output_field_filtered = np.fft.ifft2(np.fft.ifftshift(U2))
    return output_field_filtered

# 3. 角谱滤波，输出滤波后的Fourier谱，然后根据z值输出相应的滤波后的光场
def spectra_filtered(input_field,pixel_size,wavelength):
    """
    :param input_field:
    :param pixel_size:
    :param sigma: 滤波系数
    :param c: 压缩系数
    :return:
    """
    M, N = input_field.shape

    fx = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_size))
    fy = np.fft.fftshift(np.fft.fftfreq(M, d=pixel_size))
    FX, FY = np.meshgrid(fx, fy)
    sigma = 0.000007
    Mask = 1 - np.exp(-(wavelength**2)*(FX ** 2 + FY ** 2) / 2 / (sigma))
    # Mask = 1
    U1 = np.fft.fftshift(np.fft.fft2(input_field))
    U2 = U1 * Mask
    return U2, FX, FY

# 4. 聚焦判据
def MD(Uc, region=None):
    if region:
        x_min, y_min, x_max,  y_max = region
        Uc = Uc[y_min:y_max, x_min:x_max]
    # 计算复数模的平方并求和
    Md = np.sum(np.abs(Uc))
    return Md


# 5.反向 Fresnel 衍射的函数，计算在某个 z 值下的滤波后的聚焦判据值
def Md_reverse_fresnel_diffraction(z, input_spectra,FX, FY, wavelength, region=None):
    # 计算衍射场 U(z)
    H = np.exp(1j * 2* np.pi * z * np.sqrt(1 - (wavelength ** 2) * (FX ** 2 + FY ** 2))/wavelength)
    diffraction_field = np.fft.ifft2(np.fft.ifftshift(input_spectra * H))
    # 使用聚焦判据函数 MD 来计算聚焦质量
    Md_value = MD(diffraction_field, region)
    return Md_value


# 6. 拟牛顿法优化 z
def optimize_focus(input_spectra,FX, FY, wavelength, region=None):
    initial_z = -300e-3
    bounds = [(-400e-3, -200e-3)]  # z 的搜索范围是 -200mm 到 0mm
    result = minimize(Md_reverse_fresnel_diffraction, initial_z, args=(input_spectra, FX, FY, wavelength, region),
                      method='L-BFGS-B', bounds=bounds)
    return result.x, result.fun


def find_optimal_z(input_spectra,FX, FY, wavelength,   region=None):
    # Minimize the function with respect to z, passing the extra arguments via 'args'
    result = minimize_scalar(Md_reverse_fresnel_diffraction,
                             bounds=(-400e-3, -200e-3),
                             args=(input_spectra, FX, FY, wavelength, region),
                             method='bounded',
                             # tol = 0.1,
                             # options = {
                             #     'xatol': 0.001}
                             )
    return result.x, result.fun


# 7. z 值扫描
def scan_focus(input_spectra,FX, FY,wavelength, z_min, z_max, z_step,region=None):
    z_values = np.arange(z_min, z_max + z_step, z_step)  # z 的取值范围
    Md_values = []
    for z in z_values:
        Md_value = Md_reverse_fresnel_diffraction(z, input_spectra,FX, FY, wavelength, region)
        Md_values.append(Md_value)
    return z_values, Md_values


#8 压缩图像
def compress_complex_amplitude(complex_amplitude, a, pixel_size, region=None):
    """
    对复振幅进行抽样压缩，同时调整像素尺寸。

    参数:
    complex_amplitude -- 输入的复振幅矩阵（二维复数矩阵）
    a -- 压缩系数，决定压缩的比例（0 < a <= 1，a越小，压缩程度越高）
    pixel_size -- 原始图像的像素大小


    返回:
    compressed_amplitude -- 压缩后的复振幅矩阵
    pixel -- 压缩后的像素大小
    """

    # 分离实部和虚部
    real = np.real(complex_amplitude)
    imag = np.imag(complex_amplitude)
    # phase = least_squares(phase)

    # 原始图像尺寸
    original_height, original_width = complex_amplitude.shape

    # 计算压缩后的尺寸
    new_width = int(original_width * a)
    new_height = int(original_height * a)

    #实部和虚部按压缩系数进行降采样
    real_resized = cv2.resize(real, (new_width, new_height), interpolation=cv2.INTER_AREA)
    imag_resized = cv2.resize(imag, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 压缩后的像素大小
    pixel_size1 = pixel_size / a

    # 重建压缩后的复振幅矩阵
    compressed_amplitude = real_resized + 1j * imag_resized
    if region is not None and a != 1:
        # 调整区域的左上角和右下角坐标
        # 计算新的区域位置
        new_region = [
            int(region[0] * a),  # 左上角 X 坐标
            int(region[1] * a),  # 左上角 Y 坐标
            int(region[2] * a),  # 右下角 X 坐标
            int(region[2] * a)  # 右下角 Y 坐标
        ]
    else:
        # 如果没有区域或者压缩系数为 1，直接使用原区域
        new_region = region

    return compressed_amplitude, pixel_size1, new_region

# 主程序
if __name__ == "__main__":
    wavelength = 532e-9  # 532nm
    pixel_size = 4.8e-6  # 假设像素大小为4.8微米
    image_path = 'F:\BaiduSyncdisk\FocusDH\simSample\Cameraman.jpg'
    matplotlib.use('Qt5Agg')

    # Step 1: 获取复振幅
    complex_amplitude, gray_image = get_complex_amplitude(image_path)

    # Step 2:生成衍射复振幅
    Uc = angular_spectrum_diffraction(complex_amplitude, 350.20e-3, wavelength, pixel_size)  # 示例z=100mm

    A0 = np.fft.fftshift(np.fft.fft2(Uc))
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(Uc) ** 2, cmap='gray', vmin=0, vmax=np.max(np.abs(Uc) ** 2))
    plt.title("Diffraction Pattern Intensity")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(np.log(1 + np.abs(A0)))
    plt.title("Frequency Pattern Intensity")
    plt.colorbar()
    plt.show()


    U0 = anglular_spectrumm_diffraction_filter(Uc, -347E-3, wavelength, pixel_size)
    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(U0) ** 2, cmap='gray')
    plt.colorbar()  # 可选：添加颜色条来显示数值范围
    plt.title("Diffraction Pattern Intensity")
    plt.show()

    compression_rates = [1.0, 0.5, 0.2]
    z_min = -400e-3  # 最小 z 值 -400mm
    z_max = -200e-3  # 最大 z 值 -200mm
    z_step = 1e-3  # 每次步长 1mm

        # 假设有不同压缩率的结果存储在列表中
    all_z_values = []  # 用于存储所有的 z 值
    all_Md_values = []  # 用于存储所有的 Md 值
    all_Md_normalized = []  # 用于存储归一化后的 Md 值

    # 假设 min_Md 和 optimal_z 是Brent方法的结果
    all_min_Md_normalized = []  # 用于存储每个压缩率下Brent的最小Md归一化结果
    all_optimal_z = []  # 用于存储每个压缩率下Brent的最优z值

    for rate in compression_rates:
        print(f"开始处理压缩率：{rate}")
        # 压缩数据
        start_time = time.time()

        Uc_compressed, pixel_size_compressed,_ = compress_complex_amplitude(Uc, rate, pixel_size)
        if rate == 1:
            start_time = time.time()

        input_spectra, FX, FY= spectra_filtered(Uc_compressed, pixel_size_compressed,wavelength)
        # 使用 scan_focus 计算 Md-z 数据
        z_values, Md_values = scan_focus(input_spectra, FX, FY, wavelength,  z_min, z_max, z_step)
        end_time = time.time()

        # 计算运行时间
        execution_time = end_time - start_time
        print(f"压缩率 {rate} 的运行时间: {execution_time:.3f} 秒")

                # 计算Md的最大值和最小值
        Md_max = max(Md_values)
        Md_min = min(Md_values)

        # 对Md进行归一化
        Md_normalized = [(Md - Md_min) / (Md_max - Md_min) for Md in Md_values]

            # 存储每个压缩率的结果
        all_z_values.append(z_values)
        all_Md_values.append(Md_values)
        all_Md_normalized.append(Md_normalized)
        # 保存归一化后的Md-z数据到文件
        output_filename = f"SimPHA_Md_z_normalized_compression_rate_{rate}.txt"
        with open(output_filename, 'w') as f:
            for z, Md_norm in zip(z_values, Md_normalized):
                f.write(f"{-1000*z:.6f} {Md_norm:.6f}\n")
        print(f"归一化后的Md-z数据已保存到 {output_filename}")

                # 使用 Brent 方法进行优化计算
        start_time_brent = time.time()
        # Uc_compressed, pixel_size_compressed = compress_complex_amplitude(Uc, rate, pixel_size)
        optimal_z, min_Md = find_optimal_z(input_spectra,FX, FY, wavelength)
        end_time_brent = time.time()
            # 计算Brent方法运行时间
        brent_execution_time = end_time_brent - start_time_brent
        min_Md_normalized = (min_Md - Md_min) / (Md_max - Md_min)
        print(f"Brent方法最小归一化Md值: {min_Md_normalized:.6f}")


        # 存储 Brent 方法的最优值和归一化的 Md
        all_min_Md_normalized.append(min_Md_normalized)
        all_optimal_z.append(optimal_z)

        print(f"Brent方法运行时间（压缩率 {rate}）: {brent_execution_time:.3f} 秒")
        print(f"Brent方法得到的离焦距离{rate}：聚焦位置{-1000*optimal_z:.2f}mm")

    # 绘制每个压缩率的 Md-z 曲线
    for i, rate in enumerate(compression_rates):
        plt.plot(all_z_values[i], all_Md_normalized[i], label=f"压缩率 {rate}", linewidth=2)

        # 在图中标记 Brent 方法的最优点
        plt.scatter(all_optimal_z[i], all_min_Md_normalized[i], label=f"Brent最优点（压缩率 {rate})", s=100, marker="x")

    # 设置图形标题和标签
    plt.title("不同压缩率下的 Md-z 曲线")
    plt.xlabel("z (m)")
    plt.ylabel("归一化的 Md 值")

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

