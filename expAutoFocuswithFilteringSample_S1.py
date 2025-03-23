"""
The codes for the sample used in the paper
Jan 17 2025 A.D.
Dong, Zhao
Hebei University of Engineering
"""
import cv2
import numpy as np
import matplotlib

# from expAutoFocuswithFilteringPanelPhase import region_compressed

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import subRoutine4Holo as srh
import simAutoFocusWithHighPassFiltering4 as aff
import time
import compensatePhase

# image_path = 'F:/studying/Data/20171031/2017-11-04_11_53_33_328.bmp'
# image_path = 'F:/studying/Data/20171031/2017-11-04_11_58_12_808.bmp' #Phase
# image_path = 'F:/studying/Data/20171031/2017-11-04_12_04_00_312.bmp'# Intensity
# image_path = 'F:/studying/Data/20171031/2017-11-05_16_52_20_466.bmp'# Phase+Intensity
image_path = 'F:/studying/Data/20171031/2017-11-05_17_22_11_218.bmp'# Phase
# image_path = 'F:/studying/Data/20171031/2017-11-02_16_34_42_321.bmp' #Phase
# image_path = 'F:/studying/Data/20171031/2017-11-04_11_43_04_360.bmp' #Phase
# image_path = 'F:/studying/Data/20171031/2017-11-04_11_48_15_272.bmp' #Intenstiy
# image_path = 'F:/studying/Data/20171031/2017-12-23_10_27_26_118.bmp'


IU = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
IU = IU.astype(np.float64)  # 转换为浮点型
plt.figure()
plt.imshow(IU, cmap='gray')
plt.title('IU')
plt.axis('off')
plt.show(block=True)


start_X = 200
# start_Y = 1
# End_X = 423
# End_Y = 538
# IU = IU[start_Y:End_Y, start_X:End_X]  # 切片处理
IU = IU[:, start_X:start_X+1024]

plt.figure()
plt.imshow(IU, cmap='gray')
plt.title('IU')
plt.axis('off')
plt.show(block=True)

# recover_Img = srh.auto_recover_holo(IU)
#手动选取滤波区域
FI = np.fft.fftshift(np.fft.fft2(IU))
plt.figure()
plt.imshow(np.log(1 + np.abs(FI)), cmap='gray')
plt.title('Fourier Transform')
plt.axis('off')
plt.show()

W, H = np.shape(FI)
# Create a blank matrix to store the filtered Fourier image
FISlt = np.zeros((W, H), dtype=complex)

# Frequency coordinates and window size (adjust based on your needs)
fxc, fyc = 160, 214
dfx = abs(fxc - 63)
dfy = abs(fyc - 56)

Wm = W // 2 + 1
Hm = H // 2 + 1

# Apply a filter in the frequency domain (select the region around the center)
FISlt[Hm-dfy:Hm+dfy, Wm-dfx:Wm+dfx] = FI[fyc-dfy:fyc+dfy, fxc-dfx:fxc+dfx]

# Optional: Visualize the filtered Fourier image
plt.figure()
plt.imshow(np.log(1 + np.abs(FISlt)), cmap='gray')
plt.colorbar()
plt.title("Filtered Fourier Transform")
plt.show()

recover_Img = np.fft.ifft2(np.fft.ifftshift(FISlt))
region_xs=1
region_ys=3
region_xe=1
region_ye=3

plt.figure()
plt.imshow(np.abs(recover_Img), cmap='gray')
plt.title('IU')
plt.axis('off')
plt.show(block=True)

angleUc = compensatePhase.auto_compensation(np.angle(recover_Img))
Uc = abs(recover_Img)*np.exp(1j*(angleUc))

# #### 下一步进行自聚焦处理, 扫描型
wavelength = 532e-9 # 532nm
pixel_size = 4.8e-6  # 假设像素大小为4.8微米
# sigma = 0
# start_time2 = time.time()
# Uc_compressed, pixel_size_compressed = aff.compress_complex_amplitude(recover_Img, 0.1, pixel_size)
# input_spectra, FX, FY = aff.spectra_filtered(Uc_compressed, pixel_size_compressed, wavelength)
#
# z_min = -400E-3  # 最小 z 值 -400mm
# z_max = -200e-3  # 最大 z 值 -240mm
# z_step = 1e-3  # 每次步长 1mm
# z_values, Md_values = aff.scan_focus(input_spectra, FX, FY, wavelength,  z_min, z_max, z_step)
# end_time2 = time.time()
# execution_time = end_time2 - start_time2
# print(f"扫描法运行时间: {execution_time} 秒")
# Md_values_normalized = (Md_values - np.min(Md_values)) / (np.max(Md_values) - np.min(Md_values))
# # 找到 Md_values 中最小值的索引
# min_index = np.argmin(Md_values)
#
# # 通过索引得到对应的 z 值
# min_Md_z = z_values[min_index]
# print(f"聚焦位置{min_Md_z}mm")
# print(f"最小的聚焦判据值: {np.min(Md_values)}")
# # 可视化 归一化后的 Md 随 z 变化的曲线
# plt.figure(figsize=(8, 6))
# plt.plot(z_values * 1e3, Md_values_normalized, '-o')
# plt.xlabel('z (mm)')
# plt.ylabel('归一化后的 Md (聚焦判据)')
# plt.title('归一化后的聚焦判据 Md 随 z 的变化')
# plt.grid(True)
# plt.show(block=True)
#
# start_time = time.time()
# optimal_z, min_Md = aff.find_optimal_z(input_spectra,FX, FY, wavelength)
# end_time = time.time()
# print(f"最优 z 值: {optimal_z} m")
# print(f"最小的聚焦判据值: {min_Md}")
# execution_time = end_time - start_time
# print(f"Brents代码运行时间: {execution_time} 秒")
#
# Uc = aff.angular_spectrum_diffraction(recover_Img, optimal_z, wavelength, pixel_size)
#
# plt.figure(figsize=(6, 6))
# plt.imshow(np.abs(Uc) ** 2, cmap='gray', vmin=0, vmax=np.max(np.abs(Uc) ** 2))
# plt.title("Diffraction Pattern Intensity")
# plt.colorbar()
# plt.show(block=True)
#
# argUc = srh.auto_compensation(np.angle(Uc))
# argUc2 = srh.least_squares(argUc)
# plt.figure(figsize=(6, 6))
# plt.imshow(argUc2, cmap='gray')
# plt.title("Phase")
# plt.colorbar()
# plt.show(block=True)
#
#
# a = aff.MD(Uc)
# print(a)
compression_rates = [1.0, 0.5,1/3,1/4, 0.2,0.125, 0.1,1/16]
z_min = -400e-3  # 最小 z 值 -400mm
z_max = -200e-3  # 最大 z 值 -200mm
z_step = 1e-3  # 每次步长 1mm

# 存储所有结果
all_z_values = []  # 用于存储所有的 z 值
all_Md_values = []  # 用于存储所有的 Md 值
all_Md_normalized = []  # 用于存储归一化后的 Md 值
all_min_Md_normalized = []  # 用于存储每个压缩率下Brent的最小Md归一化结果
all_optimal_z = []  # 用于存储每个压缩率下Brent的最优z值

for rate in compression_rates:
    print(f"开始处理压缩率：{rate}")

    # 压缩数据
    start_time = time.time()
    if rate==1:
        Uc_compressed = Uc
        pixel_size_compressed = pixel_size
        start_time = time.time()
    else:
        Uc_compressed, pixel_size_compressed,region = aff.compress_complex_amplitude(Uc, rate, pixel_size)

    # 频率域滤波
    input_spectra, FX, FY = aff.spectra_filtered(Uc_compressed, pixel_size_compressed, wavelength)

    # 使用 scan_focus 计算 Md-z 数据
    z_values, Md_values = aff.scan_focus(input_spectra, FX, FY, wavelength, z_min, z_max, z_step)
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"压缩率 {rate} 的运行时间: {execution_time:.3f} 秒")

    # 计算 Md 的最大值和最小值
    Md_max = max(Md_values)
    Md_min = min(Md_values)

    # 对 Md 进行归一化
    Md_normalized = [(Md - Md_min) / (Md_max - Md_min) for Md in Md_values]

    # 存储每个压缩率的结果
    all_z_values.append(z_values)
    all_Md_values.append(Md_values)
    all_Md_normalized.append(Md_normalized)

    # 保存归一化后的 Md-z 数据到文件
    output_filename = f"Md_z_normalized_compression_rate_{rate}.txt"
    with open(output_filename, 'w') as f:
        for z, Md_norm in zip(z_values, Md_normalized):
            f.write(f"{-1000 * z:.6f} {Md_norm:.6f}\n")
    print(f"归一化后的 Md-z 数据已保存到 {output_filename}")

    # 使用 Brent 方法进行优化计算
    start_time_brent = time.time()
    if rate==1:
        Uc_compressed = Uc
        pixel_size_compressed = pixel_size
        start_time_brent = time.time()
    else:
        Uc_compressed, pixel_size_compressed,region = aff.compress_complex_amplitude(Uc, rate, pixel_size)
    optimal_z, min_Md = aff.find_optimal_z(input_spectra, FX, FY, wavelength)
    end_time_brent = time.time()

    # 计算 Brent 方法运行时间
    brent_execution_time = end_time_brent - start_time_brent
    min_Md_normalized = (min_Md - Md_min) / (Md_max - Md_min)
    print(f"Brent方法最小归一化Md值: {min_Md_normalized:.6f}")

    # 存储 Brent 方法的最优值和归一化的 Md
    all_min_Md_normalized.append(min_Md_normalized)
    all_optimal_z.append(optimal_z)

    print(f"Brent方法运行时间（压缩率 {rate}）: {brent_execution_time:.3f} 秒")
    print(f"Brent方法得到的离焦距离 {rate}：聚焦位置 {-1000 * optimal_z:.2f} mm")

# 绘制每个压缩率的 Md-z 曲线
plt.figure(figsize=(10, 6))
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

# 生成聚焦的振幅和相位图
optimal_Uc = aff.angular_spectrum_diffraction(Uc, optimal_z, wavelength, pixel_size)

# 显示振幅图
plt.figure(figsize=(6, 6))
plt.imshow(np.abs(optimal_Uc) ** 2, cmap='gray', vmin=0, vmax=np.max(np.abs(optimal_Uc) ** 2))
plt.title("聚焦后的振幅图")
plt.colorbar()
plt.axis('off')  # 关闭坐标轴
plt.show(block=True)

# 计算和显示相位图
argUc = srh.auto_compensation(np.angle(optimal_Uc))
argUc2 = srh.least_squares(argUc)
argUc2 = argUc2-np.min(argUc2)
plt.figure(figsize=(6, 6))
plt.imshow(argUc2, cmap='gray',vmin=0,vmax=2*np.pi)
plt.title("Phase")
plt.colorbar()
plt.axis('off')  # 关闭坐标轴
plt.show(block=True)
