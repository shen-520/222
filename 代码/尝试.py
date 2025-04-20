import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from sklearn.metrics import r2_score

# 定义文件夹路径
folder_path = r"C:\Users\lenovo\Desktop\aaa\新建文件夹"

# 获取文件夹中的所有txt文件
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 打印文件列表供用户选择
print("文件列表:")
for i, filename in enumerate(txt_files):
    print(f"{i+1}. {filename}")

# 获取用户选择的两个文件
while True:
    try:
        file_index1 = int(input("请选择第一个文件的序号: ")) - 1
        file_index2 = int(input("请选择第二个文件的序号: ")) - 1
        if 0 <= file_index1 < len(txt_files) and 0 <= file_index2 < len(txt_files):
            break
        else:
            print("请输入正确的序号")
    except ValueError:
        print("请输入数字")

file1 = os.path.join(folder_path, txt_files[file_index1])
file2 = os.path.join(folder_path, txt_files[file_index2])

# 读取第一个文件的数据
data1 = np.loadtxt(file1, skiprows=1)
data1 = data1[data1[:, 0] >= 1420]
data1 = data1[data1[:, 0] <= 1421]
freq_intensity1 = {freq: intensity for freq, intensity in data1}

# 读取第二个文件的数据
data2 = np.loadtxt(file2, skiprows=1)
data2 = data2[data2[:, 0] >= 1420]
data2 = data2[data2[:, 0] <= 1421]
freq_intensity2 = {freq: intensity for freq, intensity in data2}

# 计算差分后的数据
diff_intensity = {freq: freq_intensity2[freq] - freq_intensity1.get(freq, 0) for freq in freq_intensity2}

# 提取频率和差分后的强度用于绘图
freqs = np.array(list(diff_intensity.keys()))
intensities = np.array(list(diff_intensity.values()))

# 绘制差分后的数据图
plt.figure(figsize=(7, 4))
plt.plot(freqs, intensities, linestyle='-')
plt.xlabel('Frequency')
plt.ylabel('Difference in Intensity')
plt.title('Difference in Intensity vs Frequency')
plt.grid(True)
plt.show()

# 1. 对频谱图进行平滑处理
window_length = 15  # 滑动窗口大小（必须为奇数）
polyorder = 5       # 多项式阶数
smoothed_intensities = savgol_filter(intensities, window_length=window_length, polyorder=polyorder)

# 2. 在平滑后的数据上识别峰值
peaks, _ = find_peaks(smoothed_intensities, 
                     height=np.max(smoothed_intensities)*0.1,
                     distance=10,
                     prominence=np.std(smoothed_intensities))

# 3. 限制波峰数量（示例限制为3个）
max_peaks = 3
if len(peaks) > max_peaks:
    peak_heights = smoothed_intensities[peaks]
    peaks = peaks[np.argsort(peak_heights)[-max_peaks:]]  # 取最高的max_peaks个峰
print(f"最终使用 {len(peaks)} 个波峰进行拟合")

# 4. 设置初始参数（基于平滑后的数据）
initial_params = []
for peak_idx in peaks:
    a_guess = smoothed_intensities[peak_idx]  # 使用平滑后的幅值
    mu_guess = freqs[peak_idx]               # 频率值不变
    sigma_guess = 0.05                       # 初始宽度估计
    initial_params.extend([a_guess, mu_guess, sigma_guess])

# 5. 设置参数边界（必须与 initial_params 长度一致）
num_peaks = len(peaks)
lower_bounds = [0, freqs.min(), 0.01] * num_peaks
upper_bounds = [np.inf, freqs.max(), 0.5] * num_peaks
bounds = (lower_bounds, upper_bounds)

# 高斯函数定义
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def multi_gaussian(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a, mu, sigma = params[i:i+3]
        y += gaussian(x, a, mu, sigma)
    return y

# 6. 执行高斯拟合（对原始差分数据）
try:
    popt, pcov = curve_fit(multi_gaussian, freqs, intensities,
                          p0=initial_params, bounds=bounds, maxfev=5000)
    
    # 7. 计算拟合曲线
    fit_curve = multi_gaussian(freqs, *popt)
    
    # 8. 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, intensities, 'b-', label='Original Difference')
    plt.plot(freqs, smoothed_intensities, 'g--', label='Smoothed Data', alpha=0.7)
    plt.plot(freqs, fit_curve, 'r-', label='Gaussian Fit')
    plt.scatter(freqs[peaks], intensities[peaks], c='red', marker='x', s=100, label='Identified Peaks')
    
    # 计算并显示R²
    r2 = r2_score(intensities, fit_curve)
    plt.title(f'Gaussian Fit (R² = {r2:.4f})')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Intensity Difference')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 打印拟合参数
    print("\n拟合参数:")
    for i in range(0, len(popt), 3):
        print(f"Component {i//3 + 1}:")
        print(f"  Amplitude: {popt[i]:.4f}")
        print(f"  Center: {popt[i+1]:.4f} MHz")
        print(f"  Sigma: {popt[i+2]:.4f} MHz")
        print(f"  FWHM: {2.355*popt[i+2]:.4f} MHz\n")

except Exception as e:
    print(f"拟合失败: {str(e)}")
