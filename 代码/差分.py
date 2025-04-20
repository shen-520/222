import os
import numpy as np
import matplotlib.pyplot as plt

folder_path = r"C:\Users\lenovo\Desktop\aaa\新建文件夹"
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

print("文件列表:")
for i, filename in enumerate(txt_files):
    print(f"{i+1}. {filename}")

i=25

while True:
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

    # 读取数据并过滤频率 >= 1420 MHz
    data1 = np.loadtxt(file1, skiprows=1)
    data1 = data1[data1[:, 0] >= 1420]
    data1 = data1[data1[:, 0] <= 1421]
    freq_intensity1 = {freq: intensity for freq, intensity in data1}

    data2 = np.loadtxt(file2, skiprows=1)
    data2 = data2[data2[:, 0] >= 1420]
    data2 = data2[data2[:, 0] <= 1421]
    freq_intensity2 = {freq: intensity for freq, intensity in data2}

    # 计算差分
    diff_intensity = {freq: freq_intensity2[freq] - freq_intensity1.get(freq, 0) for freq in freq_intensity2}
    freqs = list(diff_intensity.keys())
    intensities = list(diff_intensity.values())

    # 绘图
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, intensities, linestyle='-')
    plt.xlabel('Frequency')
    plt.ylabel('Difference in Intensity')
    plt.title('Difference in Intensity vs Frequency')
    plt.grid(True)

    # 使用计数器生成文件名
    save_path = fr"C:/Users/lenovo/Desktop/aaa/图片/L{i}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    
    i += 5  
    
    plt.show()
