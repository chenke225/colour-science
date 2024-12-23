# 绘制12640 三维色域凸包 图像， 第二图是另一个立体图示例，与12640无关
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, ConvexHull, distance
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import openpyxl


#print("Available Backends:", matplotlib.rcsetup.all_backends)  # type: ignore # 所有后端列表
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端

matplotlib.use('TkAgg')
plt.ion()
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端

# 设置光源、加载数据
ILLUM_S = 'D50'  # 光源
file_path = 'LCH_gamut.xlsx'
sheet_name = '12640_LCHab'
df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

# 提取 hue, L, Chroma 值
hue = df.index.to_numpy()  # 色相角 (36个)
L_values = df.columns.to_numpy(dtype=float)  # 亮度值 (19个)
Chroma = df.to_numpy()  # 色度矩阵 (36x19)
sy1 = 1


# 创建 LCH 数据点列表
LCH_list = []
for i, L in enumerate(L_values):
    for j, h in enumerate(hue):
        C = Chroma[j, i]
        LCH_list.append([L, C, h])

LCH_array = np.array(LCH_list)  # 转换为 NumPy 数组

# LCHab 转换为 Lab
L = LCH_array[:, 0]
C = LCH_array[:, 1]
h = np.deg2rad(LCH_array[:, 2])  # 将 hue 角度转换为弧度
a = C * np.cos(h)
b = C * np.sin(h)

Lab_array = np.column_stack((L, a, b))  # 生成 Lab 数组

# 计算凸包
hull = ConvexHull(Lab_array)

# 绘制三维图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Lab_array[:, 1], Lab_array[:, 2], Lab_array[:, 0], color='blue', s=1, alpha=0.6) #type: ignore

# 绘制凸包表面
for simplex in hull.simplices:
    ax.plot_trisurf(Lab_array[simplex, 1], Lab_array[simplex, 2], Lab_array[simplex, 0], color='cyan', edgecolor='grey', alpha=0.3) # type: ignore

# 设置坐标轴标签
ax.set_xlabel('a*')
ax.set_ylabel('b*')
ax.set_zlabel('L*') # type: ignore
ax.set_title('3D Color Gamut in Lab Space (Convex Hull Surface)')

plt.show()
aa = 1
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端

# ************************************************************************************
# 3D  球形r， alpha, theta 坐标

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm

# 设置中心的 L* 值
L_center = 50  

# 16x16 扇区划分
alpha_sections = 16  # 水平 (azimuthal) 方向分区
theta_sections = 16  # 垂直 (polar) 方向分区

# 定义 alpha 和 theta 的角度范围
alphas = np.linspace(0, 2 * np.pi, alpha_sections + 1)  # 0 到 360 度，包含边界
thetas = np.linspace(0, np.pi, theta_sections + 1)  # 0 到 180 度，包含边界

# 初始化图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 最大半径 r_max = 50
r_max = 50

# 遍历每个扇区
for i in range(alpha_sections):
    for j in range(theta_sections):
        # 当前扇区的四个角落角度 (alpha, theta)
        alpha1, alpha2 = alphas[i], alphas[i + 1]
        theta1, theta2 = thetas[j], thetas[j + 1]
        
        # 定义每个扇区的四个角点，并生成 r 值，r 在不同 theta 下变化模拟球形效果
        points = []
        for alpha, theta in [(alpha1, theta1), (alpha2, theta1), (alpha2, theta2), (alpha1, theta2)]:
            r = r_max * np.sin(theta)  # 限制 r 的最大值为 r_max
            x = r * np.sin(theta) * np.cos(alpha)
            y = r * np.sin(theta) * np.sin(alpha)
            z = L_center + r * np.cos(theta)
            points.append([x, y, z])
        
        # 根据扇区的位置设置颜色
        color = cm.viridis((i * theta_sections + j) / (alpha_sections * theta_sections))  # type:ignore
        
        # 在当前扇区中添加多边形
        poly = Poly3DCollection([points], color=color, alpha=0.7)
        ax.add_collection3d(poly)  # type: ignore

# 设置坐标轴范围以显示球体效果
ax.set_xlim(-r_max, r_max)
ax.set_ylim(-r_max, r_max)
ax.set_zlim(L_center - r_max, L_center + r_max)   # type: ignore

# 设置坐标轴标签和标题
ax.set_xlabel("a* ")
ax.set_ylabel("b* ")
ax.set_zlabel("L* ") # type: ignore
ax.set_title("CIELAB colour space 16x16 gamut boundary")

# 调整视角以显示球体效果
ax.view_init(elev=30, azim=45) # type: ignore

plt.show()

aa = 1