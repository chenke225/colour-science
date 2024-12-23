
import numpy as np
import matplotlib.pyplot as plt
import colour
#from colour.models import RGB_COLOURSPACES
from colour.appearance import InductionFactors_CIECAM02, XYZ_to_CIECAM02, VIEWING_CONDITIONS_CIECAM02
from gamut_util import LCHab_to_XYZ, LCHuv_to_XYZ, XYZ_to_uv
from colour.plotting import plot_chromaticity_diagram_CIE1976UCS
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# 绘制 BT2020 色域在LAB和uv的三维和二维图。 matplotlib TkApp后端交互  ############################################# 已经通过python 3.12.3测试

matplotlib.use('TkAgg')
plt.ion()
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端


# 定义色空间和白点
cs_name = "ITU-R BT.2020"
colourspace = colour.RGB_COLOURSPACES[cs_name]
illuminant = colourspace.whitepoint


# 选择色彩空间（例如 DCI-P3）和白点
cs_name = "ITU-R BT.2020"
colourspace = colour.RGB_COLOURSPACES[cs_name]
illuminant = colourspace.whitepoint

# 定义采样数
samples = 20
n_values = np.linspace(0, 1, samples)

# 生成特定形式的 RGB 采样点
RGB_points = []
for n in n_values:
    RGB_points.extend([
       [n, 0, 1], [0, n, 1],  [0, 1, n]
    ])

#[n, 0, 0], [0, n, 0],  [0, 0, n],
#[n, 1, 0]   [1, n, 0]    [1, 0, n]
#[n, 1, 1] [1, n,1]   [1,1, n],
#[n, 0, 1], [0, n, 1],  [0, 1, n],

rgb = np.array([[1,0,0],[0,1,0], [0,0,1]])
# 转换 RGB 采样点为 XYZ 色空间
RGB_points = np.array(RGB_points)

XYZ_points = colour.RGB_to_XYZ(RGB_points, colourspace, illuminant)
XYZ_rgb = colour.RGB_to_XYZ(rgb, colourspace, illuminant)
# 转换 XYZ 为 CIELUV 中的 u*v* 色度坐标
Luv_points = colour.XYZ_to_Luv(XYZ_points, illuminant=illuminant)
Luv_rgb = colour.XYZ_to_Luv(XYZ_rgb, illuminant=illuminant)
# 提取 L*, u*, v* 坐标
L_values = Luv_points[..., 0]
u_values = Luv_points[..., 1]
v_values = Luv_points[..., 2]

# 绘制 L* 等高线在 u* v* 平面
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-200, 200)
ax.set_ylim(-150, 150)
# 将L*分成多个等高区间，按区间绘制点
interval = 5
L_levels = np.arange(0, 100, interval)

# 使用不同颜色绘制不同 L* 区间的等高线
for L in L_levels:
    mask = (L_values >= L) & (L_values < L + interval)
    ax.scatter(u_values[mask], v_values[mask], label=f"L*={L}-{L+interval}", s=10)

u_rgb = Luv_rgb[..., 1]
v_rgb = Luv_rgb[..., 2]
ax.scatter(u_rgb[0], v_rgb[0], c='r', marker='o', label="Point r")
ax.scatter(u_rgb[1], v_rgb[1], c='g', marker='o', label="Point g")
ax.scatter(u_rgb[2], v_rgb[2], c='b', marker='o', label="Point b")


ax.set_xlabel("u*")
ax.set_ylabel("v*")
ax.set_title(f"L* Contours in CIELUV u*v* Plane ({cs_name})")
ax.legend(loc="best")
plt.grid(True)
plt.show()

# 提取 u* 和 v* 坐标
u_values = Luv_points[..., 1]
v_values = Luv_points[..., 2]
u_rgb = Luv_rgb[..., 1]
v_rgb = Luv_rgb[..., 2]

# 绘制 u*v* 平面上的点
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(u_values, v_values, color='orange', marker='o', label="RGB Boundary Points")
ax.scatter(u_rgb[0], v_rgb[0], c='r', marker='o', label="RGB Boundary Points")
ax.scatter(u_rgb[1], v_rgb[1], c='g', marker='o', label="RGB Boundary Points")
ax.scatter(u_rgb[2], v_rgb[2], c='b', marker='o', label="RGB Boundary Points")
ax.set_xlabel("u*")
ax.set_ylabel("v*")
ax.set_title(f"{cs_name} RGB Boundary in CIELUV u*v* Space")
plt.legend()
plt.grid(True)
plt.show()



# ************************************************************

def generate_rgb_samples(points):
    # 每个通道的采样点数量和范围    RGB立方体6个平面采样20x20，总共2168个采样点。
    num_samples_per_side = points
    linspace_values = np.linspace(0, 1, num_samples_per_side)
    rgb_samples = set()
    
    # 遍历RGB立方体的每个面
    for fixed_channel in range(3):
        for fixed_value in [0, 1]:  # 固定当前通道为0或1
            for x in linspace_values:
                for y in linspace_values:
                    rgb = [0, 0, 0]
                    rgb[fixed_channel] = fixed_value
                    rgb[(fixed_channel + 1) % 3] = x
                    rgb[(fixed_channel + 2) % 3] = y
                    rgb_samples.add(tuple(rgb))
    
    # 将集合转为数组并返回
    return np.array(list(rgb_samples))

# 定义色空间
cs_name = "ITU-R BT.2020"    # ITU-R BT.2020,  ITU-R BT.709,  sRGB,  Adobe RGB (1998), DCI-P3
colourspace = colour.RGB_COLOURSPACES[cs_name]
illuminant= colourspace.whitepoint

# 生成 RGB 边界网格
RGB = np.array([[r, g, b] for r in [0, 1] for g in [0, 1] for b in [0, 1]])
# 创建网格，生成 DCI-P3 的 RGB 值
samples = 10
RGB = np.array(
    [[r, g, b] for r in np.linspace(0, 1, samples)
               for g in np.linspace(0, 1, samples)
               for b in np.linspace(0, 1, samples)]
)
linear_RGB = RGB
# 转换到 XYZ 色空间
XYZ = colour.RGB_to_XYZ(linear_RGB, colourspace, illuminant)
# 绘制 CIELAB 色域
Luv = colour.XYZ_to_Luv(XYZ, illuminant=illuminant)
Lab = colour.XYZ_to_Lab(XYZ, illuminant=illuminant)
#sc = ax.scatter(Lab[:, 1], Lab[:, 2],  Lab[:,0],  cmap='viridis', s=20, label="All points")


# 创建 3D 图表显示 LAB 颜色点的分布
fig = plt.figure(figsize=(14, 6))

# 3D 图：绘制 DCI-P3 颜色点在 CIELAB 空间的分布
ax_3d = fig.add_subplot(121, projection='3d')
ax_3d.scatter(Luv[:, 1], Luv[:, 2], Luv[:, 0], c=Luv[:, 0], cmap='viridis', s=20, alpha=0.5) # type: ignore
ax_3d.set_title("DCI-P3 Gamut in CIELAB Space (3D)")
ax_3d.set_xlabel("a*")
ax_3d.set_ylabel("b*")
ax_3d.set_zlabel("L*") # type: ignore
ax_3d.grid(True)

# 2D 图：将点投影到 a*b* 平面
ax_2d = fig.add_subplot(122)
ax_2d.scatter(Luv[:, 1], Luv[:, 2], c=Luv[:, 0], cmap='viridis', s=20, alpha=0.5)
ax_2d.set_title("DCI-P3 Gamut Projection on a*b* Plane normalize white, no gamma")
ax_2d.set_xlabel("a*")
ax_2d.set_ylabel("b*")
ax_2d.set_xlim(-200, 300)
ax_2d.set_ylim(-150, 150)
ax_2d.grid(True)

from scipy.spatial import ConvexHull
ab_points = np.column_stack((Lab[:, 1], Lab[:, 2])) 
# 使用ConvexHull函数
hull = ConvexHull(ab_points)
# 获取凸包顶点
hull_vertices = hull.vertices
#提取凸包顶点坐标  
hull_points_LAB = ab_points[hull_vertices]

# 应用 gamma 校正
def gamma_correct(rgb):
    corrected = np.zeros_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(3):  # 对 R, G, B 三个通道进行处理
            if rgb[i, j] <= 0.04045:
                corrected[i, j] = rgb[i, j] / 12.92
            else:
                corrected[i, j] = ((rgb[i, j] + 0.055) / 1.055) ** 2.4
                # 确保值在 0 到 1 的范围内
            corrected[i, j] = np.clip(corrected[i, j], 0, 1)
    return corrected

# 对 RGB 应用 gamma 校正
linear_RGB = gamma_correct(RGB)
linear_RGB = RGB   # 取消gamma

# 将线性 RGB 转换为 XYZ
illuminant = np.array([0.34570, 0.35850])  # 参考白点
XYZ_gamma = colour.RGB_to_XYZ(linear_RGB, colourspace, illuminant)
Lab = colour.XYZ_to_Lab(XYZ_gamma, illuminant=illuminant)

# 定义白点和观察条件（用于 CAM02-UCS）
XYZ_w = np.array([95.047, 100.000, 108.883])  # D65 白点
L_A = 318.31  # 亮适应亮度
Y_b = 20.0    # 背景亮度20.0

surround = VIEWING_CONDITIONS_CIECAM02["Dim"]   #"Average": InductionFactors_CIECAM02(1, 0.69, 1)
                                                                                                                         #"Dim": InductionFactors_CIECAM02(0.9, 0.59, 0.9),
                                                                                                                        #"Dark:: InductionFactors_CIECAM02(0.8, 0.525, 0.8)
#  dim for  c = 0.59, N c = 0.9, F = 0.9, L A = 16


# 转换 XYZ 到 CIECAM02
Jab = XYZ_to_CIECAM02(XYZ_gamma, XYZ_w, L_A, Y_b, surround)
J = np.array(Jab.J)
C = np.array(Jab.C)
h = np.array(Jab.h)
Jab_values = np.vstack((J, C, h)).T


# 绘制 DCI-P3 色域在 CIELAB 和 CAM02-UCS 空间中的投影
# 创建 3D 图表显示 LAB 颜色点的分布
fig = plt.figure(figsize=(14, 6))

# 3D 图：经gamma校正的，绘制 DCI-P3 颜色点在 CIELAB 空间的分布
ax_3d = fig.add_subplot(121, projection='3d')
ax_3d.scatter(Lab[:, 1], Lab[:, 2], Lab[:, 0], c=Lab[:, 0], cmap='viridis', s=20, alpha=0.5) # type: ignore
ax_3d.set_title("DCI-P3 Gamut in CIELAB Space (3D)")
ax_3d.set_xlabel("a*")
ax_3d.set_ylabel("b*")
ax_3d.set_zlabel("L*") # type: ignore
ax_3d.grid(True)

# 2D 图：将点投影到 a*b* 平面（经gamma校正）
ax_2d = fig.add_subplot(122)
ax_2d.scatter(Lab[:, 1], Lab[:, 2], c=Lab[:, 0], cmap='viridis', s=20, alpha=0.5)
ax_2d.set_title("DCI-P3 Gamut Projection on a*b* Plane, normalize white, gamma")
ax_2d.set_xlabel("a*")
ax_2d.set_ylabel("b*")
ax_2d.set_xlim(-150, 150)
ax_2d.set_ylim(-150, 150)
ax_2d.grid(True)

# 绘制 CAM02-UCS 色域，未经gamma校正。 将 Jab 转换为 La*b* 值
Lab_values = np.zeros((Jab_values.shape[0], 3))
Lab_values[:, 0] = Jab_values[:, 0]  # J -> L
Lab_values[:, 1] = Jab_values[:, 1] * np.cos(np.radians(Jab_values[:, 2]))  # C -> a
Lab_values[:, 2] = Jab_values[:, 1] * np.sin(np.radians(Jab_values[:, 2]))  # C -> b，“”

# 绘制 La*b* 色域
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(Lab_values[:, 1], Lab_values[:, 2], color='red', alpha=0.3, s=0.5)
ab_points = np.column_stack((Lab_values[:, 1], Lab_values[:, 2])) 
from scipy.spatial import ConvexHull
# 使用ConvexHull函数
hull = ConvexHull(ab_points)
# 获取凸包顶点
hull_vertices = hull.vertices
#提取凸包顶点坐标  
hull_points = ab_points[hull_vertices]
plt.plot(hull_points[:, 0], hull_points[:, 1], color = 'red')
plt.plot(hull_points_LAB[:, 0], hull_points_LAB[:, 1], color = 'blue')

ax.set_title("DCI-P3 CIECAM02 Gamut in La*b*")
ax.set_xlabel("a*")
ax.set_ylabel("b*")
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.grid(True)

plt.show()
aa = 1