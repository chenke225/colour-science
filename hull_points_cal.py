#  色域绘制，改进的凸包算法onvex_hull_cal ， 保证每个hue在凸包上有uv点，无hue角度标注。
#  python3.12.3测试通过 2024-12-19
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colour.plotting import plot_chromaticity_diagram_CIE1976UCS
import colour

from gamut_util import ConvexHull_general 
import matplotlib

matplotlib.use('TkAgg')
plt.ion()
print("Currently Using Backend:", matplotlib.get_backend()) 

ILLUM_C = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]['C']
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
ILLUM_S = ILLUM_C  #   确定标准光源  ******************************************************
file_path = 'LCH_gamut.xlsx'  # Replace with the path to your Excel file   *****************************
sheet_name = 'pointer_LCHab'                                   # ************************************************************************
Data_format = 'LCHab'
def LCHab_to_uv(LCHab, ILLUMINATNAT):
    Lab = colour.LCHab_to_Lab(LCHab)
    XYZ = colour.Lab_to_XYZ(Lab, ILLUMINATNAT)  # ILLUM_C
    xy_3d = colour.XYZ_to_xyY(XYZ)  # 第一种转换
    xy_2d = xy_3d[:2]
    return colour.xy_to_Luv_uv(xy_2d)

def LCHuv_to_uv(LCHuv, ILLUMINATNAT):
    Luv = colour.LCHuv_to_Luv(LCHuv)
    XYZ = colour.Luv_to_XYZ(Luv, ILLUMINATNAT)  # ILLUM_C
    xy_3d = colour.XYZ_to_xyY(XYZ)  # 第一种转换
    xy_2d = xy_3d[:2]
    return colour.xy_to_Luv_uv(xy_2d)

#from matplotlib import colors as mcolors
#import matplotlib.cm as cm
import colorspacious as cs
# 定义 CIELAB 色彩的映射函数
def hue_to_lab_color(hue_angle):
    # CIELAB 颜色空间的 a 和 b 轴可以用来表示色相
    L = 70  # 这里选择固定的亮度 L*
    C = 50  # 固定的色度 C*
    
    # 将 hue_angle 转换为弧度制并计算 a* 和 b* 值
    a = C * np.cos(np.radians(hue_angle))
    b = C * np.sin(np.radians(hue_angle))
    
    # CIELAB 颜色空间下的 (L*, a*, b*) 转换为 RGB
    lab_color = cs.cspace_convert([L, a, b], "CIELab", "sRGB1")
    
    # 确保 RGB 值在 [0, 1] 范围内
    lab_color = np.clip(lab_color, 0, 1)
    return lab_color

def plot_uv_by_hue(hue_angle_list, u_prime_list, v_prime_list):
    for hue in range(0, 360, 10):
        # 提取对应色度角下的所有 u', v' 坐标
        uv_points = [(u, v) for i, (hue_angle, u, v) in enumerate(zip(hue_angle_list, u_prime_list, v_prime_list)) if hue_angle == hue]
        
        if uv_points:  # 如果存在对应色度角的u'v'点
            u_values, v_values = zip(*uv_points)  # 将u'和v'坐标分开

            # 绘制这些点
            plt.scatter(u_values, v_values, label=f'Hue {hue}°')

            # 只在第一个点附近标注色度角
            u, v = uv_points[0]
            plt.text(u, v, f'{hue}°', fontsize=8, ha='right')  # 仅标注第一个点

import random
def plot_uv_by_hue_1(hue_angle_list, u_prime_list, v_prime_list):
    # Generate random colors for each hue angle
    colors = [tuple(random.random() for _ in range(3)) for _ in range(0, 360, 10)]

    for index, hue in enumerate(range(0, 360, 10)):
        # Extract all u', v' coordinates for the current hue angle
        uv_points = [(u, v) for hue_angle, u, v in zip(hue_angle_list, u_prime_list, v_prime_list) if hue_angle == hue]

        if uv_points:  # If there are u'v' points for this hue angle
            u_values, v_values = zip(*uv_points)  # Separate u' and v' coordinates

            # Plot these points with a unique random color
            plt.scatter(u_values, v_values, color=colors[index], label=f'Hue {hue}°')

            # Annotate the first point for the hue angle
            u, v = uv_points[0]
            plt.text(u, v, f'{hue}°', fontsize=8, ha='right')

    
    
def uv_prime_to_xy(uv_prime):
    u_prime, v_prime = uv_prime
    denominator = 6 * u_prime - 16 * v_prime + 12
    x = (9 * u_prime) / denominator
    y = (4 * v_prime) / denominator
    return np.array([x, y])

filename = 'LCH_gamut.xlsx'  # *************************************uv_hull_points.xlsx
sheet_name = 'pointer_LCHab'

# Read Excel data
df = pd.read_excel(io = filename, sheet_name=sheet_name, index_col=0)

# Extract hue angles, lightness values, chroma values
hue = df.index.to_numpy()  # Hue angles (0-350, first column)
L_values = df.columns.to_numpy(dtype=float)  # Lightness values (5, 10, ..., 95, first row)
Chroma = df.to_numpy()  # Chroma values (36x19 array)

# 1. No Convex Hull: Find the max C*ab for each hue angle and calculate u'v'
u_prime_no_hull = []
v_prime_no_hull = []
hue_angle_no_hull = []
for j, h in enumerate(hue):
    # Find the maximum C*ab for each hue angle across all lightness values
    min_C = np.min(Chroma[j, :])
    L_min_index = np.argmin(Chroma[j, :])
    L_min = L_values[L_min_index]  # Get corresponding L* value
    LCH = np.array([L_min, min_C, h])
    if Data_format == 'LCHab':
        uv_prime = LCHab_to_uv(LCH, ILLUM_S)  # uv_prime = LCHab_to_uv(LCHab, ILLUM_S)
    else:
        uv_prime = LCHuv_to_uv(LCH, ILLUM_S)
    u_prime, v_prime = uv_prime[0], uv_prime[1]

    # Store results
    u_prime_no_hull.append(u_prime)
    v_prime_no_hull.append(v_prime)
    hue_angle_no_hull.append(h)
# Now ensure the curve is closed by duplicating the 0-degree point at 360 degrees
u_prime_no_hull.append(u_prime_no_hull[0])
v_prime_no_hull.append(v_prime_no_hull[0])
hue_angle_no_hull.append(360)
aa = 1


# 2  Convert all LAB points to u'v' for Convex Hull
u_prime_list = []
v_prime_list = []
hue_angle_list = []

# Convert all LAB points to u'v'
for i, L in enumerate(L_values):
    for j, h in enumerate(hue):
        C = Chroma[j, i]  # Chroma value at (h_ab, L)
        LCH = np.array([L, C, h])
        if Data_format == 'LCHab':
            uv_prime = LCHab_to_uv(LCH, ILLUM_S)  # uv_prime = LCHab_to_uv(LCHab, ILLUM_S)
        else:
            uv_prime = LCHuv_to_uv(LCH, ILLUM_S)
        u_prime, v_prime = uv_prime[0], uv_prime[1]   ###############
        u_prime_list.append(u_prime)
        v_prime_list.append(v_prime)
        hue_angle_list.append(h)
# Convert to numpy array for convex hull computation
uv_points = np.column_stack((u_prime_list, v_prime_list))


def find_uv_by_hue(hue_angle, hue_angles, uv_points):
    # 找到所有与给定hue_angle匹配的索引
    indices = np.where(hue_angles == hue_angle)[0]
    # 提取对应的uv点
    corresponding_uv_points = uv_points[indices]
    return corresponding_uv_points




plt.gca().set_aspect('equal', adjustable='box')
# 获取当前的坐标系 (ax)
ax = plt.gca()
plt.axis((-0.1, 0.7, -0.1, 0.7))
plt.show()

plot_uv_by_hue_1(hue_angle_list, u_prime_list, v_prime_list)

uv_points_1 = np.column_stack((u_prime_list, v_prime_list))
hull_angles, new_contour_points,closest_points= ConvexHull_general(hue_angle_list, uv_points_1)
closest_points_list =  np.array([point[1] for point in closest_points])

# 绘制 CIE 1976UCS 色域图
plot_chromaticity_diagram_CIE1976UCS(show_diagram_colours=False, standalone=False) 


# 获取当前的坐标系 (ax)
ax = plt.gca()
plt.axis((-0.1, 0.7, -0.1, 0.7))
plt.show()
u2, v2 = closest_points_list[:0], closest_points_list[:1]
# 绘制缺失hue的uv点
plt.scatter(closest_points_list[:,0], closest_points_list[:,1], color='red', s=20)
# 绘制convexhull外轮廓线
u_hull_point, v_hull_point = new_contour_points[:,0], new_contour_points[:,1]
plt.plot(u_hull_point, v_hull_point, color='red', label='Concave Hull')

from matplotlib.colors import hsv_to_rgb
# 根据 hue_angle_list 生成 CIELAB 对应的颜色
colors = [hue_to_lab_color(hue) for hue in hue_angle_list]
# 在 u'v' 平面上绘制散点图
plt.scatter(u_prime_list, v_prime_list, color=colors, s=20)  # s 设置点的大小

plt.title("Scatter plot of u'v' with colors based on hue_angle")
plt.axis((-0.1, 0.7, -0.1, 0.7))
plt.xlabel("u'")
plt.ylabel("v'")
plt.grid(True)
plt.legend()
plt.show()

aa = 1