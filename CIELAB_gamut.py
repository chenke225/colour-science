
#  LCHab, LCHuv数据到u'v'转换，绘制3D和2D色域图。 其他标准光源到D65 chromatic adaptation 转换。
# 查找u'v'平面uv距离最大者，构成所有点的外轮廓。添加hue角度标注。
# 所用函数在gamut_util.py
# python 3.12.3 通过测试 ， 2020-12-20
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import colour
from colour.adaptation import chromatic_adaptation_VonKries

from scipy.spatial import ConvexHull
from openpyxl import Workbook
from colour.plotting import plot_chromaticity_diagram_CIE1976UCS
from gamut_util import LCHab_to_XYZ, LCHuv_to_XYZ, XYZ_to_uv
from gamut_util import read_excel, annotate_hue_angle, write_excel
import matplotlib

matplotlib.use('TkAgg')
plt.ion()
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端


# *******************************************************************************************************
ILLUM_C = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]['C']  # CIE xy chromaticity coordinates
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]    # CIE xy chromaticity coordinates
D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]    # CIE xy chromaticity coordinates
# 转换为XYZ白点 (假设 Y = 1)
def xy_to_XYZ(xy, Y=1.0):
    x, y = xy
    X = (Y / y) * x
    Z = (Y / y) * (1 - x - y)
    return np.array([X, Y, Z])

D50_white = xy_to_XYZ(D50)  # D50白点的XYZ值
D65_white = xy_to_XYZ(D65)  # D65白点的XYZ值
ILLUM_C_white = xy_to_XYZ(ILLUM_C)

filename = 'LCH_gamut.xlsx'  # excel数据文件名
sheet_name = '12640_LCHab'  # 表单名  # ****************************************
data_format = 'LCHab'
ILLUM_S = D50
ILLUM_S_white = D50_white

# 读取 Excel 数据
L_values, LCH_array, hue_angle_array = read_excel(filename, sheet_name)
hue = hue_angle_array[0:36]

if data_format == 'LCHab':
    XYZ = LCHab_to_XYZ(LCH_array, ILLUM_S ) 
else:
    XYZ = LCHuv_to_XYZ(LCH_array, ILLUM_S ) 
XYZ_D65 = chromatic_adaptation_VonKries(XYZ, ILLUM_S_white, D65_white, transform="Bradford")
XYZ = XYZ_D65
u_prime, v_prime = XYZ_to_uv(XYZ)
# 假设 u_prime 和 v_prime 是 shape=(576,) 的一维数组
u_prime_reshaped = u_prime.reshape(len(L_values), len(hue)).T
v_prime_reshaped = v_prime.reshape(len(L_values), len(hue)).T

L = np.tile(L_values, (len(hue), 1))  # 扩展到 (36, 16)
L_reshaped = L.reshape(len(hue), len(L_values))

# 1. 找到每个色调角度下最大 L* 的 (u', v') 点
u_Lmax, v_Lmax = [], []
for i in range(len(hue)):
    max_L_index = np.argmax(L_reshaped[i, :])  # 找到每个 hue 下的最大 L* 索引
    u_Lmax.append(u_prime_reshaped[i, max_L_index])
    v_Lmax.append(v_prime_reshaped[i, max_L_index])

# 2. 计算所有 (u'_Lmax, v'_Lmax) 的中心点
u_Lmax_center = np.mean(u_Lmax)
v_Lmax_center = np.mean(v_Lmax)

# 3. 找到距离中心点 (u'_Lmax_center, v'_Lmax_center) 最远的点
u_contour, v_contour, L_contour = [], [], []
for i in range(len(hue)):
    # 计算每个 (u', v') 到中心点的距离
    distances = np.sqrt((u_prime_reshaped[i, :] - u_Lmax_center)**2 + (v_prime_reshaped[i, :] - v_Lmax_center)**2)
    max_dist_index = np.argmax(distances)  # 找到最大距离对应的索引

    # 记录最大距离点的 u', v' 和 L* 值
    u_contour.append(u_prime_reshaped[i, max_dist_index])
    v_contour.append(v_prime_reshaped[i, max_dist_index])
    L_contour.append(L_reshaped[i, max_dist_index])

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制外轮廓点（红色标记）
sc = ax.scatter(u_contour, v_contour, L_contour, c='red', s=50, label="Contour points") # type: ignore
# 设置轴标签
ax.set_xlabel("u'")
ax.set_ylabel("v'")
ax.set_zlabel("L*") # type: ignore
# 显示颜色条
plt.colorbar(sc, ax=ax, label="L* Intensity")
plt.legend()

# 绘制 CIE 1976UCS 色域图
plot_chromaticity_diagram_CIE1976UCS(show_diagram_colours=False, color ='lightgray', standalone=False) 
# 强制保持坐标轴比例为1:1
plt.gca().set_aspect('equal', adjustable='box')
plt.axis((-0.1, 0.7, -0.1, 0.7))

plt.scatter(u_prime, v_prime, c='gray', s=10, label="All u'v' points")
#  添加起始点到末尾，形成闭合轮廓点曲线。
hue_angle_contour = hue
hue_angle_contour = np.append(hue_angle_contour, hue_angle_contour[0])
u_contour.append(u_contour[0])
v_contour.append(v_contour[0])
uv_contour = np.column_stack((u_contour, v_contour))

plt.plot(u_contour, v_contour, 'r-', marker='o', label="Contour curve")  # 绘制轮廓点曲线
# 标注色度角 (hue angle)
annotate_hue_angle(hue_angle_contour, uv_contour)

plt.xlabel("u'")
plt.ylabel("v'")
plt.title("u'v' plane with contour curve")
plt.legend()
plt.show()

#  **********************************************************
filename = 'uv_gamut.xlsx'  # 变换结果(u, v)写入Excel
sheet_name = '12640_LCHab_test'   # ***************************************************
write_excel(filename, sheet_name, hue_angle = hue_angle_contour, uv_points= uv_contour)


aa = 1