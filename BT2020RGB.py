
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colour

matplotlib.use('TkAgg')
plt.ion()
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端

# Define the color space and white point
cs_name = "ITU-R BT.2020"
colourspace = colour.RGB_COLOURSPACES[cs_name]
illuminant = colourspace.whitepoint

# Increase sampling density for RGB points along 0-r, 0-g, and 0-b lines
samples = 300
n_values = np.linspace(0, 1, samples)
RGB_points = np.array([[n, 0, 0] for n in n_values] + 
                      [[0, n, 0] for n in n_values] + 
                      [[0, 0, n] for n in n_values])

RGB_points = np.array([[n, 1, 1] for n in n_values] + 
                      [[1, n, 1] for n in n_values] + 
                      [[1, 1, n] for n in n_values])



# Convert RGB points to XYZ, then to CIELUV (u*, v*) coordinates
XYZ_points = colour.RGB_to_XYZ(RGB_points, colourspace, illuminant)
Luv_points = colour.XYZ_to_Luv(XYZ_points, illuminant=illuminant)
L_values, u_values, v_values = Luv_points[..., 0], Luv_points[..., 1], Luv_points[..., 2]

# Define the L* intervals for contour levels
interval = 2
L_levels = np.arange(0, 100, interval)

# Prepare the figure
fig, ax = plt.subplots(figsize=(8, 8))

# Function to get midpoint point for each L* interval along each line
def get_midpoint_points(L_values, u_values, v_values, L_levels, interval):
    contour_points_u, contour_points_v, contour_points_L = [], [], []
    for L in L_levels:
        # 筛选出L值在间隔中的点
        mask = (L_values >= L) & (L_values < L + interval)
        u_interval, v_interval = u_values[mask], v_values[mask]
        
        # 如果在此间隔内有点，则选择中间点
        if len(u_interval) > 0:
            mid_idx = len(u_interval) // 2  # 选择间隔中的中间点
            contour_points_u.append(u_interval[mid_idx])
            contour_points_v.append(v_interval[mid_idx])
            contour_points_L.append(L)  # 将L值添加到对应的contour列表中
    
    return contour_points_u, contour_points_v, contour_points_L



# 初始化一个字典来存储每个颜色线的contour点
contour_data = {'r': {'u': [], 'v': [], 'L': []},
                'g': {'u': [], 'v': [], 'L': []},
                'b': {'u': [], 'v': [], 'L': []}}

# 数据收集循环，r、g、b线
for color, line_mask in [('r', (RGB_points[:, 1] == 0) & (RGB_points[:, 2] == 0)), 
                         ('g', (RGB_points[:, 0] == 0) & (RGB_points[:, 2] == 0)), 
                         ('b', (RGB_points[:, 0] == 0) & (RGB_points[:, 1] == 0))]:

    # 根据line_mask筛选出相关点
    L_values_line = L_values[line_mask]
    u_values_line = u_values[line_mask]
    v_values_line = v_values[line_mask]
    
    # 获取contour点以及它们对应的L值
    u_contour, v_contour, L_contour = get_midpoint_points(L_values_line, u_values_line, v_values_line, L_levels, interval)
    
    # 将这些contour点和L值存储到contour_data中
    contour_data[color]['u'].extend(u_contour)
    contour_data[color]['v'].extend(v_contour)
    contour_data[color]['L'].extend(L_contour)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 8))
import matplotlib.cm as cm
# 定义L级别的色图
cmap = cm.get_cmap('rainbow', len(L_levels))

import matplotlib.colors as mcolors

# 创建离散的颜色映射
num_levels = len(L_levels)

discrete_cmap = mcolors.ListedColormap(plt.cm.tab20.colors[:num_levels])  # type: ignore # 选择足够的颜色


# 绘制每条射线的L等高线
for color in contour_data.keys():
    L_values_line = np.array(contour_data[color]['L'])  # 当前线的L值
    u_values_line = np.array(contour_data[color]['u'])
    v_values_line = np.array(contour_data[color]['v'])
    
    # 根据L值区间进行绘制
    for i, L in enumerate(L_levels):
        # 获取当前L区间的mask
        mask = (L_values_line >= L) & (L_values_line < L + interval)
        
        # 筛选出该区间的u、v值
        u_contour = u_values_line[mask]
        v_contour = v_values_line[mask]
        
        # 确保mask的尺寸匹配
        if len(u_contour) > 0 and len(v_contour) > 0:
            # 根据L级别为每个区间绘制点，使用离散色图
            ax.scatter(u_contour, v_contour, color=discrete_cmap(i), label=f"{color} line L*={L}-{L+interval}", s=10)


# 添加标签和图例
ax.set_xlabel("u*")
ax.set_ylabel("v*")
ax.set_title("L* Contours along RGB lines in u*v* plane")
ax.legend()
plt.show()

plt.show()
