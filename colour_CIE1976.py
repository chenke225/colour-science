
# 绘制CIE 1976 UCS uv色域图，  python3.12.3 通过测试， 2024-12-20

import colour
from colour.plotting import *
colour_style()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
plt.ion()
print("Currently Using Backend:", matplotlib.get_backend())  # 当前默认后端

# 最初的色域图方法
plot_chromaticity_diagram_CIE1976UCS(show_diagram_colours=False, standalone=False)  #standalone=False
# 强制保持坐标轴比例为1:1
plt.gca().set_aspect('equal', adjustable='box')

illuminant_D50_xy = np.array([0.34567, 0.3585])
illuminant_D50_uv = colour.xy_to_Luv_uv(illuminant_D50_xy)
illuminant_D65_xy = np.array([0.3127, 0.3291])
illuminant_D65_uv = colour.xy_to_Luv_uv(illuminant_D65_xy)
illuminant_C_xy = np.array([0.3101, 0.3162])
illuminant_C_uv = colour.xy_to_Luv_uv(illuminant_C_xy)

ILLUM_C = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]['C']
D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
# 读取Excel文件
excel_file = 'uv_hull_points.xlsx'              # '/home/chenke/文档/CIE gamut/pointer_data.xlsx'


def Lab_converter_1(excel_file, sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)    
    D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
    # 提取列LCHab数据
    column1 = df['Hab/L'].values  # UCS_u_li
    column2 = df['Cab_max'].values
    column3 = df['L'].values    #  Cmax_m_u, Cuv_mC_u
    pointer_three_ab = np.column_stack((column3, column2, column1))
    # 色空间转换
    Lab = colour.LCHab_to_Lab(pointer_three_ab)
    XYZ = colour.Lab_to_XYZ(Lab, D50)  # ILLUM_C
    xy_3d = colour.XYZ_to_xyY(XYZ)  # 第一种转换
    xy_2d = xy_3d[:,  :2]
    return colour.xy_to_Luv_uv(xy_2d)

def LCHab_to_uv(LCHab, ILLUMINATNAT):
    Lab = colour.LCHab_to_Lab(LCHab)
    XYZ = colour.Lab_to_XYZ(Lab, ILLUMINATNAT)  # ILLUM_C     
    xy_3d = colour.XYZ_to_xyY(XYZ)  # 第一种转换
    xy_2d = xy_3d[:,  :2]
    return colour.xy_to_Luv_uv(xy_2d)

def LCHuv_to_uv(LCHuv, ILLUMINATNAT):
    Luv = colour.LCHuv_to_Luv(LCHuv)
    XYZ = colour.Luv_to_XYZ(Luv, ILLUMINATNAT)  # ILLUM_C
    xy_3d = colour.XYZ_to_xyY(XYZ)  # 第一种转换
    xy_2d = xy_3d[:,  :2]
    return colour.xy_to_Luv_uv(xy_2d)


excel_file = 'uv_gamut.xlsx'
sheet_name = 'pointer_MRLI'   # 李老师  pointer  uv数据
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_bound_LI = np.column_stack((column1, column2))

excel_file =  'uv_gamut.xlsx'
sheet_name ='12640_LCHab'
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
ISO12640_boundary= np.column_stack((column1, column2))

sheet_name ='pointer_colour_science'
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_colour_science = np.column_stack((column1, column2))

sheet_name="TC300_LCHab"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
TC300_boundary= np.column_stack((column1, column2))

sheet_name="pointer_LCHab"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_LCHab_boundary= np.column_stack((column1, column2))

sheet_name="pointer_LCHab_no_hull"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_LCHab_no_hull= np.column_stack((column1, column2))

sheet_name="pointer_LCHuv"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_LCHuv_boundary= np.column_stack((column1, column2))

sheet_name="pointer_LCHuv_noD65"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_LCHuv_noD65= np.column_stack((column1, column2))

sheet_name="pointer_LCHab_no_hull_noD65"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_LCHab_no_hull_noD65= np.column_stack((column1, column2))

sheet_name="pointer_LCHab_noD65"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
pointer_LCHab_noD65= np.column_stack((column1, column2))

sheet_name="HP_printer"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
HP_printer= np.column_stack((column1, column2))

sheet_name="PhotoGamut"
df = pd.read_excel(excel_file, sheet_name=sheet_name)
column1 = df[ "u'"]
column2 = df["v'"]
PhotoGamut = np.column_stack((column1, column2))

ax = plt.gca()       # 获取CIE1976UCS的坐标系
plt.axis((-0.1, 0.7, -0.1, 0.7))    #改变坐标轴范围

ITUR_709_xy=([[.64, .33], [.3, .6], [.15, .06]])
ITUR_709_uv=colour.xy_to_Luv_uv(ITUR_709_xy)
ITUR_2020_xy=([[.708, .292], [.170, .797], [.131, .046]])
ITUR_2020_uv=colour.xy_to_Luv_uv(ITUR_2020_xy)
sRGB_uv = ([[0.4507, 0.5229], [0.1249, 0.5625], [0.1754, 0.1578]])
RGB_12640_uv =([[0.5313, 0.5033], [0.1008, 0.5700], [0.1758, 0.1751]])
DCI_P3_xy=([[.68, .32], [.265, .69], [.15, .06]])
DCI_P3_uv=colour.xy_to_Luv_uv(DCI_P3_xy)
pointer_bound_xy= ([[ 0.508, 0.226],  [ 0.538, 0.258], [ 0.588, 0.280], [ 0.637, 0.298], [ 0.659, 0.316],
                                      [ 0.634, 0.351], [ 0.594, 0.391], [ 0.557, 0.427], [ 0.523, 0.462], [ 0.482, 0.491], 
                                      [ 0.444, 0.515], [ 0.409, 0.546], [ 0.371, 0.558], [ 0.332, 0.573], [ 0.288, 0.584], 
                                      [ 0.242, 0.576], [ 0.202, 0.530 ], [ 0.177, 0.454], [ 0.151, 0.389],[ 0.151, 0.330 ],
                                      [ 0.162, 0.295], [ 0.157, 0.266], [ 0.159, 0.245], [ 0.142, 0.214], [ 0.141, 0.195], 
                                      [ 0.129, 0.168], [ 0.138, 0.141], [ 0.145, 0.129], [ 0.145, 0.106], [ 0.161, 0.094], 
                                      [ 0.188, 0.084], [ 0.252, 0.104],[ 0.324, 0.127], [ 0.393, 0.165], [ 0.451, 0.199]  ])    # colour-science 35p pointer_bound

#pointer_bound_uv = colour.xy_to_Luv_uv(pointer_bound_xy)
# matplotlib绘制四个多边形，对应四种颜色空间
gamut_709=patches.Polygon(ITUR_709_uv, linewidth=2, color='green', fill=False)
gamut_ISO12640=patches.Polygon(ISO12640_boundary, linewidth=2, color='purple', fill=False)     #  青色， ISO 12640 pointer 色域数据
gamut_TC300 = patches.Polygon(TC300_boundary, linewidth=2, color= 'blue', fill=False)
gamut_pointer_bound_colour_science = patches.Polygon(pointer_colour_science, linewidth=2, color= 'red', fill=False)
gamut_pointer_bound_LI = patches.Polygon(pointer_bound_LI, linewidth=2, color= 'magenta', fill=False)
gamut_pointer_LCHab =patches.Polygon(pointer_LCHab_boundary, linewidth=2, color='black', fill=False, linestyle = 'dashed')   # 
gamut_pointer_LCHab_noD65 = patches.Polygon(pointer_LCHab_noD65, linewidth=2, color= 'orange', fill=False, linestyle = 'dashed')
gamut_pointer_LCHab_nohull = patches.Polygon(pointer_LCHab_no_hull, linewidth=2, color= 'brown', fill=False)
gamut_pointer_LCHab_no_hull_noD65 = patches.Polygon(pointer_LCHab_no_hull_noD65, linewidth=2, color= 'pink', fill=False)
gamut_2020 = patches.Polygon(ITUR_2020_uv, linewidth=2, color= 'lightblue', fill=False )
gamut_12640_RGB = patches.Polygon(RGB_12640_uv, linewidth=2, color= 'purple', fill=False )
gamut_pointer_LCHuv =patches.Polygon(pointer_LCHuv_boundary, linewidth=2, linestyle = 'dashed', color='olive', fill=False)   # 
gamut_pointer_LCHuv_noD65 = patches.Polygon(pointer_LCHuv_noD65, linewidth=2, linestyle = 'dashed', color='gray', fill=False) 
gamut_HP_Printer = patches.Polygon(HP_printer, linewidth=2, linestyle = 'dashed', color= 'blue', fill=False)
gamut_PhotoGamut = patches.Polygon(PhotoGamut, linewidth=2, linestyle = 'dashed', color= 'orange', fill=False)
plt.show()
#ax.add_patch(gamut_709)
ax.add_patch(gamut_2020)
#ax.add_patch(gamut_12640_RGB)
#ax.add_patch(gamut_sRGB_uv)
#ax.add_patch(gamut_TC300)
ax.add_patch(gamut_ISO12640)
#ax.add_patch(gamut_pointer_LCHab)
#ax.add_patch(gamut_pointer_LCHuv)
#ax.add_patch(gamut_pointer_bound_colour_science)
#ax.add_patch(gamut_pointer_LCHab_nohull)
#ax.add_patch(gamut_pointer_LCHab_noD65)
#ax.add_patch(gamut_pointer_bound_LI)
#ax.add_patch(gamut_pointer_LCHuv_noD65)
ax.add_patch(gamut_PhotoGamut)
ax.add_patch(gamut_HP_Printer)

plt.legend([gamut_2020, gamut_ISO12640, gamut_pointer_LCHab, gamut_pointer_bound_colour_science  ],   
    ['BT.2020',                             '12640_LCHab',     'pointer_LCHab',       'pointer_colourscience'],
    loc='upper right', fontsize=14)  # 对曲线的标注



a=1
