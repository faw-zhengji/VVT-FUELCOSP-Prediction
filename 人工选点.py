import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 设置页面标题
st.title("当前人工选点方法")
st.write("此界面为当前人工开展VVT优化的方法，通过对比不同参数的多张MAP找到最优VVT")

# 设置页面标题
st.subheader("多张MAP生成")

# 上传Excel文件
uploaded_file = st.file_uploader("上传Excel文件", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 读取Excel文件
    df = pd.read_excel(uploaded_file)

    # 显示数据框
    st.write("数据预览：")
    st.dataframe(df.head())

    # 选择X轴和Y轴
    x_axis = st.selectbox('选择X轴列', list(df.columns))
    y_axis = st.selectbox('选择Y轴列', list(df.columns))

    # 确保X轴和Y轴不同
    if x_axis == y_axis:
        st.error("X轴和Y轴不能相同！")
    else:
        # 检查x和y是否包含多个唯一值
        if len(df[x_axis].unique()) < 2 or len(df[y_axis].unique()) < 2:
            st.error("X轴或Y轴必须包含多个值才能生成等高线图！")
        else:
            # 剩余的列为Z轴
            z_axes = [col for col in df.columns if col not in [x_axis, y_axis]]

            # 用户可以选择哪些Z轴变量显示
            selected_z_axes = st.multiselect('选择要绘制为Z轴的列', z_axes)

            # 插值方法选择
            interpolation_methods = ['linear', 'nearest', 'cubic']
            interpolation_method = st.selectbox('选择插值方法', interpolation_methods)

            # 控制是否显示数据点的checkbox
            show_data_points = st.checkbox('图上显示数据点')

            for z_axis in selected_z_axes:
                # 提取数据
                x = df[x_axis].values
                y = df[y_axis].values
                z = df[z_axis].values

                # 创建网格
                xi = np.linspace(min(x), max(x), 1000)
                yi = np.linspace(min(y), max(y), 1000)
                xi, yi = np.meshgrid(xi, yi)

                # 使用scipy.interpolate.griddata进行插值
                try:
                    zi = griddata((x, y), z, (xi, yi), method=interpolation_method)
                except ValueError as e:
                    st.error(f"插值过程中发生错误：{e}")
                    continue

                # 绘制等高线图
                fig, ax = plt.subplots()
                contour = ax.contourf(xi, yi, zi, 400, cmap='viridis')
                ax.set_xlabel(f"{x_axis}")
                ax.set_ylabel(f"{y_axis}")
                ax.set_title(f"{z_axis}")
                fig.colorbar(contour, ax=ax, label=f"{z_axis} colour")

                # 绘制等高线
                contour_lines = ax.contour(xi, yi, zi, 10, colors='black', linewidths=0.5)
                ax.clabel(contour_lines, inline=1, fontsize=7, fmt='%1.1f')

                # 根据checkbox的状态决定是否显示数据点
                if show_data_points:
                    scatter = ax.scatter(x, y, c='red', marker='o', s=2, label='Data Points')
                    legend = ax.legend(handles=[scatter], loc='upper right')
                    for i in range(len(x)):
                        ax.text(x[i], y[i], f'{z[i]:.2f}', ha='center', va='bottom', fontsize=3, color='red')

                st.pyplot(fig)