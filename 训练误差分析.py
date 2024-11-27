import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import time
import random


def run_progress_bar():
    # 随机生成一个持续时间（10到30秒之间）
    duration1 = random.randint(10, 15)
    st.write("运算运行中")

    # 初始化进度条
    progress_bar = st.progress(0)

    # 模拟任务进度
    current_progress = 0
    start_time = time.time()

    for i in range(duration1):
        time.sleep(1)

        elapsed_time = time.time() - start_time
        remaining_time = duration1 - elapsed_time

        if remaining_time > 0:
            # 计算下一个增量大小
            increment = min(1.0 - current_progress, 1.0 / remaining_time)
        else:
            increment = 1.0 - current_progress

        new_progress = current_progress + increment

        progress_bar.progress(new_progress)
        current_progress = new_progress

    st.write("已完成，正在输出图像")


# 设置页面标题
st.title('FUELCOSP 模型分析')
# 用户输入阈值（在 Streamlit 中，我们可以直接在界面上输入）
threshold = st.number_input("请输入模型误差允许的阈值（百分比）：", value=1.0)

# 读取模型
model1 = joblib.load('过程文件/model1.pkl')

# 读取训练结果数据
data2 = pd.read_excel('过程文件/FUELCOSP训练误差.xlsx')
MSE = data2['MSE'].values[0]
R2 = data2['R2'].values[0]


# 定义一个函数来处理数据和绘图
def process_and_plot_data():
    # 读取CSV文件
    data = pd.read_csv('过程文件/筛选结果.csv')

    # 数据预处理和反归一化准备
    data = data[(data['FUELCOSP'].astype(float) >= 0) & (data['FUELCOSP'].astype(float) <= 400)]
    data['input1'] = data['SPEED'].astype(float)
    data['input2'] = data['RL_MESS'].astype(float)
    data['input3'] = data['InCamAg'].astype(float)
    data['input4'] = data['OutCamAg'].astype(float)
    data['target1'] = data['FUELCOSP'].astype(float)
    min_max_values = {
        'input1': [np.min(data['input1']), np.max(data['input1'])],
        'input2': [np.min(data['input2']), np.max(data['input2'])],
        'input3': [np.min(data['input3']), np.max(data['input3'])],
        'input4': [np.min(data['input4']), np.max(data['input4'])],
        'target1': [np.min(data['target1']), np.max(data['target1'])],
    }

    # 预测结果和误差存储列表
    predicted_outputs = []
    errors = []

    # 自动输入数据并预测
    for _, row in data.iterrows():
        input_data = np.array([
            (row['input1'] - min_max_values['input1'][0]) / (min_max_values['input1'][1] - min_max_values['input1'][0]),
            (row['input2'] - min_max_values['input2'][0]) / (min_max_values['input2'][1] - min_max_values['input2'][0]),
            (row['input3'] - min_max_values['input3'][0]) / (min_max_values['input3'][1] - min_max_values['input3'][0]),
            (row['input4'] - min_max_values['input4'][0]) / (
                        min_max_values['input4'][1] - min_max_values['input4'][0])]).reshape(1, -1)

        predicted_output = (min_max_values['target1'][1] - min_max_values['target1'][0]) * model1.predict(input_data) + \
                           min_max_values['target1'][0]
        error_percentage = float((predicted_output.item() - row['target1']) / row['target1'] * 100)
        predicted_outputs.append(predicted_output[0])
        errors.append(error_percentage)

    # 创建预测结果DataFrame
    predicted_data = pd.DataFrame(
        {'Predicted_FUELCOSP': predicted_outputs, 'FUELCOSP': data['target1'], 'Error(%)': errors})

    data1 = predicted_data
    Predicted_FUELCOSP = data1['Predicted_FUELCOSP']
    FUELCOSP = data1['FUELCOSP']
    error = data1['Error(%)']



    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制散点图
    colors = ['blue' if threshold * (-1) <= e <= threshold else 'red' for e in error]
    ax1.scatter(Predicted_FUELCOSP, FUELCOSP, c=colors, label='Data Points', s=5)
    ax1.set_title('Correlation')
    ax1.set_xlabel('FUELCOSP')
    ax1.set_ylabel('Predicted_FUELCOSP')

    # 动态计算合适的 bins 数量
    bins = np.histogram_bin_edges(error, bins='scott')

    # 绘制误差值 Error 的正态分布图，并添加边框
    ax2.hist(error, bins, alpha=0.6, color='g', label='Error Density', edgecolor='black', linewidth=1.2)
    ax2.set_title('Normal Distribution of Error')
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Density')
    ax2.legend()

    # 添加散点图图例
    blue_patch = mpatches.Patch(color='blue', label=f'Error <= {threshold}')
    red_patch = mpatches.Patch(color='red', label=f'Error > {threshold}')
    ax1.legend(handles=[blue_patch, red_patch])

    # 添加MSE和R2的图例，限制小数位数为4位
    mse_patch = mpatches.Patch(label=f'MSE: {MSE: .4f}')
    r2_patch = mpatches.Patch(label=f'R²: {R2: .4f}')
    ax2.legend(handles=[mse_patch, r2_patch], loc='upper right')

    # 显示图表
    st.pyplot(fig)


# 添加按钮
if st.button('执行分析'):
    # 在 Streamlit 应用中调用进度条函数
    run_progress_bar()
    process_and_plot_data()