import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import streamlit as st
import plotly.graph_objs as go

# Streamlit标题
st.title("AI预测VVT")

# 读取CSV文件
data = pd.read_csv('过程文件/筛选结果.csv')

# 数据预处理
data['input1'] = data['SPEED'].astype(float)
data['input2'] = data['RL_MESS'].astype(float)
data['input3'] = data['InCamAg'].astype(float)
data['input4'] = data['OutCamAg'].astype(float)
data['target1'] = data['FUELCOSP'].astype(float)
data['target2'] = data['CIMEP0'].astype(float)

# 计算最大最小值
min_max_values = {
    'input1': [np.min(data['input1']), np.max(data['input1'])],
    'input2': [np.min(data['input2']), np.max(data['input2'])],
    'input3': [np.min(data['input3']), np.max(data['input3'])],
    'input4': [np.min(data['input4']), np.max(data['input4'])],
    'target1': [np.min(data['target1']), np.max(data['target1'])],
    'target2': [np.min(data['target2']), np.max(data['target2'])]
}

# 加载模型
model1 = joblib.load('过程文件/model1.pkl')
model2 = joblib.load('过程文件/model2.pkl')

# 输入SPEED和RL_MESS的值
num1 = st.number_input("请输入SPEED值：", value=1500)
num2 = st.number_input("请输入RL_MESS值：", value=60)

predicted_outputs = []

# 自动输入InCamAg和OutCamAg的值
for num3 in range(round(min_max_values['input3'][0]), 1 + round(min_max_values['input3'][1]), 1):
    for num4 in range(round(min_max_values['input4'][0]), 1 + round(min_max_values['input4'][1]), 1):
        input_data = np.array([[
            (num1 - min_max_values['input1'][0]) / (min_max_values['input1'][1] - min_max_values['input1'][0]),
            (num2 - min_max_values['input2'][0]) / (min_max_values['input2'][1] - min_max_values['input2'][0]),
            (num3 - min_max_values['input3'][0]) / (min_max_values['input3'][1] - min_max_values['input3'][0]),
            (num4 - min_max_values['input4'][0]) / (min_max_values['input4'][1] - min_max_values['input4'][0])
        ]])

        # 将预测的输出转换回原始数值范围
        FUELCOSP = (min_max_values['target1'][1] - min_max_values['target1'][0]) * model1.predict(input_data) + min_max_values['target1'][0]
        CIMEP = (min_max_values['target2'][1] - min_max_values['target2'][0]) * model2.predict(input_data) + min_max_values['target2'][0]

        # 将 num3 和 num4 转换为浮点数
        num3 = float(num3)
        num4 = float(num4)
        FUELCOSP = float(FUELCOSP.item())  # 提取单个元素
        CIMEP = float(CIMEP.item())  # 提取单个元素

        # 添加到 predicted_outputs
        predicted_outputs.append((num3, num4, FUELCOSP, CIMEP))

# 找到预测结果最小值对应的InCamAg、OutCamAg值、筛选CIMEP小于三
min_output = min(predicted_outputs, key=lambda x: (x[3] < 300, x[2]))
min_InCamAg = min_output[0]
min_OutCamAg = min_output[1]
min_predicted_output = min_output[2]

# 欧氏距离计算
data['Euclidean_distance'] = np.sqrt((data['input1'] - num1) ** 2 + (data['input2'] - num2) ** 2 + (data['input3'] - min_InCamAg) ** 2 + (data['input4'] - min_OutCamAg) ** 2)

# 找到最小欧氏距离对应的'FUELCOSP'值
min_distance_row = data.loc[data['Euclidean_distance'].idxmin()]['target1']

# 计算误差率
error = 100 * (min_predicted_output - min_distance_row) / min_distance_row
error = abs(error) * 10
error = 100 - error
rounded_arr = np.around(error, 2)  # 保留两位小数
st.write("最佳预测油耗：", min_predicted_output)
st.write("对应的进气VVT：", min_InCamAg)
st.write("对应的排气VVT：", min_OutCamAg)
st.write("燃烧稳定性：", min_output[3])
st.write("可信度：", rounded_arr, '%')


# 提取InCamAg、OutCamAg和predicted_output的值
InCamAg_values = [output[0] for output in predicted_outputs]
OutCamAg_values = [output[1] for output in predicted_outputs]
predicted_output_values = [output[2] for output in predicted_outputs]

# 绘制三维散点图
fig = go.Figure(data=[go.Scatter3d(
    x=InCamAg_values,
    y=OutCamAg_values,
    z=predicted_output_values,
    mode='markers',
    marker=dict(
        size=4,
        color=predicted_output_values,  # 根据预测输出值着色
        colorscale='Viridis',
        opacity=0.8
    )
)])

# 添加最小值点
fig.add_trace(go.Scatter3d(
    x=[min_InCamAg],
    y=[min_OutCamAg],
    z=[min_predicted_output],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        symbol='x'
    )
))

# 设置图表布局
fig.update_layout(
    scene=dict(
        xaxis_title='InCamAg',
        yaxis_title='OutCamAg',
        zaxis_title='Predicted FUELCOSP'
    ),
    title="Predicted FUELCOSP vs. InCamAg and OutCamAg",
    width=800,  # 增加图表宽度
    height=800   # 增加图表高度
)

# 显示三维图
st.plotly_chart(fig)


# 提取 'SPEED' 和 'RL_MESS' 列等于特定值的行
specific_rows = data[((data['SPEED'] >= num1 - 50) & (data['SPEED'] <= num1 + 50)) &
                     ((data['RL_MESS'] >= num2 - 3) & (data['RL_MESS'] <= num2 + 3))]

# 检查是否提取到了数据
if specific_rows.empty:
    st.warning("此工况实际未测量，无法画出二维图像。")
else:
    # 如果有数据，继续处理
    specific_speed_values = specific_rows['SPEED'].values
    specific_rl_mess_values = specific_rows['RL_MESS'].values
    specific_in_camag_values = specific_rows['InCamAg'].values
    specific_out_camag_values = specific_rows['OutCamAg'].values
    specific_fuelcosp_values = specific_rows['FUELCOSP'].values

    # 绘制二维图
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 绘制原始图
    x = np.linspace(min(specific_in_camag_values), max(specific_in_camag_values), 1000)
    y = np.linspace(min(specific_out_camag_values), max(specific_out_camag_values), 1000)
    X, Y = np.meshgrid(x, y)
    Z = griddata((specific_in_camag_values, specific_out_camag_values), specific_fuelcosp_values, (X, Y),
        method='linear')
    ax1.contourf(X, Y, Z, cmap='viridis')
    contour = ax1.contour(X, Y, Z, colors='black', linewidths=0.5)
    ax1.clabel(contour, inline=False, fontsize=8)
    ax1.set_xlabel('InCamAg')
    ax1.set_ylabel('OutCamAg')
    ax1.set_title('Actual FUELCOSP vs. InCamAg and OutCamAg')

    # 创建预测图
    sc = ax2.scatter(InCamAg_values, OutCamAg_values, c=predicted_output_values, cmap='viridis', s=300)
    ax2.scatter(min_InCamAg, min_OutCamAg, c='r', marker='x', label='Lowest Point')
    ax2.set_xlabel('InCamAg')
    ax2.set_ylabel('OutCamAg')
    ax2.set_title('Predicted FUELCOSP vs. InCamAg and OutCamAg')

    x = np.array(InCamAg_values)
    y = np.array(OutCamAg_values)
    z = np.array(predicted_output_values)
    contour = ax2.contour(x.reshape(1 + round(min_max_values['input3'][1]) - round(min_max_values['input3'][0]),
                                    1 + round(min_max_values['input4'][1]) - round(min_max_values['input4'][0])),
        y.reshape(1 + round(min_max_values['input3'][1]) - round(min_max_values['input3'][0]),
                  1 + round(min_max_values['input4'][1]) - round(min_max_values['input4'][0])),
        z.reshape(1 + round(min_max_values['input3'][1]) - round(min_max_values['input3'][0]),
                  1 + round(min_max_values['input4'][1]) - round(min_max_values['input4'][0])),
        7, colors='black', linestyles='solid', linewidths=0.5)
    ax2.clabel(contour, inline=False, fontsize=8, colors='black', fmt='%1.2f')

    plt.colorbar(sc, ax=ax2, label='FUELCOSP')
    st.pyplot(fig2)

