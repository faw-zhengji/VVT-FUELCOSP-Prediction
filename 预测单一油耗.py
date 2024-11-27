import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 设置标题
st.title('AI预测FUELCOSP')

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

# 创建输入框
speed = st.number_input("请输入SPEED值：")
rl_mess = st.number_input("请输入RL_MESS值：")
in_cam_ag = st.number_input("请输入进气VVT：")
out_cam_ag = st.number_input("请输入排气VVT：")
predict_button = st.button("预测")
if predict_button:
    # 预处理输入数据
    input_data = np.array([[(speed - min_max_values['input1'][0]) / (min_max_values['input1'][1] - min_max_values['input1'][0]),
                            (rl_mess - min_max_values['input2'][0]) / (min_max_values['input2'][1] - min_max_values['input2'][0]),
                            (in_cam_ag - min_max_values['input3'][0]) / (min_max_values['input3'][1] - min_max_values['input3'][0]),
                            (out_cam_ag - min_max_values['input4'][0]) / (min_max_values['input4'][1] - min_max_values['input4'][0])]])

    # 预测输出
    predicted_output = (min_max_values['target1'][1] - min_max_values['target1'][0]) * model1.predict(input_data) + min_max_values['target1'][0]
    cimep = (min_max_values['target2'][1] - min_max_values['target2'][0]) * model2.predict(input_data) + min_max_values['target2'][0]

    # 将 num3 和 num4 转换为浮点数
    predicted_output = float(predicted_output.item())  # 提取单个元素
    cimep = float(cimep.item())  # 提取单个元素

    # 计算欧氏距离
    data['Euclidean_distance'] = np.sqrt((data['input1'] - speed) ** 2 + (data['input2'] - rl_mess) ** 2 + (data['input3'] - in_cam_ag) ** 2 + (data['input4'] - out_cam_ag) ** 2)
    min_distance_row = data.loc[data['Euclidean_distance'].idxmin()]['target1']

    # 计算误差率
    error = 100 * (predicted_output - min_distance_row) / min_distance_row
    # 换算可信度
    error_absolute = abs(error) * 10
    confidence = 100 - error_absolute

    # 显示结果
    st.subheader('预测结果')
    st.write("神经网络预测结果：", predicted_output)
    st.write("燃烧稳定性：", cimep)
    if(confidence<0):
        st.write("可信度极低")
    else:
        st.write("可信度：", confidence, '%')
else:
    st.warning("请确保所有输入字段都已填写。")