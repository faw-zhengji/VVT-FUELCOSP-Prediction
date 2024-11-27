import io
import streamlit as st
import pandas as pd

# 设置标题
st.title("Excel数据处理应用")
st.write("此界面为数据清洗过程，筛选出异常值，降低后续处理难度")
# 上传Excel文件
uploaded_file = st.file_uploader("请上传Excel文件", type=["xlsx"])

if uploaded_file is not None:
    # 读取上传的Excel文件
    excel_file = pd.ExcelFile(uploaded_file)

    # 读取选定的sheet
    df = excel_file.parse(0)
    st.subheader("原始数据：")
    st.write(df)

    # 手动填写的筛选条件
    st.write("填写数据筛选条件")
    max_fuelcosp = st.number_input("油耗上限值", value=400, min_value=0, max_value=1000)
    max_speed = st.number_input("转速上限值", value=5500, min_value=0, max_value=10000)
    max_rl_mess = st.number_input("最大RL_MESS", value=150, min_value=0, max_value=1000)
    max_cimep0 = st.number_input("最大CIMEP0", value=3, min_value=0, max_value=10)

    # 过滤+筛选数据
    filtered_df = df[(df['FUELCOSP'] > 0) & (df['FUELCOSP'] < max_fuelcosp) &
                     (df['SPEED'] < max_speed) &  # 确保这是您想要的逻辑
                     (df['RL_MESS'] > 0) & (df['RL_MESS'] < max_rl_mess) &  # 假设RL_MESS的最小有效值大于0
                     (df['CIMEP0'] > 0) & (df['CIMEP0'] < max_cimep0)] \
            [['SPEED', 'RL_MESS', 'wnwse_w', 'wnwsa_w', 'FUELCOSP', 'CIMEP0', 'PC_CoDc', 'EM_CO2_1', 'EM_THC_1', 'COCO2', 'MFB50_0']]

    # 检查并处理缺失值
    filtered_df_cleaned = filtered_df.dropna()  # 简化缺失值处理

    # 显示筛选后的数据
    st.subheader("筛选后的数据：")
    st.write(filtered_df_cleaned)

    # 将DataFrame保存到内存中的Excel文件
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        filtered_df_cleaned.to_excel(writer, index=False, sheet_name='筛选结果')
    excel_buffer.seek(0)  # 重置指针到文件开始

    # 下载筛选后的Excel数据
    st.download_button(
        label="下载筛选后的数据",
        data=excel_buffer.read(),
        file_name='筛选结果.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
else:
    st.warning("请上传一个Excel文件。")