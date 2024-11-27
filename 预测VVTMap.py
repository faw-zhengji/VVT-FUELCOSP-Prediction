import streamlit as st
import pandas as pd

# 读取Excel文件
df1 = pd.read_excel('数据结果/进气MAP.xlsx', index_col=0)  # 假设第一列为索引
df2 = pd.read_excel('数据结果/排气MAP.xlsx', index_col=0)  # 假设第一列为索引

def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: lightcoral' if v else '' for v in is_min]

# 应用色阶
styled_df1 = df1.style.apply(highlight_max, axis=0).apply(highlight_min, axis=0)
styled_df2 = df2.style.apply(highlight_max, axis=0).apply(highlight_min, axis=0)

# 显示样式化的DataFrame
st.markdown("<h2>进气MAP</h2>", unsafe_allow_html=True)
st.markdown(f"<div>{styled_df1.to_html()}</div>", unsafe_allow_html=True)

st.markdown("<h2>排气MAP</h2>", unsafe_allow_html=True)
st.markdown(f"<div>{styled_df2.to_html()}</div>", unsafe_allow_html=True)
