import importlib.util
import streamlit as st
st.set_page_config(layout="wide", page_title="VVT模型展示程序")
# 初始化session state
if 'current_page' not in st.session_state:
    # 设置页面标题
    centered_title_html = """
    <div style="text-align: center; font-size: 3em; font-weight: bold; margin-top: 0px;">
        VVT模型回归预测程序
    </div>
    """
    st.markdown(centered_title_html, unsafe_allow_html=True)
    st.session_state.current_page = None
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image('1.gif')
    with col2:
        st.write("""
        欢迎使用VVT模型回归预测程序\n
        在这里，您可以通过选择不同的功能来执行数据处理、模型训练和预测分析。
        - 点击侧边栏中的按钮来选择您想要执行的操作。
        """)

# 使用beta_expander来组织侧边栏内容
with st.sidebar:
    st.markdown("### 导航菜单")

    # 数据处理部分
    with st.expander("数据处理"):
        group1_buttons = [
            ("数据提取和筛选", "数据提取和筛选.py"),
            ("人工选点", "人工选点.py")
        ]
        for button_text, script_path in group1_buttons:
            if st.button(button_text, key=f"button_{button_text.replace(' ', '_')}"):
                st.session_state.current_page = script_path

    # 模型回归部分
    with st.expander("模型回归"):
        group2_buttons = [
            ("AI模型误差分析", "训练误差分析.py"),
        ]
        for button_text, script_path in group2_buttons:
            if st.button(button_text, key=f"button_{button_text.replace(' ', '_')}"):
                st.session_state.current_page = script_path

    # 预测分析部分
    with st.expander("预测分析"):
        group3_buttons = [
            ("预测单一油耗", "预测单一油耗.py"),
            ("预测单一VVT", "预测单一VVT.py"),
            ("预测VVTMap", "预测VVTMap.py")
        ]
        for button_text, script_path in group3_buttons:
            if st.button(button_text, key=f"button_{button_text.replace(' ', '_')}"):
                st.session_state.current_page = script_path

# 处理按钮点击事件
if st.session_state.current_page:
    spec = importlib.util.spec_from_file_location(st.session_state.current_page.split('.')[0],
        st.session_state.current_page)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)