import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_img_base64

CHAT_PAGE_INTRODUCTION = "你好，我是你的 Chatchat 智能助手，当前页面为`对话模式`，可以直接与大模型对话，有什么可以帮助你的吗？"

#加载聊天模型
def get_chat_response(platform, model, temperature, input):
    llm = get_chatllm(platform, model, temperature=temperature)
    for chunk in llm.stream(input):     # 使用流式响应，逐步获取模型生成的内容
        yield chunk.content

# 显示聊天历史记录
def display_chat_history():
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"], avatar=get_img_base64("chatchat_avatar.png") if message["role"] == "assistant" else None):
            st.write(message["content"])

# 清空聊天历史记录
def clear_chat_history():
    st.session_state["chat_history"] = [
            {"role": "assistant", "content": CHAT_PAGE_INTRODUCTION}
        ]


def chat_page():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": CHAT_PAGE_INTRODUCTION}
        ]

    with st.sidebar:
        st.title("Chatbot")


    display_chat_history()

    #页面底部输入栏
    with st._bottom:
        cols = st.columns([1.2, 10, 1])     # 设置列宽比例，第一列为配置模型，第二列为输入框，第三列为清空按钮
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            llm_models = get_llm_models(platform)
            model = st.selectbox("请选择要使用的模型", llm_models)
            temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    
    # 如果用户输入了内容，则发送消息并获取模型响应
    if input:
        with st.chat_message("user"):
            st.write(input)
        st.session_state["chat_history"] += [{'role': 'user', 'content': input}]

        #生成回复
        stream_response = get_chat_response(
            platform,
            model,
            temperature,
            st.session_state["chat_history"][-history_len:]
        )

        with st.chat_message("assistant", avatar=get_img_base64("chatchat_avatar.png")):
            response = st.write_stream(stream_response)
        st.session_state["chat_history"] += [{'role': 'assistant', 'content': response}]