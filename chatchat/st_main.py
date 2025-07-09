import streamlit as st
from webui import chat_page, rag_chat_page, agent_chat_page, knowledge_base_page #, platforms_page
from utils import get_img_base64

if __name__ == "__main__":
    with st.sidebar:
        st.logo(
            get_img_base64("chatchat_lite_logo.png"),
            size="large",
            icon_image=get_img_base64("chatchat_lite_small_logo.png"),
        )
        
    pg = st.navigation({
        "对话": [
            st.Page(chat_page, title="对话", icon=":material/chat_bubble:"),
            st.Page(rag_chat_page, title="RAG 对话", icon=":material/chat:"),
            st.Page(agent_chat_page, title="Agent 对话", icon=":material/chat_add_on:"),
        ],
        "设置": [
            st.Page(knowledge_base_page, title="知识库管理", icon=":material/library_books:"),
            # st.Page(platforms_page, title="模型平台管理", icon=":material/settings:"),
        ]
    })
    pg.run()
