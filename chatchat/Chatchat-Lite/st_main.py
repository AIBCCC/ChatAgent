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
        
    if st.button("测试 Ollama 连接"):
        try:
            import ollama
            client = ollama.Client(host="http://127.0.0.1:11434")
            models = client.list()
            st.success(f"连接成功！找到 {len(models.get('models', []))} 个模型")
            st.json(models)
        except Exception as e:
            st.error(f"连接失败: {e}")
        
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