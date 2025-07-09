import os
from typing import Literal,get_args
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout
import base64
from io import BytesIO
import streamlit as st


PlatformType = Literal["Ollama", "Xinference", "OpenAI", "ZhipuAI"]

# 自动生成选项列表
PLATFORMS = list(get_args(PlatformType)) # ["fastchat"] "ZhipuAI",



def get_llm_models(platform_type: PlatformType, base_url: str="", api_key: str="EMPTY"): #platform_type平台类型，base_url可选服务地址， api_key可选API密钥
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url:
                base_url = "http://127.0.0.1:11434"         # 默认 Ollama 服务地址
            client = ollama.Client(host=base_url)           # 创建 Ollama 客户端
            llm_models = [model["model"] for model in client.list()["models"] if "bert" not in model.details.families]  # 获取所有 LLM 模型，排除包含 "bert" 的模型（只获取生成式模型）
            return llm_models
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)          # 创建 Xinference 客户端
            llm_models = client.list_models()
            return [k for k,v in llm_models.items() if v.get("model_type") == "LLM"]        #获取所有模型（键为模型名，值为模型详情 dict）,并且只返回模型类型为 "LLM" 的条目（排除 Embedding/Embedding-only）
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 LLM 模型时发生错误：\n{e}")
            return []
    
    #敬请期待
    elif platform_type == "ZhipuAI":
        # from zhipuai import ZhipuAI
        #
        # client = ZhipuAI(
        #     api_key="",  # 填写您的 APIKey
        # )
        # llm_models=zhipuai.list_models()
        # return [model.id for model in llm_models.data]
        return [
            'glm-4-alltools',
            'glm-4-plus',
            'glm-4-0520',
            'glm-4',
            'glm-4-air',
            'glm-4-airx',
            'glm-4-long',
            'glm-4-flashx',
            'glm-4-flash'
        ]
    elif platform_type == "OpenAI":
        # from OpenAI import OpenAI
        #
        # client = OpenAI(
        #     api_key="",  # 填写您的 APIKey
        # )
        # llm_models=openai.Model.list()
        # return [model.id for model in llm_models.data]
        return [
            'gpt-4o-mini',
            'gpt-3.5-turbo'
        ]


def get_embedding_models(platform_type:PlatformType, base_url: str="", api_key: str="EMPTY"):
    if platform_type == "Ollama":
        try:
            import ollama
            if not base_url:
                base_url = "http://127.0.0.1:11434"
            client = ollama.Client(host=base_url)
            embedding_models = [model["model"] for model in client.list()["models"] if "bert" in model.details.families]            # 获取所有 Embedding 模型，包含 "bert" 的模型
            return embedding_models
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []
    elif platform_type == "Xinference":
        try:
            from xinference_client import RESTfulClient as Client
            if not base_url:
                base_url = "http://127.0.0.1:9997"
            client = Client(base_url=base_url)
            embedding_models = client.list_models()
            return [k for k,v in embedding_models.items() if v.get("model_type") == "embedding"]               # 获取所有模型（键为模型名，值为模型详情 dict）,并且只返回模型类型为 "embedding" 的条目
        except Exception as e:
            st.toast(f"尝试连接 {platform_type} 获取 Embedding 模型时发生错误：\n{e}")
            return []


def get_chatllm(
        platform_type: PlatformType,
        model: str,
        base_url: str = "",
        api_key: str = "",
        temperature: float = 0.1
):
    if platform_type == "Ollama":
        if not base_url:
            base_url = "http://127.0.0.1:11434"
        return ChatOllama(              # ChatOllama 是 Ollama 的聊天模型接口,实例化本地模型
            temperature=temperature,
            streaming=True,
            model=model,
            base_url=base_url
        )
    
    elif platform_type == "Xinference":
        from langchain_community.chat_models import ChatXinference
        if not model:
            model = "chatglm-6b"
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        if not api_key:
            api_key = "EMPTY"
        return ChatXinference(
            temperature=temperature,
            model_uid=model,
            streaming=True,
            server_url=base_url,
            api_key=api_key,
        )

    elif platform_type == "ZhipuAI":
        from langchain_community.chat_models import ChatZhipuAI
        if not base_url:
            base_url = "https://open.bigmodel.cn/api/paas/v4"
        if not api_key:
            api_key = "EMPTY"
        return ChatZhipuAI(
            temperature=temperature,
            model=model,
            streaming=True,
            base_url=base_url,
            api_key=api_key,
        )
    
    elif platform_type == "OpenAI":
        if not base_url:
            base_url = "https://api.openai.com/v1"
        if not api_key:
            api_key = "EMPTY"
        return ChatOpenAI(
            temperature=temperature,
            model_name=model,
            streaming=True,
            base_url=base_url,
            api_key=api_key,
        )


def show_graph(graph):
    flow_state = StreamlitFlowState(
                       nodes=[StreamlitFlowNode(
                           id=node.id,      # 节点 ID
                           pos=(0,0),       # 节点位置，(0,0) 是默认位置
                           data={"content": node.id},       # 节点数据，这里使用节点 ID 作为内容
                           node_type="input" if node.id == "__start__"
                                             else "output" if node.id == "__end__"
                                             else "default",
                       ) for node in graph.nodes.values()],     # 节点列表，使用 StreamlitFlowNode 创建节点对象
                       edges=[StreamlitFlowEdge(
                           id=str(enum),        # 边的 ID，使用枚举值作为 ID
                           source=edge.source,      # 边的源节点 ID
                           target=edge.target,      # 边的目标节点 ID
                           animated=True,           # 是否动画化边
                       ) for enum, edge in enumerate(graph.edges)],     # 边列表，使用 StreamlitFlowEdge 创建边对象
                   )
    streamlit_flow('example_flow',
                   flow_state,
                   layout=TreeLayout(direction='down'),     # 布局方式,树形布局，自上而下
                   fit_view=True
    )


def get_kb_names():
    kb_root = os.path.join(os.path.dirname(__file__), "kb")         # 获取kb文件夹的绝对路径,知识库根目录
    if not os.path.exists(kb_root):
        os.mkdir(kb_root)
    kb_names = [f for f in os.listdir(kb_root) if os.path.isdir(os.path.join(kb_root, f))]      # 获取所有子文件夹的名称,获取知识库名称列表
    return kb_names

#根据平台动态加载 Embedding 模型，供后续文本向量化使用，适配性强，支持本地+云服务。
def get_embedding_model(
        platform_type: PlatformType = "Ollama",
        model: str = "quentinz/bge-large-zh-v1.5",
        base_url: str = "",
        api_key: str = "EMPTY",
):
    if platform_type == "Ollama":
        # from langchain_ollama import ChatOllama
        # return ChatOllama
        if not base_url:
            base_url = "http://127.0.0.1:11434/"
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(base_url=base_url, model=model)
    elif platform_type == "Xinference":
        from langchain_community.embeddings.xinference import XinferenceEmbeddings
        if not base_url:
            base_url = "http://127.0.0.1:9997/v1"
        return XinferenceEmbeddings(server_url=base_url, model_uid=model)
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(base_url=base_url, api_key=api_key, model=model)

#在 Streamlit 中加载本地图片， HTML <img> 标签形式内嵌展示。
def get_img_base64(file_name: str) -> str:
    """
    get_img_base64 used in streamlit.
    absolute local path not working on windows.
    """
    image_path = os.path.join(os.path.dirname(__file__), "img", file_name)
    # 读取图片
    with open(image_path, "rb") as f:
        buffer = BytesIO(f.read())      # 使用 BytesIO 将图片内容读入内存缓冲区
        base_str = base64.b64encode(buffer.getvalue()).decode("utf-8")      # 将图片内容编码为 base64 字符串
    return f"data:image/png;base64,{base_str}"

if __name__ == "__main__":
    print(get_embedding_model(platform_type="Ollama"))
