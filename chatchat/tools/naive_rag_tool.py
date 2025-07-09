from langchain.tools.retriever import create_retriever_tool
import os

from utils import get_embedding_model


def get_naive_rag_tool(vectorstore_name):
    from langchain_chroma import Chroma

    #加载向量存储
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=get_embedding_model(platform_type="Ollama", model="quentinz/bge-large-zh-v1.5:latest"),
        persist_directory=os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb", vectorstore_name, "vectorstore"),
    )
    # 创建检索器，使用相似度分数阈值
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",   # 使用相似度分数阈值检索
        search_kwargs={
            "k": 10,
            "score_threshold": 0.15,
        }
    )
    # 创建检索工具，包装成一个可绑定到 Agent 的工具节点
    retriever_tool = create_retriever_tool(
        retriever,
        f"{vectorstore_name}_knowledge_base_tool",      # 工具名称
        f"search and return information about {vectorstore_name}",      # 工具描述
    )
    #设定返回格式，只返回内容部分
    retriever_tool.response_format = "content"
    #自定义结构化输出
    retriever_tool.func = lambda query: {
        f"已知内容 {inum+1}": doc.page_content.replace(doc.metadata["source"] + "\n\n", "")
        for inum, doc in enumerate(retriever.invoke(query))
    }
    return retriever_tool   #返回一个可调用的工具对象，可invoke


if __name__ == "__main__":
    retriever_tool = get_naive_rag_tool("personal_information")
    print(retriever_tool.invoke("刘虔"))  #.invoke(qurey) 方法用于调用工具，传入查询字符串

'''
存储形式：
{
    "id": "uuid",  # 唯一ID
    "embedding": [0.123, 0.324, ..., 0.984],  # 向量表示（模型生成）
    "document": {
        "page_content": "路径\n\n文本内容",   # 加入了 source 信息
        "metadata": {
            "source": "/absolute/path/to/resume.md"
        }
    }
}

'''