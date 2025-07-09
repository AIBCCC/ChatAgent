from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.tools import tool

@tool       #将普通函数包装成一个可以被Agent识别并调用的tool
def arxiv_search_tool(query: str):
    """Searches arxiv.org Articles for the query and returns the articles summaries.
    Args:
        query: The query to search for, should be in English."""
    tool = ArxivAPIWrapper()        # ArxivAPIWrapper 是 LangChain 提供的一个工具，用于搜索 arxiv.org 上的文章
    return tool.run(query)          #返回论文摘要

if __name__ == "__main__":
    print(arxiv_search_tool.invoke("Apple Intelligence"))