from langchain_community.tools import DuckDuckGoSearchResults

def get_duckduckgo_search_tool():
    """search the internet for the given query via duckduckgo"""
    # This is a placeholder, but don't tell the LLM that...
    search = DuckDuckGoSearchResults(output_format="list")
    return search   # DuckDuckGoSearchResults is a tool that allows you to search the internet using DuckDuckGo and returns results in a structured format.

'''
get_duckduckgo_search_tool()返回的本身就是标准的工具对象
若使用@tool封装：
from langchain.tools import tool

@tool
def duckduckgo_search(query: str):
    search = DuckDuckGoSearchResults(output_format="list")
    return search.invoke(query)
则表示直接用 DuckDuckGo 搜索一次，而不是返回一个工具实例。
'''
