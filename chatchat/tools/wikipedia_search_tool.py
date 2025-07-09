from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


@tool
def wikipedia_search_tool(query: str):
    """Searches arxiv.org Articles for the query and returns the articles summaries.
    Args:
        query: The query to search for, should be in English."""
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

if __name__ == "__main__":
    print(wikipedia_search_tool.invoke("Alan Turing"))