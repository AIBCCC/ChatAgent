import streamlit as st
from utils import PLATFORMS, get_llm_models, get_chatllm, get_kb_names, get_img_base64
from langchain_core.messages import AIMessageChunk, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tools import get_naive_rag_tool
import json

RAG_PAGE_INTRODUCTION = "你好，我是你的 Chatchat 智能助手，当前页面为`RAG 对话模式`，可以在对话让大模型基于左侧所选知识库进行回答，有什么可以帮助你的吗？"

#构建一个 RAG 工作流图，并返回可执行的 Agent 应用实例
def get_rag_graph(platform, model, temperature, selected_kbs, KBS):
    tools = [KBS[k] for k in selected_kbs]
    tool_node = ToolNode(tools)
    
    def call_model(state):
        llm = get_chatllm(platform, model, temperature=temperature)
        llm_with_tools = llm.bind_tools(tools, tool_choice="any")       #绑定RAG 工具到 LLM，让模型具备调用这些知识工具的能力
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 创建一个状态流图，节点之间表示不同处理环节，MessagesState 是图的共享状态数据结构，保存当前对话历史。
    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", call_model)  # 模型推理节点
    workflow.add_node("tools", tool_node)   # RAG 工具节点

    workflow.add_conditional_edges("agent", tools_condition)    # 根据工具调用条件，决定是否调用 RAG 工具
    workflow.add_edge("tools", "agent")     # 工具调用完成后返回到模型推理节点
    workflow.set_entry_point("agent")       # 设置入口节点为模型推理节点

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)       # 编译工作流图，生成可执行的 Agent 应用实例
    return app

def graph_response(graph, input):
    '''
    1.接收输入对话 messages，发送给 LangGraph 工作流
    2.监听 stream 输出的每一个事件 event：
    '''
    # 使用 graph.stream() 方法流式获取模型响应
    for event in graph.stream(             # event 是一个元组，包含了当前状态和消息内容， event[0] 是当前状态，event[1] 是消息内容
        {"messages": input},
        config={"configurable": {"thread_id": 42}}, # 配置线程 ID（"thread_id": 42可随时调整），便于跟踪对话状态
        stream_mode="messages",
    ):
        '''
        - 如果是 AI 回答（AIMessageChunk）：
            a. 如果包含工具调用，记录调用信息
            b. 否则 yield 输出回答文本
        '''
        #检查 event[0] 的类型，如果是 AIMessageChunk，则表示模型生成的内容（模型输出）
        if type(event[0]) == AIMessageChunk:
            if len(event[0].tool_calls):
                # st.write(event[0].tool_calls)
                st.session_state["rag_tool_calls"].append(
                    {
                        "status": "正在查询...",
                        "knowledge_base": event[0].tool_calls[0]["name"].replace("_knowledge_base_tool", ""),
                        "query": str(event[0].tool_calls[0]["args"]["query"]),
                    }
                )
            yield event[0].content

            '''
        - 如果是工具响应（ToolMessage）：
            a. 显示查询中状态
            b. 显示知识库检索内容与结果
            c. 更新 session_state 中的调用记录
            '''
        # 如果是 ToolMessage，则表示调用了 RAG 工具（工具调用结果返回）
        elif type(event[0]) == ToolMessage:
            status_placeholder = st.empty()
            with (status_placeholder.status("正在查询...", expanded=True) as s):
                st.write("已调用 `", event[0].name.replace("_knowledge_base_tool", ""), "` 知识库进行查询")  #展示调用哪个知识库
                continue_save = False
                if len(st.session_state["rag_tool_calls"]):
                    if "content" not in st.session_state["rag_tool_calls"][-1].keys() \
                            and event[0].name.replace("_knowledge_base_tool", "") == \
                        st.session_state["rag_tool_calls"][-1]["knowledge_base"]:
                        continue_save = True
                        st.write("知识库检索输入: ")
                        st.code(st.session_state["rag_tool_calls"][-1]["query"],
                                wrap_lines=True)  # Display the input data sent to the tool
                st.write("知识库检索结果：")
                for k, content in json.loads(event[0].content).items():
                    st.write(f"- {k}:")
                    st.code(content, wrap_lines=True) # Placeholder for tool output that will be updated later below
                s.update(label="已完成知识库检索！", expanded=False)
                if continue_save:
                    st.session_state["rag_tool_calls"][-1]["status"] = "已完成知识库检索！"
                    st.session_state["rag_tool_calls"][-1]["content"] = json.loads(event[0].content)
                else:
                    st.session_state["rag_tool_calls"].append(
                        {
                            "status": "已完成知识库检索！",
                            "knowledge_base": event[0].name.replace("_knowledge_base_tool", ""),
                            "content": json.loads(event[0].content)
                        })

# 获取 RAG 对话响应
'''
 1.获取 RAG 工作流图
 2.调用 graph_response() 方法，传入工作流图和输入对话
 3.返回模型生成的回答 
''' 
def get_rag_chat_response(platform, model, temperature, input, selected_tools, KBS):
    app = get_rag_graph(platform, model, temperature, selected_tools, KBS)
    return graph_response(graph=app, input=input)

'''
message结构：
{
    "role": "assistant" or "user",
    "content": "message content",
    "tool_calls": [
        {
            "status": "正在查询..." or "已完成知识库检索！",
            "knowledge_base": "knowledge_base_name",（百科）
            "query": "query input",（查询输入）
            "content": { "key1": "content1", "key2": "content2" }（查询结果）
        }
    ]       
}
'''
def display_chat_history():
    for message in st.session_state["rag_chat_history_with_tool_call"]:
        with st.chat_message(message["role"], avatar=get_img_base64("chatchat_avatar.png") if message["role"] == "assistant" else None):
            if "tool_calls" in message.keys():
                for tool_call in message["tool_calls"]:
                    with st.status(tool_call["status"], expanded=False):
                        st.write("已调用 `", tool_call["knowledge_base"], "` 知识库进行查询")
                        if "query" in tool_call.keys():
                            st.write("知识库检索输入: ")
                            st.code(tool_call["query"], wrap_lines=True)  # Display the input data sent to the tool
                        st.write("知识库检索结果：")
                        for k, content in tool_call["content"].items():
                            st.write(f"- {k}:")
                            st.code(content,wrap_lines=True)  # Placeholder for tool output that will be updated later below

            st.write(message["content"])

def clear_chat_history():
    st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    st.session_state["rag_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    st.session_state["rag_tool_calls"] = []


def rag_chat_page():
    kbs = get_kb_names()
    KBS = dict()        # 用于存储知识库工具的字典，key 为知识库名称，value 为该知识库对应的 RAG 工具对象
    for k in kbs:
        KBS[f"{k}"] = get_naive_rag_tool(k)

    #rag_chat_history 是最基础的对话记录列表，仅保存用户和 AI 间的文本消息
    if "rag_chat_history" not in st.session_state:
        st.session_state["rag_chat_history"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    #rag_chat_history_with_tool_call 是包含工具调用信息的对话记录列表，保存用户、AI 和工具调用间的消息
    if "rag_chat_history_with_tool_call" not in st.session_state:
        st.session_state["rag_chat_history_with_tool_call"] = [
            {"role": "assistant", "content": RAG_PAGE_INTRODUCTION}
        ]
    #rag_tool_calls 用于存储当前对话中所有的工具调用记录
    if "rag_tool_calls" not in st.session_state:
        st.session_state["rag_tool_calls"] = []

    #从侧边栏选择知识库
    with st.sidebar:
        selected_kbs = st.multiselect("请选择对话中可使用的知识库", kbs, default=kbs)

    display_chat_history()

    with st._bottom:
        cols = st.columns([1.2, 10, 1])
        with cols[0].popover(":gear:", use_container_width=True, help="配置模型"):
            platform = st.selectbox("请选择要使用的模型加载方式", PLATFORMS)
            model = st.selectbox("请选择要使用的模型", get_llm_models(platform))
            temperature = st.slider("请选择模型 Temperature", 0.1, 1., 0.1)
            history_len = st.slider("请选择历史消息长度", 1, 10, 5)
        input = cols[1].chat_input("请输入您的问题")
        cols[2].button(":wastebasket:", help="清空对话", on_click=clear_chat_history)
    
    if input:
        #展示用户输入
        with st.chat_message("user"):
            st.write(input)
        #将用户输入添加到对话历史中
        st.session_state["rag_chat_history"] += [{"role": 'user', "content": input}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'user', "content": input}]

        stream_response = get_rag_chat_response(
            platform,
            model,
            temperature,
            st.session_state["rag_chat_history"][-history_len:],
            selected_kbs,
            KBS
        )

        #展示 AI 响应
        with st.chat_message("assistant", avatar=get_img_base64("chatchat_avatar.png")):
            response = st.write_stream(stream_response)
        st.session_state["rag_chat_history"] += [{"role": 'assistant', "content": response}]
        st.session_state["rag_chat_history_with_tool_call"] += [{"role": 'assistant', "content": response, "tool_calls": st.session_state["rag_tool_calls"]}]
        st.session_state["rag_tool_calls"] = []     # 清空工具调用记录，确保下一轮对话的工具调用状态不会混入前一轮内容。

'''
st.session_state = {
    # 对话页面相关
    "chat_history": [
        {"role": "user" | "assistant", "content": str}
    ],

    # RAG页面普通对话记录（只记录角色与内容）
    "rag_chat_history": [
        {"role": "user" | "assistant", "content": str}
    ],

    # RAG页面包含 tool 调用的完整记录（用于展示）
    "rag_chat_history_with_tool_call": [
        {
            "role": "user" | "assistant",
            "content": str,
            "tool_calls": [  # assistant 时可选
                {
                    "status": str,
                    "knowledge_base": str,
                    "query": str,
                    "content": dict[str, str]  # 检索结果
                }
            ]
        }
    ],

    # 当前这轮中 AI 生成期间触发的 tool_calls
    "rag_tool_calls": [
        {
            "status": "正在查询..." | "已完成知识库检索！",
            "knowledge_base": str,
            "query": str (可选),
            "content": dict[str, str] (可选)
        }
    ],

    # 其他可选参数（可扩展）
    # "selected_platform": "Ollama" / "OpenAI" 等
    # "selected_model": "qwen:7b" / "gpt-4" 等
}

1.每次对话输入后：
chat_history / rag_chat_history 会记录 "user" 和 "assistant" 两轮内容

2.如果是 RAG 页面，还会更新：
rag_tool_calls：中间过程收集 AI 的 tool 调用内容
rag_chat_history_with_tool_call：记录完整 tool 调用结果

3.每轮对话结束后：
rag_tool_calls 会被清空，防止下轮混入旧的调用信息

实例：
{
    "role": "assistant",
    "content": "这是从知识库中获取的信息...",
    "tool_calls": [
        {
            "status": "已完成知识库检索！",
            "knowledge_base": "chatgpt_docs",
            "query": "RAG 是什么？",
            "content": {
                "doc1": "RAG 是检索增强生成（Retrieval-Augmented Generation）方法...",
                "doc2": "用于结合外部知识进行问答..."
            }
        }
    ]
}

'''