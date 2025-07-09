# ChatAgent
## 运行项目

在项目文件夹下运行以下命令启动WebUI

```
streamlit run st_main.py --theme.primaryColor "#165dff"

#可选启动参数
#--server.port	指定 Streamlit 应用运行的端口。
#--server.address	指定 Streamlit 应用的绑定地址。如果设置为 0.0.0.0，表示应用将绑定到所有可用的网络接口上，使其可以从外部访问。
#--server.runOnSave	 是否在每次保存文件时自动重新启动应用。
#--server.headless	指示 Streamlit 是否以无图形界面模式运行
```

## 效果展示

1. 对话模块

   用户可直接与本地大语言模型（如 Ollama、Xinference 等平台的模型）进行自然语言交互，支持多轮历史对话管理、温度调节、模型切换等功能。使用 `streaming` 流式输出方式，实现类 ChatGPT 的交互体验。

   ![chat](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/chat.png)

2. RAG对话模块

   RAG对话引入知识库增强回答能力。在用户提问后，系统先从所选知识库中检索相关文档片段，再将其连同问题一并发送给大模型生成更准确的回答。采用了 LangGraph 构建可流式处理的 Tool-augmented Agent，能实时显示调用知识库过程及结果。

   创建知识库，并上传知识库的md文件

   ![知识库管理](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/知识库管理.png)

   在RAG对话模块选择你要使用的数据库后，即可让模型基于知识库进行RAG对话。

   ![RAG对话](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/RAG对话.png)

3. Agent对话模块

   基于工具调用机制，构建可自主调用外部工具（如文献检索、网络搜索）的智能 Agent，对多步任务进行自主规划与执行。支持与大模型配合，通过 tool calling 接口实现自动决策与任务完成，是构建多工具组合 AI Agent 的原型场景。

   ![AgentChat](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/AgentChat.png)

   - 天气查询

     ![天气查询](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/天气查询.png)

   - Duckduckgo搜索

     

   - Arxiv搜索

     ![image-20250709170852312](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/Arxiv搜索.png)

   - Wikipedia搜索

   - 今日AI论文查询

     ![image-20250709171955109](https://github.com/AIBCCC/ChatAgent/blob/main/chatchat/img/今日AI论文查询.png)
