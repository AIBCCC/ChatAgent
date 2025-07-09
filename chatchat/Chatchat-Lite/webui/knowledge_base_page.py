import os

import streamlit as st
from utils import PLATFORMS, get_embedding_models, get_kb_names,get_embedding_model
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter



def knowledge_base_page():
    if "selected_kb" not in st.session_state:
        st.session_state["selected_kb"] = ""    # 当前选中的知识库名称，默认为空字符串
    st.title("知识库管理")
    kb_names = get_kb_names()
    selected_kb = st.selectbox("请选择知识库",
                               ["新建知识库"] + kb_names,
                               index=kb_names.index(st.session_state["selected_kb"]) + 1
                               if st.session_state["selected_kb"] in kb_names
                               else 0
                               )    # 选择知识库下拉框，包含新建知识库选项和已存在的知识库名称
    
    if selected_kb == "新建知识库":
        status_placeholder = st.empty()
        with status_placeholder.status("知识库配置", expanded=True) as s:
            cols = st.columns(2)
            kb_name = cols[0].text_input("请输入知识库名称", placeholder="请使用英文，如：companies_information")   # 输入知识库名称
            vs_type = cols[1].selectbox("请选择向量库类型", ["Chroma"])                                            # 选择向量库类型，目前仅支持 Chroma
            st.text_area("请输入知识库描述", placeholder="如：介绍企业基本信息")                                     # 输入知识库描述
            cols = st.columns(2)
            platform = cols[0].selectbox("请选择要使用的 Embedding 模型加载方式", PLATFORMS)                        # 选择平台
            embedding_models = get_embedding_models(platform)
            embedding_model = cols[1].selectbox("请选择要使用的 Embedding 模型", embedding_models)                  # 选择要使用的 Embedding 模型
            submit = st.button("创建知识库")
            #新建知识库文件夹
            if submit and kb_name.strip():
                kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
                kb_path = os.path.join(kb_root, kb_name)
                file_storage_path = os.path.join(kb_path, "files")
                vs_path = os.path.join(kb_path, "vectorstore")
                if not os.path.exists(kb_path):
                    os.mkdir(kb_path)
                if not os.path.exists(file_storage_path):
                    os.mkdir(file_storage_path)
                if not os.path.exists(vs_path):
                    os.mkdir(vs_path)
                else:
                    st.error("知识库已存在")
                    s.update(label=f'知识库配置', expanded=True, state="error")
                    st.stop()
                st.success("创建知识库成功")
                s.update(label=f'已创建知识库"{kb_name}"', expanded=False)
                st.session_state["selected_kb"] = kb_name
                st.rerun()
            elif submit and not kb_name.strip():
                st.error("知识库名称不能为空")
                s.update(label=f'知识库配置', expanded=True, state="error")
                st.stop()
    else:
        kb_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
        kb_path = os.path.join(kb_root, selected_kb)
        file_storage_path = os.path.join(kb_path, "files")
        vs_path = os.path.join(kb_path, "vectorstore")
        uploader_placeholder = st.empty()

        supported_file_formats = ["md"] # 支持的文件格式列表，当前仅支持 Markdown 文件
        with uploader_placeholder.status("上传文件至知识库", expanded=True) as s:
            files = st.file_uploader("请上传文件", type=supported_file_formats, accept_multiple_files=True)
            upload = st.button("上传")
        if upload:
            # 保存上传的文件到知识库文件夹
            for file in files:
                b = file.getvalue()
                with open(os.path.join(file_storage_path, file.name), "wb") as f:
                    f.write(b)

            from langchain_community.document_loaders import DirectoryLoader, TextLoader
            text_loader_kwargs = {"autodetect_encoding": True}
            # 创建目录加载器，加载上传的文件
            loader = DirectoryLoader(
                file_storage_path,
                glob=[f"**/{file.name}" for file in files],# 仅加载上传的文件
                show_progress=True,# 显示加载进度
                use_multithreading=True,# 使用多线程加载文件
                loader_cls=TextLoader,# 文本加载器
                loader_kwargs=text_loader_kwargs,# 文本加载器参数
            )
            # 加载文档
            docs_list = loader.load()
            # 文本分割器，分割加载的文档为小块
            text_splitter = MarkdownTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            doc_splits = text_splitter.split_documents(docs_list)
            for doc in doc_splits:
                doc.page_content = doc.metadata["source"] + "\n\n" + doc.page_content

            import chromadb.api
            # 清除 ChromaDB 系统缓存
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            # 创建 Chroma 向量存储实例
            vectorstore = Chroma(
                collection_name=selected_kb,
                embedding_function=get_embedding_model(model="quentinz/bge-large-zh-v1.5:latest"),
                persist_directory=vs_path,# 向量存储路径
            )

            vectorstore.add_documents(doc_splits)# 添加文档到向量存储
            st.success("上传文件成功")

'''
拼接前：
docs = [
    Document(
        page_content="小米科技有限公司总部位于北京市海淀区。",
        metadata={"source": "docs/mi.md"}
    ),
    Document(
        page_content="小米成立于2010年，由雷军创办。",
        metadata={"source": "docs/mi.md"}
    ),
    ...
]
'''

