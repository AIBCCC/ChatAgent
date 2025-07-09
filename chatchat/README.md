# Chatchat-lite

## 运行环境
Python >= 3.9
建议使用 3.10

可参考如下命令进行环境创建
```commandline
conda create -n chatchat-lite python=3.10 -y
conda activate chatchat-lite
```

## 安装依赖
```commandline
pip install -r requirements.txt
```

## 启动本地模型
当前项目仅支持接入 Ollama 模型
请前往 [Ollama官网](https://ollama.com/download) 下载最新版 Ollama， 安装完成后再命令行中执行以下命令：
```commandline
ollama run qwen2.5
ollama pull quentinz/bge-large-zh-v1.5
```

## 运行项目
使用以下命令行运行webui
```commandline
streamlit run st_main.py --theme.primaryColor "#165dff"
```
或使用暗色模式启动：
```commandline
streamlit run st_main.py --theme.base "dark" --theme.primaryColor "#165dff"
```

启动后界面如下：
- Agent 对话界面
    ![webui.png](img/webui.png)

- 模型配置界面
    ![webui2.png](img/webui2.png)