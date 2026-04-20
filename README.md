# Multi-Tool RAG Agent with Pre-Retrieval for NBA Question Answering

🌐 **Language / 语言切换：中文 | [English Version](./README_en.md)**

这是一个面向 NBA 问答场景并基于 RAG 与多工具Agent 的智能问答系统，能够根据问题动态选择数据来源并融合多源信息。


## 📖 项目概述

本项目是一个面向 NBA 场景的智能问答系统，基于 **RAG（检索增强生成）+ 多工具 Agent** 架构实现。

与传统仅依赖大模型或固定 RAG 流程的问答系统不同，本项目引入了 **“RAG 预检索 + Agent 推理决策”** 的机制：

- 在用户提问后，系统首先通过 RAG 从本地知识库中检索相关上下文，作为参考信息提供给 Agent；
- Agent 在推理过程中并不被强制依赖该上下文，而是可以自主判断是否采纳；
- 同时，Agent 可以根据问题需求，按需调用不同工具，包括：
  - 本地知识库（RAG）
  - NBA 数据接口（`nba_api`）
  - 网页搜索
- 在多轮工具调用与推理过程中，逐步整合来自不同信息源的结果，最终生成答案。

这种设计使系统不再依赖单一信息源，而是具备动态决策与多源信息融合的能力，从而提升回答的准确性与鲁棒性。


## 📋工作流程

整个问答流程如下：

1. 启动应用后，用户在页面侧边栏填写 OpenAI 兼容接口、Qdrant 和模型相关配置；
2. 程序读取 `data/` 目录中的本地文件，并将文本切块后写入向量数据库；
3. 用户输入问题后，系统首先执行一次 RAG 预检索，获取可能相关的上下文信息；
4. 将“用户问题 + 检索上下文”一并输入给顶层 Agent（上下文为可选参考，而非强制依据）；
5. Agent 基于 ReAct 模式进行推理，并按需调用工具：
   - 调用本地知识库（RAG）
   - 调用 `nba_api` 获取结构化数据
   - 调用网页搜索获取外部信息
6. Agent 在多步推理与工具调用过程中整合信息，最终生成回答。


## 🎯 功能概览

- 📚 **本地知识库问答**  
  支持加载 `txt / pdf / docx / csv` 文件，基于向量检索进行问答

- 🏀 **NBA 数据查询（结构化工具）**  
  基于 `nba_api` 获取球员信息、球队数据、战绩与排名、实时比赛比分情况

- 🌐 **网页搜索补充**  
  当本地知识与 API 无法覆盖问题时，引入网页搜索作为兜底

- 🧠 **多工具自动选择**  
  Agent 根据问题类型自动决定使用哪种数据源，而非固定流程


## 📁 项目结构

```
NBA_RAG_Agent/
├── nba_rag_agent.py             # 应用入口（Streamlit UI + 主流程控制）
├── rag_agent.py                 # RAG 检索链 + 顶层 ReAct Agent 组装
├── db_layer.py                  # 向量数据库构建（Embedding + Qdrant 入库）
├── nba_logic.py                 # NBA 工具函数（球员/球队/比赛数据查询）
├── session_state.py             # Streamlit 会话状态管理
├── app_config.py                # 项目全局配置
├── data/ # 本地知识库目录（txt / pdf / docx / csv）
```

## 🧩 环境要求

- Python 3.10+
- 推荐环境：建议使用 conda 虚拟环境，可自行修改
- 一个可用的 OpenAI 兼容接口
- 一个可用的 Qdrant 服务
- 首次运行时可访问 Hugging Face，用于下载 embedding 模型

**配置说明：**

1. **获取 OpenAI API Key**  
   - 请前往 [OpenAI](https://platform.openai.com/) 平台申请
   - 若资金不足，可以参考（[https://github.com/popjane/free_chatgpt_api](https://github.com/popjane/free_chatgpt_api)）

2. **配置 Qdrant Cloud**  

   - 访问官网：[Qdrant Cloud](https://cloud.qdrant.io/)
   - 注册账号或登录
   - 创建一个新的 Cluster（集群）
   - 获取配置信息：
     - Qdrant API Key：在 API Keys 页面获取  
     - Qdrant URL：您的集群 URL（格式：https ://xxx-xxx.aws.cloud.qdrant.io ）


## 🛠️ 安装与启动

1. **克隆仓库**:
   ```bash
  
   ```

2. **安装依赖项**:
   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用程序**:
   ```bash
   streamlit run nba_rag_agent.py
   ```

启动后默认访问：

```text
http://localhost:8501
```


## ⚙️ 配置说明

启动后需要在页面侧边栏填写这些配置：

- `OpenAI API Key`
- `OpenAI Base URL`
- `Qdrant URL`
- `Qdrant API Key`
- `Chat model`

当前项目中的相关默认配置位于 `app_config.py`：

- `OPENAI_BASE_URL` 默认留空，需要用户自行填写可用的 OpenAI 兼容接口地址
- `Chat model` 默认gpt-4o-mini，用户按需自行填写该接口支持的可用的模型名称
- `LOCAL_EMBEDDING_PATH` 当前默认是 `aspire/acge_text_embedding`

如果你想替换 embedding 模型，可以修改：

```python
LOCAL_EMBEDDING_PATH = "your-embedding-model"
```

**需要注意：**

- 新模型需要与当前加载方式兼容，即能够被 `HuggingFaceEmbeddings` 正常加载
- 不同 embedding 模型的向量维度可能不同
- 修改 embedding 模型后，Qdrant 集合可能需要重建


## 📝 本地知识库说明

项目会递归扫描 `data/` 目录，并自动加载以下格式：

- `.txt`
- `.pdf`
- `.docx`
- `.csv`

读取后的文件会被切分为文本块，再写入 Qdrant 中的 `custom_collection`。

如果 `data/` 目录内容较多，首次入库可能会比较慢。这是正常现象。


## 🔍 常见问题

### 1. 首次启动卡在 embedding 模型下载

这是正常现象。第一次运行时，如果本地没有缓存，embedding 模型会从 Hugging Face 下载并初始化。

Windows 常见缓存路径：

```text
C:\Users\<你的用户名>\.cache\huggingface\hub
```

### 2. 模型加载日志里出现 `UNEXPECTED`

例如：

```text
embeddings.position_ids | UNEXPECTED
```

这类信息很多时候只是模型加载提示，不一定表示程序报错。真正耗时的阶段通常是：

- 第一次执行 embedding
- 本地文档切块与向量化
- 向 Qdrant 写入文档

### 3. PowerShell 里中文显示乱码

这通常是终端编码显示问题，不一定是文件内容损坏。优先在 IDE 中确认文件是否正常显示。

### 4. 修改 embedding 模型后出现向量库问题

不同 embedding 模型的输出维度可能不同。当前代码在维度不一致时会重建 Qdrant 集合，因此切换模型后重新入库是正常现象。
