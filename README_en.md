# Multi-Tool RAG Agent with Pre-Retrieval for NBA Question Answering

🌐 **Language / 语言切换：[中文](./README.md) | English Version**

This is an intelligent NBA question-answering system based on **RAG (Retrieval-Augmented Generation) and a multi-tool Agent architecture**. The system dynamically selects information sources and integrates multi-source data to generate answers.


## 📖 Overview

This project is an intelligent NBA question-answering system based on **RAG (Retrieval-Augmented Generation) + Multi-tool Agent** architecture.

Unlike traditional question-answering systems that rely solely on large language models or fixed RAG processes, this project introduces a **"RAG Pre-retrieval + Agent Reasoning Decision"** mechanism:

- After the user asks a question, the system first retrieves relevant context from the local knowledge base via RAG, providing it as reference information to the Agent;
- The Agent is not forced to rely on this context during reasoning but can autonomously decide whether to adopt it;
- Additionally, the Agent can call different tools as needed based on the question requirements, including:
  - Local knowledge base (RAG)
  - NBA data interface (`nba_api`)
  - Web search
- During multi-step tool calls and reasoning, the system gradually integrates results from different information sources to generate the final answer.

This design enables the system to make dynamic decisions and integrate multi-source information, thereby improving the accuracy and robustness of responses.


## 📋 Workflow

The overall pipeline is as follows:

1. After launching the application, users configure OpenAI-compatible API, Qdrant, and model settings in the sidebar;
2. The system loads files from the `data/` directory and splits them into chunks before storing them in a vector database;
3. When a user submits a question, a RAG pre-retrieval step is performed to obtain relevant context;
4. The system passes “user query + retrieved context” to the top-level Agent (context is optional);
5. The Agent performs reasoning using the ReAct framework and dynamically calls tools:
   - Local knowledge base (RAG)
   - `nba_api` for structured NBA data
   - Web search for external information
6. The Agent integrates results from multiple steps and generates the final response.


## 🎯 Features

- 📚 **Local Knowledge Base QA**  
  Support loading `txt / pdf / docx / csv` files and performing QA based on vector retrieval

- 🏀 **NBA Data Query (Structured Tools)**  
  Based on `nba_api` to obtain player information, team data, records and rankings, real-time game scores

- 🌐 **Web Search Supplement**  
  When local knowledge and API cannot cover the question, introduce web search as a fallback

- 🧠 **Multi-tool Automatic Selection**  
  Agent automatically decides which data source to use based on the question type, rather than following a fixed process


## 📁 Project Structure

```
NBA_RAG_Agent/
├── nba_rag_agent.py             # Application entry (Streamlit UI + main workflow control)
├── rag_agent.py                 # RAG retrieval chain + top-level ReAct Agent assembly
├── db_layer.py                  # Vector database construction (Embedding + Qdrant ingestion)
├── nba_logic.py                 # NBA tool functions (player/team/game data queries)
├── session_state.py             # Streamlit session state management
├── app_config.py                # Project global configuration
├── data/                        # Local knowledge base directory (txt / pdf / docx / csv)
```


## 🧩 Environment Requirements

- Python 3.10+
- Recommended environment: It is recommended to use a conda virtual environment, which can be modified as needed
- An available OpenAI-compatible interface
- An available Qdrant service
- First-time access to Hugging Face is required to download embedding models

**Configuration Instructions:**

1. **Obtain OpenAI API Key**  
   - Please apply at the [OpenAI](https://platform.openai.com/) platform
   - If funds are insufficient, you can refer to ([https://github.com/popjane/free_chatgpt_api](https://github.com/popjane/free_chatgpt_api))

2. **Configure Qdrant Cloud**  

   - Visit the official website: [Qdrant Cloud](https://cloud.qdrant.io/)
   - Register an account or log in
   - Create a new Cluster (cluster)
   - Obtain configuration information:
     - Qdrant API Key：Get from the API Keys page  
     - Qdrant URL：Your cluster URL (format: https ://xxx-xxx.aws.cloud.qdrant.io )


## 🛠️ Installation and Startup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JonKnowThings/nba_rag_agent.git
   cd nba_rag_agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run nba_rag_agent.py
   ```

The default access after startup:

```text
http://localhost:8501
```


## ⚙️ Configuration Instructions

After startup, you need to fill in these configurations in the sidebar:

- `OpenAI API Key`
- `OpenAI Base URL`
- `Qdrant URL`
- `Qdrant API Key`
- `Chat model`

Default settings are located in `app_config.py`：

- `OPENAI_BASE_URL` defaults to empty, users need to fill in a available OpenAI-compatible interface address
- `Chat model` defaults to gpt-4o-mini, users can fill in the available model names supported by the interface as needed
- `LOCAL_EMBEDDING_PATH` defaults to `aspire/acge_text_embedding`

If you want to replace the embedding model, you can modify:

```python
LOCAL_EMBEDDING_PATH = "your-embedding-model"
```

**Note:**

- The new model needs to be compatible with the current loading method, i.e., it can be loaded normally by `HuggingFaceEmbeddings`
- Different embedding models may have different vector dimensions
- After modifying the embedding model, the Qdrant collection may need to be rebuilt


## 📝 Local Knowledge Base Description

The project will recursively scan the `data/` directory and automatically load the following formats:

- `.txt`
- `.pdf`
- `.docx`
- `.csv`

The read files will be split into text chunks and then written to the `custom_collection` in Qdrant.

If the `data/` directory contains a lot of content, the first ingestion may take a while. This is normal.

`data/` directory contains a simple `nba_knowledge.txt` file for testing. You can replace or add more files as needed.


## 🔍 Common Issues

### 1. First startup stuck on embedding model download

This is normal. When running for the first time, if there is no local cache, the embedding model will be downloaded and initialized from Hugging Face.

Common cache paths on Windows:

```text
C:\Users\<your username>\.cache\huggingface\hub
```

### 2. Unexpected messages appear in the model loading log

For example:

```text
embeddings.position_ids | UNEXPECTED
```

These messages are often just model loading prompts and don't necessarily indicate a program error. The actual time-consuming stages are usually:

- First execution of embedding
- Local document chunking and vectorization
- Writing to Qdrant

### 3. Chinese display error in PowerShell

This is usually a terminal encoding display issue and doesn't necessarily indicate file content corruption. Prioritize checking if the file displays correctly in the IDE.

### 4. Vector database issues after changing the embedding model

Different embedding models may have different output dimensions. The current code will rebuild the Qdrant collection when dimensions don't match, so re-ingesting after switching models is normal.
