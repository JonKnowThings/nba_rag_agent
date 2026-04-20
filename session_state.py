"""负责初始化 Streamlit 的会话状态。"""

import streamlit as st

from app_config import LOCAL_EMBEDDING_PATH, OPENAI_BASE_URL


def init_session_state() -> None:
    """为应用运行时依赖的全部状态字段补齐默认值。"""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "qdrant_url" not in st.session_state:
        st.session_state.qdrant_url = ""
    if "qdrant_api_key" not in st.session_state:
        st.session_state.qdrant_api_key = ""

    if "openai_base_url" not in st.session_state:
        st.session_state.openai_base_url = OPENAI_BASE_URL
    if "local_embedding_path" not in st.session_state:
        st.session_state.local_embedding_path = LOCAL_EMBEDDING_PATH
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = "gpt-4o-mini"
    if "qdrant_timeout" not in st.session_state:
        st.session_state.qdrant_timeout = 60
    if "qdrant_batch_size" not in st.session_state:
        st.session_state.qdrant_batch_size = 64

    # 这些对象会在运行过程中初始化，并缓存在当前会话里复用。
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "database" not in st.session_state:
        st.session_state.database = None

    # 避免本地知识文件在每次页面重跑时重复写入向量库。
    if "local_kb_loaded" not in st.session_state:
        st.session_state.local_kb_loaded = False
