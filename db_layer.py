"""模型初始化与本地知识入库逻辑。"""

import os
from typing import List, Optional

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app_config import CUSTOM_COLLECTION_NAME, LOCAL_EMBEDDING_PATH


def initialize_models() -> bool:
    """初始化 embedding、LLM 与单个 custom Qdrant 集合。"""
    if not (
        st.session_state.openai_api_key
        and st.session_state.qdrant_url
        and st.session_state.qdrant_api_key
    ):
        return False

    # 部分下游库会从环境变量读取配置，因此这里同步一份。
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    os.environ["OPENAI_BASE_URL"] = st.session_state.openai_base_url

    # 第一次启动时，如果本地没有缓存，embedding 模型可能会先从 Hugging Face 下载。
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    st.session_state.llm = ChatOpenAI(
        temperature=0,
        base_url=st.session_state.openai_base_url,
        model=st.session_state.chat_model,
    )

    def _get_collection_vector_size(collection_info) -> Optional[int]:
        """从已有 Qdrant 集合配置中提取向量维度。"""
        try:
            vectors = collection_info.config.params.vectors
            if isinstance(vectors, dict):
                first_key = next(iter(vectors))
                return int(vectors[first_key].size)
            return int(vectors.size)
        except Exception:
            return None

    def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
        """确保 custom 集合存在，且向量维度与当前 embedding 模型一致。"""
        try:
            collection_info = client.get_collection(collection_name)
            existing_size = _get_collection_vector_size(collection_info)
            if existing_size is None or existing_size != vector_size:
                client.delete_collection(collection_name=collection_name)
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        except Exception:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    try:
        client = QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=st.session_state.qdrant_timeout,
        )
        client.get_collections()

        # 先探测一次 embedding 维度，再创建或修正集合结构。
        vector_size = len(st.session_state.embeddings.embed_query("vector size probe"))
        _ensure_collection(client, CUSTOM_COLLECTION_NAME, vector_size)
        st.session_state.database = Qdrant(
            client=client,
            collection_name=CUSTOM_COLLECTION_NAME,
            embeddings=st.session_state.embeddings,
        )
        return True
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {str(e)}")
        return False


def load_local_knowledge_docs(path: str = "data") -> List[Document]:
    """读取本地知识文件，并切分成适合向量检索的小文本块。"""
    documents: List[Document] = []
    try:
        if not os.path.exists(path):
            return []

        # 递归扫描整个目录，允许在 data 下继续分子目录整理资料。
        for root, _, files in os.walk(path):
            for filename in files:
                file_path = os.path.join(root, filename)
                lower_name = filename.lower()
                if lower_name.endswith(".txt"):
                    documents.extend(TextLoader(file_path, encoding="utf-8").load())
                elif lower_name.endswith(".pdf"):
                    documents.extend(PyPDFLoader(file_path).load())
                elif lower_name.endswith(".docx"):
                    documents.extend(Docx2txtLoader(file_path).load())
                elif lower_name.endswith(".csv"):
                    documents.extend(CSVLoader(file_path).load())

        if not documents:
            return []

        # 大文件直接整段向量化效果较差，切块后更适合做语义检索。
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error loading local knowledge base: {e}")
        return []
