"""Streamlit 页面入口。"""

import streamlit as st
from langchain.schema import HumanMessage

from app_config import LOCAL_EMBEDDING_PATH
from db_layer import initialize_models, load_local_knowledge_docs
from rag_agent import create_master_agent, rag_tool_wrapper
from session_state import init_session_state


def main():
    """渲染页面并执行完整问答流程。"""
    init_session_state()
    st.set_page_config(page_title="NBA RAG Agent", page_icon="🏀")
    st.title("🏀🔥 NBA RAG Agent")

    with st.sidebar:
        # 侧边栏承担运行配置面板的角色，所有外部依赖都从这里输入。
        st.header("Configuration")
        api_key = st.text_input("Enter OpenAI API Key:", type="password", value=st.session_state.openai_api_key, key="api_key_input")
        base_url = st.text_input("OpenAI Base URL:", value=st.session_state.openai_base_url)
        qdrant_url = st.text_input("Enter Qdrant URL:", value=st.session_state.qdrant_url, help="Example: https://your-cluster.qdrant.tech")
        qdrant_api_key = st.text_input("Enter Qdrant API Key:", type="password", value=st.session_state.qdrant_api_key)
        qdrant_timeout = st.number_input("Qdrant timeout (seconds):", min_value=10, max_value=300, value=int(st.session_state.qdrant_timeout), step=10)
        qdrant_batch_size = st.number_input("Qdrant batch size:", min_value=8, max_value=512, value=int(st.session_state.qdrant_batch_size), step=8)
        st.caption(f"Embedding model (fixed): `{LOCAL_EMBEDDING_PATH}`")
        chat_model = st.text_input("Chat model (API):", value=st.session_state.chat_model, help="Use a model supported by your API gateway (e.g. gpt-4o-mini).")

        if api_key:
            st.session_state.openai_api_key = api_key
        if base_url:
            st.session_state.openai_base_url = base_url
        if qdrant_url:
            st.session_state.qdrant_url = qdrant_url
        if qdrant_api_key:
            st.session_state.qdrant_api_key = qdrant_api_key
        st.session_state.qdrant_timeout = int(qdrant_timeout)
        st.session_state.qdrant_batch_size = int(qdrant_batch_size)
        st.session_state.local_embedding_path = LOCAL_EMBEDDING_PATH
        if chat_model:
            st.session_state.chat_model = chat_model

        if st.session_state.openai_api_key and st.session_state.qdrant_url and st.session_state.qdrant_api_key:
            if initialize_models():
                st.success("Connected to OpenAI and Qdrant successfully!")

                # 本地知识库只在当前 session 首次成功初始化时写入一次，
                # 避免 Streamlit 每次重跑都重复 add_documents。
                if not st.session_state.local_kb_loaded:
                    local_docs = load_local_knowledge_docs("data")
                    if local_docs and st.session_state.database is not None:
                        st.session_state.database.add_documents(local_docs, batch_size=st.session_state.qdrant_batch_size)
                        st.info(f"Loaded {len(local_docs)} local chunks into Custom Knowledge Base.")
                    st.session_state.local_kb_loaded = True
            else:
                st.error("Failed to initialize. Please check your credentials.")
        else:
            st.warning("Please enter all required credentials to continue")
            st.stop()
        st.markdown("---")

    st.header("Ask Questions")
    st.info("Enter your question below to find answers from the local knowledge base or NBA API.")
    question = st.text_input("Enter your question:")

    if question:
        with st.spinner("Thinking..."):
            rag_prefetch = None

            try:
                # 先尝试从内部知识库取一轮上下文，后续主 agent 可以把它当作补充证据使用。
                rag_result = rag_tool_wrapper(question)
                if "No relevant documents found" not in rag_result:
                    rag_prefetch = rag_result
            except Exception:
                rag_prefetch = None

            if rag_prefetch:
                enhanced_question = f"""
User Question:
{question}

The following contextual knowledge was retrieved from the internal knowledge base.
You may use it if helpful, but you are NOT required to trust it.
You may call other tools (NBA API, Web Search, etc.) if necessary.

[Retrieved Context]
{rag_prefetch}
"""
            else:
                enhanced_question = question

            master_agent = create_master_agent()
            agent_input = {
                "messages": [HumanMessage(content=enhanced_question)],
                "is_last_step": False,
            }

            try:
                # LangGraph / ReAct agent 的返回结构不完全固定，因此这里统一做一次文本提取。
                response = master_agent.invoke(
                    agent_input,
                    config={"recursion_limit": 50},
                )

                if hasattr(response, "content"):
                    answer = response.content
                elif isinstance(response, dict):
                    last_message = ""
                    messages = response.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                    if hasattr(last_message, "content"):
                        answer = last_message.content
                    elif isinstance(last_message, dict):
                        answer = last_message.get("content", "")
                    else:
                        answer = str(last_message)
                else:
                    answer = str(response)
            except Exception as e:
                answer = f"Error during Agent reasoning: {e}"

            st.write("### Answer")
            st.write(answer)


if __name__ == "__main__":
    main()
