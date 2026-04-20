"""RAG 检索链与总控 agent 的组装逻辑。"""

import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Qdrant
from langgraph.prebuilt import create_react_agent

from nba_logic import nba_stats_tool


def query_database(db: Qdrant, question: str) -> tuple[str, list]:
    """在指定向量库中检索相关文档，并基于检索结果生成回答。"""
    try:
        # 先取与问题最相近的几个文本块，再交给 LLM 做基于上下文的回答。
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        relevant_docs = retriever.get_relevant_documents(question)
        if relevant_docs:
            # 这个 prompt 明确限制模型尽量只依据检索到的上下文回答，
            # 目的是减少“脑补”内容。
            retrieval_qa_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a helpful AI assistant that answers questions based on provided context.
                             Always be direct and concise in your responses.
                             If the context doesn't contain enough information to fully answer the question, acknowledge this limitation.
                             Base your answers strictly on the provided context and avoid making assumptions.""",
                    ),
                    ("human", "Here is the context:\n{context}"),
                    ("human", "Question: {input}"),
                    ("assistant", "I'll help answer your question based on the context provided."),
                    ("human", "Please provide your answer:"),
                ]
            )
            combine_docs_chain = create_stuff_documents_chain(st.session_state.llm, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            response = retrieval_chain.invoke({"input": question})
            return response["answer"], relevant_docs
        raise ValueError("No relevant documents found in database")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return "I encountered an error. Please try rephrasing your question.", []


def rag_tool_wrapper(question: str) -> str:
    """仅使用 `custom` 知识库执行检索问答。"""
    db = st.session_state.databases.get("custom")
    if db is not None:
        answer, _ = query_database(db, question)
        return answer

    return "No relevant documents found."


def create_master_agent():
    """创建顶层 ReAct Agent并注册可调用的工具。"""
    tools = [
        Tool(
            name="NBA_Stats",
            func=nba_stats_tool,
            description=(
                "Use this tool for ANY NBA player statistics question, "
                "including specific seasons, averages, totals, or comparisons. "
                "IMPORTANT: Pass the FULL original user question as the input argument.\n"
                "Do NOT summarize or shorten it.\n"
                "Always use this tool first for player stats."
            ),
        ),
        Tool(
            name="RAG_Database",
            func=rag_tool_wrapper,
            description=(
                "Use this tool for ANY question about internal knowledge, "
                "custom players, or information that might exist in uploaded documents "
                "or the local data directory. "
                "If unsure, try this tool before Web_Search."
            ),
        ),
        Tool(
            name="Web_Search",
            func=DuckDuckGoSearchRun(num_results=5).run,
            description="Use ONLY if the question is not about NBA stats or internal documents.",
        ),
    ]

    # 这个系统提示词的作用不是回答问题，而是约束 agent 的决策顺序：
    # NBA 数据优先走 NBA_Stats，内部知识优先走 RAG，最后才考虑 Web 搜索。
    system_prompt = """
You are a master NBA assistant.

Rules:
1. If the question is about NBA player statistics OR NBA teams -> ALWAYS use NBA_Stats tool.
2. Do NOT use Web_Search for player stats or team information.
3. If the question might be about internal documents or custom knowledge -> use RAG_Database BEFORE Web_Search.
4. Only use Web_Search for general knowledge or news.
5. Think step by step.
6. Never skip tools.
"""

    agent = create_react_agent(
        model=st.session_state.llm,
        tools=tools,
        debug=True,
        state_modifier=system_prompt,
    )
    return agent


def _handle_web_fallback(question: str) -> tuple[str, list]:
    """当本地知识库没有结果时，执行一次网页搜索作为兜底。"""
    st.info("No relevant documents found. Searching web...")
    search = DuckDuckGoSearchRun(num_results=5)
    try:
        result = search.run(question)
        return f"Web Search Result:\n{result}", []
    except Exception as e:
        return f"Web search failed: {e}", []
