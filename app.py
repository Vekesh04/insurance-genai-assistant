from __future__ import annotations

import streamlit as st

from config import DATABASE_PATH, LLM_PROVIDER, OPENAI_API_KEY
from graph.workflow import ask_assistant

st.set_page_config(page_title="Insurance Gen-AI Assistant", page_icon="🛡️", layout="wide")

st.title("🛡️ Enterprise Insurance Claims & Policy Intelligence Assistant")
st.caption("Python • SQLite • Hybrid RAG • LangGraph • Guardrails • FastAPI")

with st.sidebar:
    st.header("System Status")
    st.write("Database:", "✅ Ready" if DATABASE_PATH.exists() else "❌ Not created")
    if LLM_PROVIDER == "openai":
        st.write("LLM:", "✅ OpenAI configured" if OPENAI_API_KEY else "⚪ OpenAI key not set — local fallback active")
    else:
        st.write("LLM provider:", LLM_PROVIDER)
    st.markdown("**Demo questions**")
    st.code("What is the status of claim CLM300001?")
    st.code("What is the maternity waiting period under Health Secure Plus?")
    st.code("Analyze claim CLM300001 and tell me if documents are missing.")

if not DATABASE_PATH.exists():
    st.error("Database not found. In VS Code terminal run: python -m database.create_database")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.write(item["answer"])
        st.caption(f"Route: {item['route']} — {item.get('route_reason') or ''}")

question = st.chat_input("Ask about a claim, policy, customer, hospital, documents, coverage, waiting period...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Routing question and retrieving grounded evidence..."):
            result = ask_assistant(question)
        st.write(result["answer"])
        st.caption(f"Route: {result['route']} — {result.get('route_reason') or ''}")

        if result.get("rag_result") and result["rag_result"].get("sources"):
            with st.expander("Policy evidence"):
                for source in result["rag_result"]["sources"]:
                    st.markdown(f"**{source['product_name']} — Section {source['section_number']}: {source['section_title']}**")
                    st.write(source["text"])
                    st.caption(f"Hybrid score: {source['score']} • {source['source_file']}")

        if result.get("sql_result"):
            with st.expander("Structured database evidence"):
                st.json(result["sql_result"])

    st.session_state.history.append(result)
