"""
Streamlit frontend for the Enterprise AI Knowledge Assistant.

A single-file dashboard with:
  - Sidebar: document upload, document list, delete/reindex actions
  - Main panel: chat window with conversation history, sources,
    confidence score, latency, and token usage per answer
"""
import os
import uuid
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------- Session state ----------
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: role, content, meta
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ---------- Theming ----------
if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #e6e6e6; }
        [data-testid="stSidebar"] { background-color: #161a23; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #ffffff; color: #1a1a1a; }
        [data-testid="stSidebar"] { background-color: #f5f5f7; }
        </style>
        """,
        unsafe_allow_html=True,
    )
# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 Documents")
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Upload & Index", use_container_width=True):
        with st.spinner("Processing documents..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{API_BASE_URL}/upload", files=files_payload, timeout=180)
                resp.raise_for_status()
                st.success(resp.json().get("message", "Uploaded."))
            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Indexed Documents")

    try:
        docs_resp = requests.get(f"{API_BASE_URL}/documents", timeout=30)
        docs_resp.raise_for_status()
        documents = docs_resp.json()
    except requests.RequestException as e:
        documents = []
        st.warning(f"Could not reach backend: {e}")

    selected_doc_ids = []
    for doc in documents:
        status_icon = {"indexed": "✅", "processing": "⏳", "failed": "❌"}.get(doc["status"], "•")
        col1, col2, col3 = st.columns([5, 1, 1])
        with col1:
            checked = st.checkbox(
                f"{status_icon} {doc['filename']} ({doc['num_chunks']} chunks)",
                key=f"chk_{doc['id']}",
            )
            if checked:
                selected_doc_ids.append(doc["id"])
        with col2:
            if st.button("🔁", key=f"reindex_{doc['id']}", help="Re-index"):
                requests.post(f"{API_BASE_URL}/reindex/{doc['id']}", timeout=180)
                st.rerun()
        with col3:
            if st.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                requests.delete(f"{API_BASE_URL}/document/{doc['id']}", timeout=30)
                st.rerun()

    st.session_state.selected_doc_ids = selected_doc_ids

    st.divider()
    strategy = st.selectbox("Retrieval strategy", ["similarity", "mmr", "hybrid"], index=0)
    top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)

    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# ---------- Main chat window ----------
st.title("🧠 Enterprise AI Knowledge Assistant")
st.caption("Ask questions grounded in your uploaded company documents.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", f"{meta['confidence_score']:.0%}")
            c2.metric("Latency", f"{meta['latency_ms']:.0f} ms")
            c3.metric("Tokens", meta["token_usage"]["total_tokens"])
            if meta.get("sources"):
                with st.expander(f"📎 Sources ({len(meta['sources'])})"):
                    for s in meta["sources"]:
                        st.markdown(
                            f"**{s['filename']}** · chunk {s['chunk_index']} "
                            f"· relevance {s['relevance_score']:.2f}"
                        )
                        st.text(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "question": question,
                "conversation_id": st.session_state.conversation_id,
                "document_ids": st.session_state.get("selected_doc_ids") or None,
                "retrieval_strategy": strategy,
                "top_k": top_k,
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Confidence", f"{data['confidence_score']:.0%}")
                c2.metric("Latency", f"{data['latency_ms']:.0f} ms")
                c3.metric("Tokens", data["token_usage"]["total_tokens"])

                if data.get("sources"):
                    with st.expander(f"📎 Sources ({len(data['sources'])})"):
                        for s in data["sources"]:
                            st.markdown(
                                f"**{s['filename']}** · chunk {s['chunk_index']} "
                                f"· relevance {s['relevance_score']:.2f}"
                            )
                            st.text(s["content"][:300] + ("..." if len(s["content"]) > 300 else ""))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": data,
                })
            except requests.RequestException as e:
                st.error(f"Query failed: {e}")
