from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import APIError
from langsmith import tracing_context
from sentence_transformers import SentenceTransformer

from rag import (
    MODEL_NAME,
    answer_with_groq,
    build_index,
    evidence_only_answer,
    has_pdf_signature,
    load_pdf_documents,
    retrieve,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent

load_dotenv(APP_DIR / ".env")
os.environ.setdefault("LANGSMITH_PROJECT", "intelligent-hr-policy-rag")

st.set_page_config(
    page_title="Intelligent HR Policy RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the semantic search model...")
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def load_local_documents() -> tuple[list[tuple[str, bytes]], list[str]]:
    return load_pdf_documents(DATA_DIR)


def document_digest(documents: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, content in documents:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def render_sources(matches: list[dict[str, Any]]) -> None:
    if not matches:
        return
    with st.expander(f"Sources ({len(matches)})", expanded=False):
        for item in matches:
            st.markdown(
                f"**[{item['citation']}] {item['source']} — page {item['page']}**  \n"
                f"Relevance: {item['score']:.0%}"
            )
            st.write(item["text"])
            if item is not matches[-1]:
                st.divider()


if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.title("📚 Intelligent HR Policy RAG")
st.markdown(
    "Ask questions in plain language and get answers grounded in your HR policy PDFs, "
    "with page-level citations."
)

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Add policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDFs are processed in memory and are not saved to disk.",
    )
    st.caption(f"Looking for local PDFs in `{DATA_DIR}`")

    st.divider()
    st.header("Answer settings")
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    langsmith_configured = bool(os.getenv("LANGSMITH_API_KEY"))
    st.caption(
        f"Groq API key: {'configured' if groq_configured else 'not configured'}"
    )
    use_groq = st.checkbox(
        "Generate answers with Groq",
        value=groq_configured,
        disabled=not groq_configured,
        help="Answers are grounded in retrieved passages and include citations.",
    )
    model_name = st.text_input(
        "Groq model",
        value=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        disabled=not use_groq,
        help="If you set GROQ_MODEL in .env, make sure it is an active Groq model ID.",
    )
    trace_enabled = st.checkbox(
        "Trace with LangSmith",
        value=langsmith_configured,
        disabled=not langsmith_configured,
        help="Tracing sends the question, retrieved passages, and answer to LangSmith.",
    )
    if trace_enabled:
        st.caption("Question, retrieved policy text, and generated answers are sent to LangSmith.")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=8, value=4)
    relevance_threshold = st.slider(
        "Minimum relevance",
        min_value=-0.1,
        max_value=0.8,
        value=0.15,
        step=0.05,
        help="Passages below this semantic similarity are not used to answer.",
    )
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()
    st.divider()
    st.link_button(
        "View on GitHub",
        "https://github.com/charan26937/intelligent-rag-using-ai",
        width="stretch",
    )

local_documents, local_read_errors = load_local_documents()
documents = list(local_documents)
upload_errors: list[str] = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        if not has_pdf_signature(pdf_bytes):
            upload_errors.append(
                f"{uploaded_file.name}: this file is not a PDF. "
                "Upload the original PDF document."
            )
            continue
        documents.append((f"Upload: {uploaded_file.name}", pdf_bytes))

if local_read_errors:
    with st.expander("Local document read errors", expanded=not local_documents):
        for error in local_read_errors:
            st.warning(error)

if upload_errors:
    for error in upload_errors:
        st.warning(error)

if documents:
    st.caption(
        f"{len(documents)} PDF file(s) selected "
        f"({len(local_documents)} local, {len(documents) - len(local_documents)} uploaded)."
    )
else:
    st.info("Upload one or more searchable PDFs from the sidebar to get started.")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Your question",
        placeholder="For example: How many days of annual leave are employees entitled to?",
    )
    submitted = st.form_submit_button("Search policies", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question before searching.")
    elif not documents:
        st.error("No PDF documents are available. Upload a policy PDF in the sidebar.")
    elif use_groq and not model_name.strip():
        st.error("Enter a Groq model name or turn off Groq answer generation.")
    else:
        digest = document_digest(documents)
        store = st.session_state.get("document_store")
        if store is None or st.session_state.get("document_digest") != digest:
            try:
                with st.spinner("Reading PDFs and building the semantic search index..."):
                    model = get_embedding_model()
                    store = build_index(documents, model)
                st.session_state.document_store = store
                st.session_state.document_digest = digest
            except (OSError, ValueError, RuntimeError) as error:
                st.error(f"Could not index the selected PDFs: {error}")
                store = None

        if store is not None:
            for invalid_document in store["invalid_documents"]:
                st.warning(
                    f"Skipped **{invalid_document['source']}**: "
                    f"{invalid_document['error']}"
                )
            try:
                matches = retrieve(question, store, top_k=top_k)
            except (OSError, RuntimeError, ValueError) as error:
                st.error(str(error))
                matches = []

            relevant_matches = [
                match for match in matches if match["score"] >= relevance_threshold
            ]
            if not relevant_matches:
                answer = (
                    "I couldn't find a sufficiently relevant passage in the indexed "
                    "policies to answer that. Try rephrasing the question or adding "
                    "a more relevant policy document."
                )
            elif use_groq:
                history = [
                    {"role": message["role"], "content": message["content"]}
                    for message in st.session_state.conversation
                ]
                try:
                    with tracing_context(enabled=trace_enabled):
                        answer = answer_with_groq(
                            question,
                            relevant_matches,
                            model_name.strip(),
                            history,
                        )
                except (APIError, RuntimeError, ValueError) as error:
                    if isinstance(error, APIError) and getattr(
                        error, "status_code", None
                    ) == 404:
                        st.error(
                            f"Groq returned 404 for model `{model_name.strip()}`. "
                            "It may be retired or unavailable. Try "
                            "`openai/gpt-oss-120b`, and update "
                            "`GROQ_MODEL` in `.env` if it is set."
                        )
                    else:
                        st.error(f"Groq request failed: {error}")
                    answer = ""
            else:
                answer = evidence_only_answer(relevant_matches)

            if answer:
                st.session_state.conversation.extend(
                    [
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": answer,
                            "matches": relevant_matches,
                        },
                    ]
                )
                st.rerun()

if st.session_state.conversation:
    st.divider()
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("matches", []))
