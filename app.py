"""
app.py
------
Professional Streamlit frontend for the Legal RAG Chatbot.

CHANGES FROM PREVIOUS VERSION:
  [CHANGE-UI-1] Complete visual redesign — luxury legal dark-gold aesthetic
  [CHANGE-UI-2] Disclaimer banners for invalid PDF uploads (legal_validator gate)
  [CHANGE-UI-3] Disclaimer banners for invalid chat queries (legal_validator gate)
  [CHANGE-UI-4] Document management panel showing indexed docs + chunk count
  [CHANGE-UI-5] Source viewer with relevance scores per retrieved chunk
  [CHANGE-UI-6] Professional loading states and status indicators

WHERE TO PUT YOUR GROQ API KEY:
  → In the  .env  file in your project root:
      GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
  → This file reads it automatically via rag_pipeline.py → python-dotenv
  → You do NOT need to paste the key anywhere in this file.
"""

import streamlit as st
import os
from rag_pipeline import (
    ingest_pdf,
    ask_question,
    get_indexed_documents,
    get_total_chunks,
    clear_index,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexaRAG — Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# [CHANGE-UI-1] CUSTOM CSS — Luxury Legal Dark-Gold Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #0D0D0F;
    color: #E8E0D0;
}

/* ── Main container ───────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2.5rem 3rem;
    max-width: 1200px;
    background-color: #0D0D0F;
}

/* ── Hide default Streamlit chrome ───────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111115 0%, #0A0A0D 100%);
    border-right: 1px solid #2A2520;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem;
}

/* ── Sidebar headings ─────────────────────────────────────────────────────── */
.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #C9A84C;
    letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: #6B6560;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1E1C18;
}
.sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7A7268;
    margin-bottom: 0.6rem;
    margin-top: 1.2rem;
}

/* ── File uploader ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed #2E2920 !important;
    border-radius: 8px !important;
    background: #141410 !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #C9A84C !important;
}
[data-testid="stFileUploader"] label {
    color: #9A9088 !important;
    font-size: 0.82rem !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent;
    border: 1px solid #C9A84C;
    color: #C9A84C;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    padding: 0.45rem 1.1rem;
    border-radius: 4px;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    background: #C9A84C;
    color: #0D0D0F;
}

/* ── Page header ──────────────────────────────────────────────────────────── */
.page-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #1E1C18;
    margin-bottom: 1.5rem;
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #E8E0D0;
    letter-spacing: -0.01em;
    line-height: 1.2;
    margin: 0;
}
.page-title span { color: #C9A84C; }
.page-subtitle {
    font-size: 0.85rem;
    color: #6B6560;
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
}

/* ── Status bar ───────────────────────────────────────────────────────────── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: #111115;
    border: 1px solid #1E1C18;
    border-radius: 6px;
    padding: 0.6rem 1.1rem;
    margin-bottom: 1.25rem;
    font-size: 0.78rem;
    color: #7A7268;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #3DA05A;
    display: inline-block;
    margin-right: 0.4rem;
    box-shadow: 0 0 6px #3DA05A80;
}
.status-dot.inactive { background: #4A4440; box-shadow: none; }
.status-item { display: flex; align-items: center; gap: 0.3rem; }
.status-value { color: #C9A84C; font-weight: 500; }

/* ── Chat messages ────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    flex-direction: row-reverse;
}

/* user bubble */
.user-bubble {
    background: #1A1A20;
    border: 1px solid #2A2530;
    border-radius: 12px 4px 12px 12px;
    padding: 0.85rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 75%;
    margin-left: auto;
    color: #D8D0C0;
}

/* assistant bubble */
.assistant-bubble {
    background: #111115;
    border: 1px solid #C9A84C22;
    border-left: 3px solid #C9A84C;
    border-radius: 4px 12px 12px 4px;
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    line-height: 1.75;
    max-width: 90%;
    color: #D0C8B8;
}

/* ── Disclaimer boxes ─────────────────────────────────────────────────────── */
.disclaimer-box {
    background: #160E0A;
    border: 1px solid #8B3A20;
    border-left: 4px solid #D9521A;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #C8A090;
}
.disclaimer-box .disc-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: #D9521A;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Success / Info boxes ─────────────────────────────────────────────────── */
.success-pill {
    background: #0A160F;
    border: 1px solid #2E5A38;
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    font-size: 0.78rem;
    color: #5CB87A;
    margin: 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Source excerpt cards ─────────────────────────────────────────────────── */
.source-card {
    background: #0E0E12;
    border: 1px solid #1E1C18;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.78rem;
    line-height: 1.65;
    color: #8A8278;
    font-family: 'JetBrains Mono', monospace;
}
.source-card .source-meta {
    font-family: 'Source Sans 3', sans-serif;
    font-size: 0.7rem;
    color: #5A5450;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
}
.source-card .score-badge {
    color: #C9A84C;
    font-weight: 500;
}

/* ── Doc list in sidebar ──────────────────────────────────────────────────── */
.doc-pill {
    background: #111115;
    border: 1px solid #1E1C18;
    border-radius: 5px;
    padding: 0.45rem 0.75rem;
    font-size: 0.75rem;
    color: #9A9088;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.doc-pill::before {
    content: "⚖";
    color: #C9A84C;
    font-size: 0.7rem;
    flex-shrink: 0;
}

/* ── Chat input ───────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #111115 !important;
    border: 1px solid #2A2520 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 2px #C9A84C20 !important;
}
[data-testid="stChatInput"] textarea {
    color: #E8E0D0 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border-color: #1E1C18 !important; }

/* ── Expander ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #0E0E12 !important;
    border: 1px solid #1E1C18 !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #7A7268 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p { color: #7A7268 !important; font-size: 0.8rem !important; }

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0D0D0F; }
::-webkit-scrollbar-thumb { background: #2A2520; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #C9A84C; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚖ LexaRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Legal Document Intelligence</div>', unsafe_allow_html=True)

    # ── Upload section ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">Upload Legal Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # [CHANGE-UI-2] Process uploads with legal validation gate
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                save_dir = "./data/uploaded_docs"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, uploaded_file.name)

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner(f"Analysing {uploaded_file.name[:28]}..."):
                    result = ingest_pdf(save_path)

                if result["success"]:
                    # ✅ Legal document accepted
                    st.markdown(
                        f'<div class="success-pill">✓ {uploaded_file.name[:32]} '
                        f'— {result["chunks"]} sections indexed</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.processed_files.add(uploaded_file.name)
                else:
                    # [CHANGE-UI-2] Show disclaimer for non-legal document
                    st.markdown(
                        f'<div class="disclaimer-box">'
                        f'<div class="disc-title">⊘ Document Rejected</div>'
                        f'{result["reason"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    # Remove the invalid file
                    if os.path.exists(save_path):
                        os.remove(save_path)

    # ── Indexed documents ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">Indexed Documents</div>', unsafe_allow_html=True)

    indexed_docs = get_indexed_documents()
    if indexed_docs:
        for doc in indexed_docs:
            name = doc[:38] + "…" if len(doc) > 38 else doc
            st.markdown(f'<div class="doc-pill">{name}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:0.75rem;color:#4A4440;padding:0.4rem 0;">'
            'No documents indexed yet.</div>',
            unsafe_allow_html=True
        )

    # ── Reset button ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">Actions</div>', unsafe_allow_html=True)

    if st.button("⊘  Clear All Data"):
        clear_index()
        st.session_state.processed_files = set()
        st.session_state.messages = []
        st.rerun()

    if st.button("✕  Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # ── Legal notice ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem;color:#3E3C38;line-height:1.6;padding:0 0.1rem;">'
        'LexaRAG provides document-grounded information only and does not '
        'constitute legal advice. Always consult a qualified attorney for '
        'case-specific legal matters.'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div class="page-title">Legal Document <span>Intelligence</span></div>
    <div class="page-subtitle">
        Ask precise questions. Receive answers grounded exclusively in your uploaded legal documents.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Status bar ─────────────────────────────────────────────────────────────
total_chunks = get_total_chunks()
total_docs   = len(get_indexed_documents())
is_ready     = total_chunks > 0

st.markdown(f"""
<div class="status-bar">
    <div class="status-item">
        <span class="status-dot {'active' if is_ready else 'inactive'}"></span>
        <span>System {'Ready' if is_ready else 'Idle'}</span>
    </div>
    <div class="status-item">
        Documents indexed: <span class="status-value">{total_docs}</span>
    </div>
    <div class="status-item">
        Total sections: <span class="status-value">{total_chunks}</span>
    </div>
    <div class="status-item">
        Vector DB: <span class="status-value">FAISS</span>
    </div>
    <div class="status-item">
        LLM: <span class="status-value">Groq · Llama-3</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: #3E3C38;
        font-family: 'Playfair Display', serif;
    ">
        <div style="font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.4;">⚖</div>
        <div style="font-size: 1.1rem; color: #5A5450; margin-bottom: 0.5rem;">
            Upload a legal document to begin
        </div>
        <div style="font-size: 0.8rem; color: #3A3830; max-width: 440px; margin: 0 auto; line-height: 1.7;">
            This system analyses contracts, court orders, legislation, NDAs, wills,
            corporate filings, and all professional legal documents.
        </div>
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            # Could be answer or disclaimer
            if message.get("is_disclaimer"):
                st.markdown(
                    f'<div class="disclaimer-box">'
                    f'<div class="disc-title">⊘ Out-of-Scope Query</div>'
                    f'{message["content"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
                # Show sources if stored
                if message.get("sources"):
                    with st.expander("📄 View source excerpts & relevance scores"):
                        for i, src in enumerate(message["sources"], 1):
                            score_pct = round(src["score"] * 100, 1)
                            st.markdown(
                                f'<div class="source-card">'
                                f'<div class="source-meta">'
                                f'<span>Excerpt {i} — {src["source"]}</span>'
                                f'<span class="score-badge">Relevance: {score_pct}%</span>'
                                f'</div>'
                                f'{src["text"][:500]}{"…" if len(src["text"]) > 500 else ""}'
                                f'</div>',
                                unsafe_allow_html=True
                            )


# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your legal documents…"):

    # Display user message
    with st.chat_message("user"):
        st.markdown(
            f'<div class="user-bubble">{prompt}</div>',
            unsafe_allow_html=True
        )
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant sections and formulating response…"):
            result = ask_question(prompt)

        if not result["success"]:
            # [CHANGE-UI-3] Show disclaimer for invalid / off-topic query
            st.markdown(
                f'<div class="disclaimer-box">'
                f'<div class="disc-title">⊘ Query Not Processed</div>'
                f'{result["reason"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["reason"],
                "is_disclaimer": True,
                "sources": []
            })

        else:
            # ✅ Valid answer
            answer  = result["answer"]
            sources = result.get("sources", [])

            st.markdown(
                f'<div class="assistant-bubble">{answer}</div>',
                unsafe_allow_html=True
            )

            # [CHANGE-UI-5] Source cards with relevance scores
            if sources:
                with st.expander("📄 View source excerpts & relevance scores"):
                    for i, src in enumerate(sources, 1):
                        score_pct = round(src["score"] * 100, 1)
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="source-meta">'
                            f'<span>Excerpt {i} — {src["source"]}</span>'
                            f'<span class="score-badge">Relevance: {score_pct}%</span>'
                            f'</div>'
                            f'{src["text"][:500]}{"…" if len(src["text"]) > 500 else ""}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "is_disclaimer": False,
                "sources": sources
            })