"""
rag_pipeline.py  — FIXED VERSION
FIX-1: ChatGroq initialized INSIDE generate_answer(), never at module level
FIX-2: GROQ_API_KEY has a hardcoded fallback
FIX-3: Updated imports to langchain_core.messages and langchain_text_splitters
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv
import fitz
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from legal_validator import validate_pdf_is_legal, validate_chat_query

load_dotenv(override=True)

# ── PASTE YOUR GROQ KEY BETWEEN THE QUOTES BELOW ──────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
# ──────────────────────────────────────────────────────────────────────────────

VECTOR_STORE_DIR = "./vector_store/faiss_index"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
METADATA_PATH    = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")
EMBEDDING_DIM    = 384

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs("./data/uploaded_docs", exist_ok=True)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def _load_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []
    return index, metadata


def _save_faiss_index(index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def embed_and_store(chunks, doc_name):
    index, metadata = _load_faiss_index()
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    embeddings = _normalize(np.array(embeddings, dtype="float32"))
    index.add(embeddings)
    for i, chunk in enumerate(chunks):
        metadata.append({"text": chunk, "source": doc_name, "chunk_index": i})
    _save_faiss_index(index, metadata)


def ingest_pdf(pdf_path):
    doc_name = os.path.basename(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    is_legal, reason = validate_pdf_is_legal(text, filename=doc_name)
    if not is_legal:
        return {"success": False, "reason": reason}
    chunks = chunk_text(text)
    embed_and_store(chunks, doc_name)
    return {"success": True, "chunks": len(chunks)}


def retrieve_relevant_chunks(query, top_k=5):
    index, metadata = _load_faiss_index()
    if index.ntotal == 0:
        return []
    query_emb = embedding_model.encode([query], show_progress_bar=False)
    query_emb = _normalize(np.array(query_emb, dtype="float32"))
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata[idx]
        results.append({"text": entry["text"], "source": entry["source"], "score": float(score)})
    return results


def generate_answer(query, context_chunks):
    context = "\n\n---\n\n".join([c["text"] for c in context_chunks])

    system_prompt = """You are a highly specialized legal document intelligence assistant.
STRICT RULES:
1. Answer ONLY from the provided document excerpts below.
2. If the answer is not in the excerpts, say: "I was unable to locate this information within the uploaded legal documents. Please consult a qualified legal professional."
3. Do NOT fabricate or infer beyond the text.
4. Maintain a formal, professional, precise tone at all times.
5. Cite relevant clause numbers or section headings when present."""

    user_prompt = f"""DOCUMENT EXCERPTS:
{context}

LEGAL QUERY: {query}

PROFESSIONAL RESPONSE:"""

    # ChatGroq is created HERE inside the function — never at module level
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.05,
        max_tokens=1024,
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)
    return response.content


def ask_question(query):
    is_valid, reason = validate_chat_query(query)
    if not is_valid:
        return {"success": False, "reason": reason}

    chunks = retrieve_relevant_chunks(query, top_k=5)
    if not chunks:
        return {
            "success": False,
            "reason": "⚠️ **No Documents Indexed**\n\nPlease upload a legal PDF document first."
        }

    answer = generate_answer(query, chunks)
    return {"success": True, "answer": answer, "sources": chunks}


def get_indexed_documents():
    _, metadata = _load_faiss_index()
    return list({m["source"] for m in metadata})


def get_total_chunks():
    index, _ = _load_faiss_index()
    return index.ntotal


def clear_index():
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)