import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import faiss
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_KEY_DEFAULT = os.getenv("GROQ_API_KEY", "")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-PDF Chat AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: #0f1117;
        color: #e8e8e8;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Title */
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6ee7f7, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }

    /* Chat bubbles */
    .user-bubble {
        background: #1e2433;
        border: 1px solid #2a3040;
        border-radius: 12px 12px 2px 12px;
        padding: 12px 16px;
        margin: 8px 0 8px 20%;
        color: #e8e8e8;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .bot-bubble {
        background: linear-gradient(135deg, #1a1f30, #1e2535);
        border: 1px solid #6ee7f730;
        border-left: 3px solid #6ee7f7;
        border-radius: 2px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 20% 8px 0;
        color: #d1d5db;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .role-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
        color: #6b7280;
    }

    .user-label { color: #a78bfa; }
    .bot-label  { color: #6ee7f7; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6ee7f7, #a78bfa);
        color: #0f1117;
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        width: 100%;
        letter-spacing: 0.05em;
    }

    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
        transition: all 0.2s;
    }

    /* File uploader */
    .stFileUploader {
        border: 1.5px dashed #2a3040;
        border-radius: 10px;
        padding: 6px;
    }

    /* Status box */
    .status-box {
        background: #161b27;
        border: 1px solid #2a3040;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #6b7280;
        font-family: 'Space Mono', monospace;
    }

    /* Input */
    .stTextInput > div > div > input {
        background: #161b27;
        border: 1px solid #2a3040;
        border-radius: 8px;
        color: #e8e8e8;
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ───────────────────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files):
    """Extract raw text from all uploaded PDFs."""
    full_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def split_into_chunks(raw_text):
    """Split text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(raw_text)


def build_vector_store(chunks):
    """Embed chunks using TF-IDF and store in FAISS index."""
    vectorizer = TfidfVectorizer(max_features=512)
    tfidf_matrix = vectorizer.fit_transform(chunks).toarray().astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_matrix = tfidf_matrix / norms
    index = faiss.IndexFlatIP(tfidf_matrix.shape[1])
    index.add(tfidf_matrix)
    return {"index": index, "vectorizer": vectorizer, "chunks": chunks}


def search_chunks(store, query, k=4):
    """Retrieve top-k chunks for a query."""
    vectorizer = store["vectorizer"]
    index = store["index"]
    chunks = store["chunks"]
    qvec = vectorizer.transform([query]).toarray().astype(np.float32)
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec = qvec / norm
    _, indices = index.search(qvec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def build_llm(api_key):
    """Build Groq LLM."""
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
    )


def ask_question(store, llm, question, chat_history):
    """Retrieve context and ask LLM."""
    context_chunks = search_chunks(store, question, k=4)
    context = "\n\n".join(context_chunks)
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"
    prompt = f"""You are a helpful AI. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the uploaded documents."

Context:
{context}

Conversation so far:
{history_text}
User: {question}
Assistant:"""
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def render_chat_history():
    """Render all messages in styled bubbles."""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="role-label user-label">▸ YOU</div>
            <div class="user-bubble">{msg["content"]}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="role-label bot-label">◈ AI</div>
            <div class="bot-bubble">{msg["content"]}</div>
            """, unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Multi-PDF Chat AI")
    st.markdown("---")

    # API Key inputs
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        value=GROQ_KEY_DEFAULT,
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com"
    )

    st.markdown("---")

    # PDF Upload
    st.markdown("**Upload Your PDFs**")
    uploaded_pdfs = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_pdfs:
        st.markdown(f"<div class='status-box'>📄 {len(uploaded_pdfs)} file(s) ready</div>", unsafe_allow_html=True)
        for f in uploaded_pdfs:
            st.markdown(f"<div style='color:#6b7280;font-size:0.78rem;padding:2px 0;'>· {f.name}</div>", unsafe_allow_html=True)

    st.markdown("")

    process_btn = st.button("⚡ PROCESS PDFs")

    if process_btn:
        if not groq_api_key:
            st.error("Please enter your Groq API key.")
        elif not uploaded_pdfs:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting → Chunking → Indexing..."):
                raw_text = extract_text_from_pdfs(uploaded_pdfs)
                if not raw_text.strip():
                    st.error("Could not extract text from PDFs.")
                else:
                    chunks = split_into_chunks(raw_text)
                    st.session_state.vector_store = build_vector_store(chunks)
                    st.session_state.llm = build_llm(groq_api_key)
                    st.session_state.processed = True
                    st.session_state.pdf_names = [f.name for f in uploaded_pdfs]
                    st.session_state.chat_history = []
                    st.success(f"✅ {len(chunks)} chunks indexed!")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.conversation_chain = None
        st.session_state.processed = False

    st.markdown("""
    <div style='color:#4b5563;font-size:0.72rem;margin-top:1rem;font-family:Space Mono,monospace;'>
    RAG PIPELINE<br>
    PyPDF2 → LangChain<br>
    FAISS + MiniLM-L6<br>
    Groq · LLaMA3-8B
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>Multi-PDF Chat AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload PDFs → Process → Ask anything across all documents</div>", unsafe_allow_html=True)

if not st.session_state.processed:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#374151;'>
        <div style='font-size:3rem;'>📄</div>
        <div style='font-family:Space Mono,monospace;font-size:0.85rem;margin-top:1rem;'>
            Upload PDFs in the sidebar and click PROCESS PDFs to begin
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Show indexed doc names
    names_html = " &nbsp;·&nbsp; ".join([f"<span style='color:#6ee7f7'>{n}</span>" for n in st.session_state.pdf_names])
    st.markdown(f"<div style='font-size:0.78rem;color:#6b7280;margin-bottom:1rem;'>Indexed: {names_html}</div>", unsafe_allow_html=True)

    # Chat history
    render_chat_history()

    # Input
    user_question = st.text_input(
        "Ask a question about your PDFs",
        placeholder="e.g. Summarize the key findings... / What does section 3 say about...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([5, 1])
    with col2:
        ask_btn = st.button("ASK →")

    if ask_btn and user_question.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner("Thinking..."):
            answer = ask_question(
                st.session_state.vector_store,
                st.session_state.llm,
                user_question,
                st.session_state.chat_history[:-1]
            )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()