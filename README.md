# 📚 Multi-PDF Chat AI

> An AI-powered application that lets you chat with multiple PDF documents simultaneously using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-purple?style=flat-square)

---

## 🎯 What It Does

Upload one or more PDF files and ask questions in natural language. The app retrieves the most relevant sections from your documents and uses LLaMA-3.1 to generate accurate, context-grounded answers — no hallucination, answers come only from your uploaded documents.

---

## 🧠 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  PDF Files  │────▶│ Text Extract │────▶│ Chunk & Split   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  TF-IDF Embed   │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  FAISS Index    │
                                          └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐     ┌────────▼────────┐
│  User Query │────▶│ Similarity   │────▶│  Top-K Chunks   │
└─────────────┘     │   Search     │     └────────┬────────┘
                    └──────────────┘              │
                                         ┌────────▼────────┐
                                         │ LLaMA-3.1 (Groq)│
                                         └────────┬────────┘
                                                  │
                                         ┌────────▼────────┐
                                         │    Answer ✅     │
                                         └─────────────────┘
```

---

## 🛠️ Tech Stack

|     Component     |       Technology        |      Purpose      |
|-----------|-----------|---------|
|        UI         |        Streamlit        |     Web interface |
|    PDF Parsing    |          PyPDF2         | Extract text from PDFs |
|  Text Splitting   |        LangChain        | Chunk documents with overlap |
|    Embeddings     | TF-IDF (scikit-learn)   | Convert text to vectors |
|   Vector Store    |           FAISS         | Fast similarity search |
|        LLM        | LLaMA-3.1-8B via Groq   | Generate answers |
| Orchestration     |       LangChain         | Pipeline management |

---

## ✨ Features

- 📄 **Multi-PDF support** — Upload and query multiple PDFs at once
- 💬 **Conversational memory** — Ask follow-up questions naturally
- ⚡ **Fast inference** — Groq API delivers ~500 tokens/sec
- 🔒 **Grounded answers** — LLM answers only from your documents
- 🎨 **Clean dark UI** — Professional interface built with Streamlit

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/DishaP612/Multi-pdf-chat-AI.git
cd Multi-pdf-chat-AI
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Live Demo

🔗 [**[Try it here → [your-app-link.streamlit.app](https://multi-pdf-chat-ai-e84kxwyta9vyl5nspjpxkn.streamlit.app/)](#)**]

*(Upload any PDF and start asking questions!)*

---

## 📁 Project Structure

```
multi-pdf-chat-ai/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Files to exclude from git
└── README.md           # Project documentation
```

---

## 💡 How RAG Works (Interview Ready)

**RAG = Retrieval-Augmented Generation**

Traditional LLMs hallucinate because they rely only on training data. RAG fixes this by:

1. **Indexing** — Documents are split into chunks and converted to vectors
2. **Retrieval** — When you ask a question, the most similar chunks are fetched
3. **Generation** — The LLM sees your question + retrieved chunks and generates a grounded answer

This means the model can **only answer from your actual documents** — no guessing.

---

## 🔑 Getting API Keys

| Key | Where to Get | Cost |
|-----|-------------|------|
| Groq API Key | [console.groq.com](https://console.groq.com) | Free |

---

## 👩‍💻 Author

**Disha** — Final Year CSE Student
📧 dishaprasanna.6125@gmail.com


---

## 📄 License

MIT License — feel free to use and modify.
