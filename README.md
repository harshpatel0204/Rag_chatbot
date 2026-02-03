# 🤖 RAG Chatbot

A production-ready Python RAG (Retrieval-Augmented Generation) Chatbot with Streamlit UI. Chat normally or upload PDFs to have AI-powered conversations about your documents.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **💬 Normal Chat Mode**: Have natural conversations with the AI
- **📚 RAG Mode**: Upload PDFs and ask questions about their content
- **🔄 Auto-Switch**: Automatically switches between normal and RAG mode
- **💾 Persistent Storage**: Vector embeddings are saved locally
- **⚙️ Configurable**: Adjust chunk size and overlap via UI
- **⚡ Super Fast**: Uses Groq API for lightning-fast inference (free tier available)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq (Llama 3.3 / Mixtral) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS |
| **PDF Parsing** | PyPDF2 |
| **UI** | Streamlit |
| **Orchestration** | LangChain |

## 📋 Prerequisites

### 1. Python 3.10+

Make sure you have Python 3.10 or higher installed:

```bash
python --version
```

### 2. Groq API Key (Free)

Get your free API key from Groq Console:

1. Visit [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign in or create an account
3. Click "Create API Key"
4. Copy the key for use in the setup

## 🚀 Installation

### 1. Clone/Navigate to Project

```bash
cd rag_chatbot
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API Key

Choose one of these methods:

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell:
$env:GROQ_API_KEY = "your-api-key-here"

# Windows Command Prompt:
set GROQ_API_KEY=your-api-key-here

# macOS/Linux:
export GROQ_API_KEY="your-api-key-here"
```

**Option B: Direct in config.py**
Edit `config.py` and add your key:
```python
GROQ_API_KEY = "your-api-key-here"
```

## 🎮 Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Normal Chat Mode

1. Simply type your message in the chat input
2. Press Enter to send
3. The AI will respond naturally

### RAG Mode (Chat with PDFs)

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Configure Chunking** (optional): Adjust chunk size and overlap
3. **Process**: Click "Process PDFs" to index the documents
4. **Ask Questions**: Type questions about your documents
5. **View Sources**: Expand the "Sources" section to see where answers came from

## 📁 Project Structure

```
rag_chatbot/
│
├── app.py                      # Streamlit entry point
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── loaders/
│   ├── __init__.py
│   └── pdf_loader.py           # PDF loading & parsing
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py             # Sentence transformer embeddings
│
├── vectorstore/
│   ├── __init__.py
│   └── store.py                # FAISS vector storage
│
├── retriever/
│   ├── __init__.py
│   └── retriever.py            # Document retrieval
│
├── chat/
│   ├── __init__.py
│   └── conversation.py         # Groq chat integration
│
├── prompts/
│   ├── __init__.py
│   ├── chat_prompt.py          # Normal chat prompts
│   └── rag_prompt.py           # RAG-specific prompts
│
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Utility functions
│
└── data/
    ├── pdfs/                   # Uploaded PDFs (auto-created)
    └── vectorstore/            # Saved embeddings (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Change the Groq model
GROQ_MODEL = "llama-3.3-70b-versatile"  # or "mixtral-8x7b-32768", etc.

# Adjust default chunking
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Modify retrieval settings
TOP_K_DOCUMENTS = 4
SIMILARITY_THRESHOLD = 0.3
```

## 🔧 Troubleshooting

### API Key Error

**Error**: `Groq API key not found`

**Solution**: 
- Make sure you've set the `GROQ_API_KEY` environment variable
- Or add it directly to `config.py`
- Get a free key at: https://console.groq.com/keys

### Rate Limit Error

**Error**: `Rate limit reached`

**Solution**: 
- Groq's free tier has limits on requests per minute (RPM) and tokens per minute (TPM).
- Wait a few seconds before retrying.
- Switch to a smaller model like `llama-3.1-8b-instant` in `config.py` if needed.

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for lightning-fast AI inference
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io) for the UI framework
- [LangChain](https://langchain.com) for orchestration