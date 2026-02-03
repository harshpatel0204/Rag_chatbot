"""
RAG Chatbot - Streamlit Application

A chatbot with RAG capabilities for chatting with PDF documents.
Uses Groq for LLM inference and sentence-transformers for embeddings.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE,
    MIN_CHUNK_OVERLAP, MAX_CHUNK_OVERLAP,
    GROQ_MODEL
)
from loaders.pdf_loader import PDFLoader
from vectorstore.store import get_vector_store, reset_vector_store
from retriever.retriever import get_retriever
from chat.conversation import ConversationManager
from utils.helpers import chunk_documents

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Mode indicator styling */
    .mode-indicator {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    
    .mode-normal {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    
    .mode-rag {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    
    /* Document info styling */
    .doc-info {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }
    
    /* Status message styling */
    .status-message {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'conversation_manager' not in st.session_state:
        st.session_state.conversation_manager = ConversationManager()
    
    if 'pdf_loader' not in st.session_state:
        st.session_state.pdf_loader = PDFLoader()
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vector_store()
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = get_retriever()
    
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    
    if 'chunk_overlap' not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
    
    if 'mode' not in st.session_state:
        st.session_state.mode = 'normal'
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

init_session_state()


# ============================================
# SIDEBAR
# ============================================

def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.markdown("---")
        
        # Model info
        st.markdown("### 🔧 Model")
        st.info(f"Using: **{GROQ_MODEL}**")
        
        st.markdown("---")
        
        # PDF Upload Section
        st.markdown("### 📄 PDF Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        # Chunk configuration
        st.markdown("#### Chunking Settings")
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=MIN_CHUNK_SIZE,
            max_value=MAX_CHUNK_SIZE,
            value=st.session_state.chunk_size,
            step=100,
            help="Number of characters per chunk"
        )
        st.session_state.chunk_size = chunk_size
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=MIN_CHUNK_OVERLAP,
            max_value=MAX_CHUNK_OVERLAP,
            value=st.session_state.chunk_overlap,
            step=50,
            help="Overlap between consecutive chunks"
        )
        st.session_state.chunk_overlap = chunk_overlap
        
        # Process PDFs button
        if uploaded_files:
            if st.button("📥 Process PDFs", type="primary", use_container_width=True):
                process_pdfs(uploaded_files)
        
        st.markdown("---")
        
        # Vector Store Status
        st.markdown("### 📊 Vector Store")
        
        doc_count = st.session_state.vector_store.get_document_count()
        sources = st.session_state.vector_store.get_sources()
        
        if doc_count > 0:
            st.success(f"✅ {doc_count} chunks indexed")
            if sources:
                st.markdown("**Loaded documents:**")
                for source in sources:
                    st.markdown(f"- {source}")
            
            # Mode indicator
            if st.session_state.mode == 'rag':
                st.markdown(
                    '<div class="mode-indicator mode-rag">📚 RAG Mode Active</div>',
                    unsafe_allow_html=True
                )
        else:
            st.warning("No documents loaded")
            st.markdown(
                '<div class="mode-indicator mode-normal">💬 Normal Chat Mode</div>',
                unsafe_allow_html=True
            )
        
        # Reset button
        if st.button("🗑️ Reset Vector Store", use_container_width=True):
            reset_vector_store()
            st.session_state.vector_store = get_vector_store()
            st.session_state.retriever = get_retriever()
            st.session_state.mode = 'normal'
            st.session_state.processed_files = []
            st.success("Vector store cleared!")
            st.rerun()
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Normal Chat**: Just type a message to chat with the AI
            2. **PDF Chat**: 
               - Upload PDF files in the sidebar
               - Click "Process PDFs" to index them
               - Ask questions about the documents
            3. **Reset**: Use the reset button to clear indexed documents
            
            **Tips:**
            - Adjust chunk size for different document types
            - Larger chunks = more context, slower processing
            - The app auto-switches between normal and RAG mode
            """)


def process_pdfs(uploaded_files):
    """Process uploaded PDF files."""
    with st.spinner("Processing PDFs..."):
        all_documents = []
        
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Read file bytes
            file_bytes = uploaded_file.read()
            
            # Load PDF
            documents = st.session_state.pdf_loader.load_pdf_from_bytes(
                file_bytes, 
                uploaded_file.name
            )
            
            all_documents.extend(documents)
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_documents:
            # Chunk documents
            st.info(f"Chunking {len(all_documents)} pages...")
            chunked_docs = chunk_documents(
                all_documents,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            
            # Add to vector store
            st.info(f"Indexing {len(chunked_docs)} chunks...")
            st.session_state.vector_store.add_documents(chunked_docs)
            
            # Update mode
            st.session_state.mode = 'rag'
            st.session_state.processed_files = [f.name for f in uploaded_files]
            
            st.success(f"✅ Processed {len(uploaded_files)} PDF(s) into {len(chunked_docs)} chunks!")
            st.rerun()
        else:
            st.error("No text could be extracted from the PDFs")


# ============================================
# CHAT INTERFACE
# ============================================

def render_chat():
    """Render the main chat interface."""
    # Header
    st.header("💬 Chat")
    
    # Mode indicator in main area
    if st.session_state.vector_store.get_document_count() > 0:
        st.caption("📚 RAG Mode: Answers will be based on uploaded documents")
    else:
        st.caption("💬 Normal Mode: General conversation")
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for RAG responses
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("📖 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- **{source['source']}** (Page {source['page']})")
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get chat history for context
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # Exclude current message
                ]
                
                # Check if we have documents for RAG
                if st.session_state.retriever.has_documents():
                    # Get context
                    context, docs = st.session_state.retriever.retrieve_with_context(prompt)
                    
                    if context:
                        # RAG response
                        response = st.session_state.conversation_manager.rag_chat(
                            prompt, context, chat_history
                        )
                        sources = [doc['metadata'] for doc in docs]
                        st.session_state.mode = 'rag'
                    else:
                        # No relevant context found
                        response = st.session_state.conversation_manager.chat(
                            prompt, chat_history
                        )
                        sources = []
                        st.session_state.mode = 'normal'
                else:
                    # Normal chat
                    response = st.session_state.conversation_manager.chat(
                        prompt, chat_history
                    )
                    sources = []
                    st.session_state.mode = 'normal'
                
                st.markdown(response)
                
                # Show sources if available
                if sources:
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.markdown(f"- **{source.get('source', 'Unknown')}** (Page {source.get('page', 'N/A')})")
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources if sources else None
        })


# ============================================
# MAIN APP
# ============================================

def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat()


if __name__ == "__main__":
    main()
