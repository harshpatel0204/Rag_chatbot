"""
Configuration settings for the RAG Chatbot application.
All configurable parameters are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# PATH CONFIGURATION
# ============================================

# Base directory of the project
BASE_DIR = Path(__file__).parent.absolute()

# Directory for storing uploaded PDFs
PDF_STORAGE_DIR = BASE_DIR / "data" / "pdfs"

# Directory for storing the vector database
VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore"

# Create directories if they don't exist
PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# MODEL CONFIGURATION
# ============================================

# Groq API Key - set via environment variable or directly here
# Get your free API key from: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Groq model name
# Examples: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embedding model from sentence-transformers
# This is a small, fast model with good quality
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ============================================
# TEXT CHUNKING CONFIGURATION
# ============================================

# Default chunk size (number of characters per chunk)
DEFAULT_CHUNK_SIZE = 1000

# Default chunk overlap (overlap between consecutive chunks)
DEFAULT_CHUNK_OVERLAP = 200

# Minimum and maximum values for UI sliders
MIN_CHUNK_SIZE = 500
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_OVERLAP = 50
MAX_CHUNK_OVERLAP = 500

# ============================================
# RETRIEVAL CONFIGURATION
# ============================================

# Number of documents to retrieve for context
TOP_K_DOCUMENTS = 4

# Minimum similarity score threshold (0.0 to 1.0)
# Documents below this score will be filtered out
SIMILARITY_THRESHOLD = 0.3

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Enable/disable verbose logging
VERBOSE_LOGGING = True

# Log file path (None for console only)
LOG_FILE = None
