import os

# RAG System Configuration

# Google Gemini API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-api-key-here")

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local model
# EMBEDDING_MODEL = "models/embedding-001"  # Google model (requires API key)

# Vector Database Configuration
VECTOR_DB_TYPE = "chromadb"  # Options: "chromadb", "faiss"
VECTOR_DB_PATH = "./vector_db"

# Enhanced Chunking Configuration
CHUNK_SIZE = 800  # Smaller chunks for better precision
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 100  # Minimum chunk size to avoid tiny fragments

# Retrieval Configuration
TOP_K_RESULTS = 7  # More results for better context
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity threshold

# Enhanced PDF Processing
PDF_EXTRACT_IMAGES = False  # Set to True if you want image text extraction
PDF_PRESERVE_LAYOUT = True  # Better layout preservation

# LLM Configuration
LLM_MODEL = "gemini-1.5-flash"  # Options: gemini-1.5-flash, gemini-1.5-pro
MAX_TOKENS = 2000  # Increased for more detailed responses
TEMPERATURE = 0.7

# Enhanced Answer Generation
ENABLE_FALLBACK_LLM = True  # Enable LLM responses when no relevant docs found
FALLBACK_CONFIDENCE_THRESHOLD = 0.4  # Below this, use LLM fallback
COMBINE_RETRIEVAL_AND_LLM = True  # Combine document info with LLM knowledge

# File Upload Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = ['.pdf', '.xlsx', '.xls', '.docx', '.txt']

# UI Configuration
PAGE_TITLE = "RAG System - Document Q&A"
PAGE_ICON = "📚"