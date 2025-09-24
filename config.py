# RAG System Configuration

# Google Gemini API Configuration
GOOGLE_API_KEY = "AIzaSyDTSILO7BRCgTtgx2-tbh1N9tGngDSO_Q4"

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local model
# EMBEDDING_MODEL = "models/embedding-001"  # Google model (requires API key)

# Vector Database Configuration
VECTOR_DB_TYPE = "chromadb"  # Options: "chromadb", "faiss"
VECTOR_DB_PATH = "./vector_db"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K_RESULTS = 5

# LLM Configuration
LLM_MODEL = "gemini-1.5-flash"  # Options: gemini-1.5-flash, gemini-1.5-pro
MAX_TOKENS = 1500
TEMPERATURE = 0.7

# File Upload Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = ['.pdf', '.xlsx', '.xls', '.docx', '.txt']

# UI Configuration
PAGE_TITLE = "RAG System - Document Q&A"
PAGE_ICON = "📚"