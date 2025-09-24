# RAG System - Enhanced Document Q&A with AI Fallback

A comprehensive Retrieval-Augmented Generation (RAG) system for document Q&A with intelligent fallback, enhanced PDF processing, and transparent reasoning.

## 🚀 Key Features

### 📚 **Enhanced Document Processing**
- **Multi-format Support**: PDF, Excel (.xlsx/.xls), Word (.docx), Plain Text (.txt)
- **Intelligent PDF Processing**: 
  - Advanced text cleaning and artifact removal
  - Smart chunking with sentence/paragraph boundary detection
  - Minimum chunk size filtering for quality control
  - Rich metadata tracking (word counts, page numbers, chunks)
- **Excel Processing**: Row-level extraction with sheet and column information
- **Word Processing**: Paragraph-level extraction with structure preservation

### 🧠 **Smart AI Integration**
- **Dual Response Strategy**:
  - **Document-based responses** when relevant information is found
  - **AI-generated fallback** when documents lack sufficient information
  - **Enhanced responses** combining document info with AI knowledge
- **Confidence-based Switching**: Automatically determines best response strategy
- **Transparent Attribution**: Clear indication of information sources

### 🔍 **Advanced Retrieval & Generation**
- **Flexible Embedding Models**: 
  - Local models via SentenceTransformers (default: all-MiniLM-L6-v2)
  - Google's embedding models (optional)
- **Dual Vector Databases**:
  - ChromaDB (persistent, production-ready)
  - FAISS (fast similarity search)
- **Enhanced Confidence Scoring**: Multi-factor confidence calculation
- **Configurable Retrieval**: Adjustable similarity thresholds and result counts

### 🎯 **Intelligent Response System**
- **Three Response Modes**:
  1. **Document-Based** (High Confidence >40%): Uses uploaded documents
  2. **Enhanced** (Medium Confidence): Combines documents + AI knowledge  
  3. **AI-Generated** (Low Confidence <40%): Falls back to AI knowledge
- **Transparent Reasoning**: Shows exactly how answers were generated
- **Source Attribution**: Precise references to pages, rows, or paragraphs
- **Quality Assurance**: Minimum chunk sizes and confidence thresholds

### 🎨 **Enhanced Web Interface**
- **User-Friendly UI**: Built with Streamlit
- **Real-time Processing**: Upload and query documents instantly
- **Dynamic Configuration**: Adjust settings in real-time
- **Rich Source Visualization**: View original text chunks with metadata
- **Response Type Indicators**: Clear distinction between response types
- **Advanced Settings**: Confidence thresholds, fallback options, knowledge enhancement

## 📋 System Requirements

```
Python 3.8+
Google API Key (for Gemini)
4GB+ RAM recommended
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure Google Gemini API**:

   **Option A: Environment Variable (Recommended)**
   ```bash
   # Windows (PowerShell)
   $env:GOOGLE_API_KEY="your-google-api-key-here"
   
   # Windows (Command Prompt)
   set GOOGLE_API_KEY=your-google-api-key-here
   
   # Linux/Mac
   export GOOGLE_API_KEY="your-google-api-key-here"
   ```
   
   **Option B: Direct Configuration**
   Edit `config.py` and replace the placeholder:
   ```python
   GOOGLE_API_KEY = "your-actual-google-api-key-here"
   ```

4. **Get your Google API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy the key and use it in step 3

## 🚀 Usage

### Start the Application
```bash
streamlit run app.py
```

### Using the System

#### 1. Document Upload & Ingestion
- Navigate to the "Upload Documents" tab
- Select one or more files (PDF, Excel, Word, Text)
- Click "Process Documents"
- System will:
  - Extract text from documents
  - Split into optimized chunks
  - Generate embeddings
  - Store in vector database

#### 2. Ask Questions
- Go to "Ask Questions" tab
- Enter your question in natural language
- Click "Get Answer"
- Review:
  - AI-generated answer
  - Confidence score
  - Reasoning explanation
  - Source documents with page/row references

#### 3. Multiple Answers
- Use "Multiple Answers" tab for complex queries
- Get perspectives from different document sources
- Compare information across multiple files

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │    │   Text Chunks    │    │   Embeddings    │
│  (PDF/Excel/    │───▶│  (Smart Split)   │───▶│  (Vectors)      │
│   Word/Text)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Final Answer  │    │   LLM Response   │    │  Vector Store   │
│  + Sources +    │◀───│   (OpenAI GPT)   │◀───│ (ChromaDB/FAISS)│
│   Reasoning     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
RAG/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── document_processor.py # Document ingestion & text extraction
├── embedding_generator.py # Embedding generation
├── vector_store.py       # Vector database implementations
├── rag_pipeline.py       # Main RAG logic
├── requirements.txt      # Python dependencies
├── __init__.py          # Package initialization
└── README.md            # This file
```

## ⚙️ Configuration

Edit `config.py` to customize system behavior:

### Core Settings
```python
# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Local
# EMBEDDING_MODEL = "models/embedding-001"  # Google (requires API key)

# Vector Database
VECTOR_DB_TYPE = "chromadb"  # or "faiss"

# LLM Model
LLM_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"
```

### Enhanced Processing
```python
# Advanced Chunking
CHUNK_SIZE = 800  # Smaller chunks for better precision
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 100  # Filter tiny fragments

# Retrieval Settings
TOP_K_RESULTS = 7  # More context for better answers
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity

# AI Fallback System
ENABLE_FALLBACK_LLM = True  # Enable AI fallback
FALLBACK_CONFIDENCE_THRESHOLD = 0.4  # Threshold for fallback
COMBINE_RETRIEVAL_AND_LLM = True  # Enhance with AI knowledge
```

## 🎯 Use Cases

### Business Documents
- Policy manuals and procedures
- Financial reports and statements
- Meeting minutes and presentations
- Technical documentation

### Research & Academic
- Research papers and journals
- Literature reviews
- Academic datasets
- Reference materials

### Legal & Compliance
- Contracts and agreements
- Regulatory documents
- Compliance manuals
- Legal precedents

## 🔧 Advanced Features

### Custom Embedding Models
```python
# Use different embedding models
embedding_generator = EmbeddingGenerator("sentence-transformers/paraphrase-MiniLM-L6-v2")
```

### Vector Database Selection
```python
# Choose vector database
vector_store = VectorStoreManager(db_type="faiss", embedding_dim=384)
```

### Batch Processing
```python
# Process multiple files programmatically
rag_system = RAGSystem()
results = rag_system.ingest_documents(file_list)
```

## 📊 Performance

- **Embedding Generation**: ~100-1000 docs/sec (local model)
- **Vector Search**: Sub-second retrieval for 100K+ documents
- **Memory Usage**: ~1GB for 10K documents (with embeddings)
- **Storage**: ~10MB per 1K documents (compressed)

## 🤝 Contributing

Feel free to contribute by:
- Adding new document format support
- Implementing additional vector databases
- Improving chunking strategies
- Enhancing the UI/UX
- Adding new embedding models

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **Google Gemini API Key Error**
   - Ensure your API key is set as environment variable or in `config.py`
   - Check API key validity at [Google AI Studio](https://aistudio.google.com/)
   - Verify quota limits and billing status

2. **"Clear Database" Error**
   - This is now fixed in the latest version
   - If you still see errors, restart the application
   - Check that `vector_db` directory has write permissions

3. **Memory Issues with Large Documents**
   - Reduce `CHUNK_SIZE` to 600 or lower
   - Increase `MIN_CHUNK_SIZE` to filter small fragments
   - Use FAISS instead of ChromaDB for large datasets
   - Process documents in smaller batches

4. **Poor Answer Quality**
   - Increase `TOP_K_RESULTS` to 10+ for more context
   - Lower `FALLBACK_CONFIDENCE_THRESHOLD` to use documents more often
   - Enable `COMBINE_RETRIEVAL_AND_LLM` for enhanced answers
   - Check PDF text quality after upload

5. **Slow Performance**
   - Use local embedding models for faster processing
   - Reduce `MAX_TOKENS` for faster generation
   - Consider using `gemini-1.5-flash` instead of `gemini-1.5-pro`

6. **Import/Dependencies Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)
   - Try creating a new virtual environment

## � Security

- **API Key Safety**: Never commit API keys to version control
- **Environment Variables**: Use environment variables for sensitive data
- **Local Processing**: Documents are processed locally and not sent to external services (except for LLM queries)
- **Data Privacy**: Vector embeddings and document chunks are stored locally

## �📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review configuration settings in `config.py`
3. Ensure all dependencies are properly installed
4. Verify your Google API key is valid and has sufficient quota

---

**Built with ❤️ using Streamlit, LangChain, ChromaDB, and Google Gemini**

*Enhanced with intelligent fallback, advanced PDF processing, and transparent AI reasoning*