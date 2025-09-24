"""
RAG System - Retrieval-Augmented Generation for Document Q&A

A complete implementation of a RAG system that allows users to:
1. Upload and ingest documents (PDF, Excel, Word, Text)
2. Ask questions and get AI-powered answers
3. View source attribution and reasoning
4. Get multiple answers from different document sources

Usage:
    streamlit run app.py

Components:
- document_processor.py: Handles document ingestion and text extraction
- embedding_generator.py: Generates embeddings using various models
- vector_store.py: Vector database implementation (ChromaDB/FAISS)
- rag_pipeline.py: Main RAG pipeline for retrieval and generation
- app.py: Streamlit web interface
- config.py: Configuration settings

Requirements:
    pip install -r requirements.txt

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (required for GPT models)
"""

from app import main, RAGSystem
from document_processor import DocumentProcessor, TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStoreManager, ChromaDBVectorStore, FAISSVectorStore
from rag_pipeline import RAGPipeline
import config

__version__ = "1.0.0"
__author__ = "RAG System Developer"

# Make key classes available at package level
__all__ = [
    'RAGSystem',
    'DocumentProcessor',
    'TextChunker', 
    'EmbeddingGenerator',
    'VectorStoreManager',
    'ChromaDBVectorStore',
    'FAISSVectorStore',
    'RAGPipeline',
    'main',
    'config'
]