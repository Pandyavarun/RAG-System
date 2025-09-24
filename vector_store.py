import os
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
#import chromadb
import numpy as np
import faiss
import config

class VectorDatabase(ABC):
    """Abstract base class for vector databases"""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        pass

class ChromaDBVectorStore(VectorDatabase):
    """ChromaDB vector database implementation"""
    
    def __init__(self, persist_directory: str = config.VECTOR_DB_PATH, collection_name: str = "documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize collection using helper method
        self._initialize_collection()
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents with embeddings to the vector store"""
        ids = [f"doc_{i}_{hash(doc['text'][:100])}" for i, doc in enumerate(documents)]
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            doc_data = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
            documents.append((doc_data, similarity_score))
        
        return documents
    
    def delete_collection(self, collection_name: str = None) -> None:
        """Delete a collection and reinitialize"""
        if collection_name is None:
            collection_name = self.collection_name
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            # Collection might not exist, which is fine
            pass
        
        # Reinitialize the collection after deletion
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get collection"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection with error handling"""
        try:
            count = self.collection.count()
        except Exception as e:
            # Collection might not exist, reinitialize it
            self._initialize_collection()
            try:
                count = self.collection.count()
            except Exception:
                count = 0
        
        return {
            'name': self.collection_name,
            'document_count': count,
            'type': 'chromadb'
        }

class FAISSVectorStore(VectorDatabase):
    """FAISS vector database implementation"""
    
    def __init__(self, persist_directory: str = config.VECTOR_DB_PATH, embedding_dim: int = 384):
        self.persist_directory = persist_directory
        self.embedding_dim = embedding_dim
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.index_path = os.path.join(persist_directory, "faiss_index.index")
        self.metadata_path = os.path.join(persist_directory, "metadata.pkl")
        
        # Initialize or load FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.documents = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
            self.documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents with embeddings to the FAISS index"""
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store document metadata
        self.documents.extend(documents)
        
        # Persist to disk
        self._save_index()
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        # Search
        similarities, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                similarity_score = float(similarities[0][i])
                results.append((self.documents[idx], similarity_score))
        
        return results
    
    def delete_collection(self, collection_name: str = None) -> None:
        """Delete the FAISS index and metadata, then reinitialize"""
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
        except Exception:
            pass
            
        try:
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
        except Exception:
            pass
        
        # Reinitialize empty index and documents
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.documents, f)
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection with error handling"""
        try:
            document_count = len(self.documents)
        except Exception:
            document_count = 0
            
        return {
            'name': 'faiss_collection',
            'document_count': document_count,
            'type': 'faiss'
        }

class VectorStoreManager:
    """Manager class for vector database operations"""
    
    def __init__(self, db_type: str = config.VECTOR_DB_TYPE, embedding_dim: int = 384):
        self.db_type = db_type
        
        if db_type.lower() == "chromadb":
            self.vector_store = ChromaDBVectorStore()
        elif db_type.lower() == "faiss":
            self.vector_store = FAISSVectorStore(embedding_dim=embedding_dim)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add documents to the vector store"""
        return self.vector_store.add_documents(documents, embeddings)
    
    def search(self, query_embedding: List[float], top_k: int = config.TOP_K_RESULTS) -> List[Tuple[Dict[str, Any], float]]:
        """Search the vector store"""
        return self.vector_store.search(query_embedding, top_k)
    
    def clear_database(self) -> None:
        """Clear all data from the vector database"""
        self.vector_store.delete_collection()
        # Force a fresh info query to update the UI immediately
        return self.get_info()
    
    def get_info(self) -> Dict[str, Any]:
        """Get database information"""
        return self.vector_store.get_collection_info()