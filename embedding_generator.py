import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import config

class EmbeddingGenerator:
    """Handles text embedding generation using various models"""
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.is_google_model = "models/embedding" in model_name
        
        if self.is_google_model:
            genai.configure(api_key=config.GOOGLE_API_KEY)
        else:
            # Load local sentence transformer model
            self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if self.is_google_model:
            return self._generate_google_embeddings(texts)
        else:
            return self._generate_local_embeddings(texts)
    
    def _generate_google_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Generative AI API"""
        embeddings = []
        
        # Process in batches to handle rate limits
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                for text in batch:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embeddings.append(result['embedding'])
                
            except Exception as e:
                print(f"Error generating Google embeddings: {e}")
                # Fallback to zeros or handle error appropriately
                embeddings.extend([[0.0] * 768] * len(batch))  # Google embeddings are 768-dimensional
        
        return embeddings
    
    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local sentence transformer model"""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating local embeddings: {e}")
            # Return zero embeddings as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return [[0.0] * embedding_dim] * len(texts)
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if self.is_google_model:
            return 768  # Google embedding-001 dimension
        else:
            return self.model.get_sentence_embedding_dimension()