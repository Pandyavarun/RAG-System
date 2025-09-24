import google.generativeai as genai
from typing import List, Dict, Any, Tuple, Optional
import config

class RAGPipeline:
    """Main RAG pipeline that combines retrieval and generation"""
    
    def __init__(self, vector_store_manager, embedding_generator):
        self.vector_store = vector_store_manager
        self.embedding_generator = embedding_generator
        
        # Initialize Google Generative AI client
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.llm_model = config.LLM_MODEL
    
    def query(self, question: str, top_k: int = config.TOP_K_RESULTS, return_sources: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of relevant documents to retrieve
            return_sources: Whether to include source information
            
        Returns:
            Dictionary containing answer, sources, and reasoning
        """
        # Step 1: Generate embedding for the query
        query_embedding = self.embedding_generator.generate_single_embedding(question)
        
        # Step 2: Search vector database for relevant chunks
        search_results = self.vector_store.search(query_embedding, top_k)
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information in the uploaded documents to answer your question.",
                'sources': [],
                'confidence': 0.0,
                'reasoning': "No relevant documents found in the database."
            }
        
        # Step 3: Prepare context from retrieved chunks
        context_chunks = []
        source_info = []
        
        for doc, similarity_score in search_results:
            context_chunks.append(doc['text'])
            source_info.append({
                'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'metadata': doc['metadata'],
                'similarity_score': similarity_score
            })
        
        # Step 4: Generate response using LLM
        context = "\n\n".join([f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        response = self._generate_llm_response(question, context, source_info)
        
        # Step 5: Format and return results
        result = {
            'answer': response['answer'],
            'sources': source_info if return_sources else [],
            'confidence': self._calculate_confidence(search_results),
            'reasoning': response['reasoning'],
            'retrieved_chunks': len(search_results)
        }
        
        return result
    
    def _generate_llm_response(self, question: str, context: str, sources: List[Dict]) -> Dict[str, str]:
        """Generate response using LLM with retrieved context"""
        
        # Create system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on provided document context. 
        
        Instructions:
        1. Answer the question using ONLY the information provided in the context
        2. If the context doesn't contain enough information, say so clearly
        3. Cite specific document sections when possible
        4. Explain your reasoning and which parts of the context you used
        5. Be concise but comprehensive
        6. If multiple documents contain relevant information, synthesize the information appropriately
        """
        
        # Create user prompt with context
        user_prompt = f"""Context from retrieved documents:
        {context}
        
        Question: {question}
        
        Please provide:
        1. A clear answer to the question
        2. Explanation of how you derived this answer from the context
        3. Reference to specific document sections used"""
        
        try:
            model = genai.GenerativeModel(self.llm_model)
            
            # Combine system and user prompts for Gemini
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = model.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
            )
            
            full_response = response.text
            
            # Try to split answer and reasoning
            parts = full_response.split("Explanation:", 1)
            if len(parts) == 2:
                answer = parts[0].replace("Answer:", "").strip()
                reasoning = parts[1].strip()
            else:
                # If no clear split, use the full response as answer
                answer = full_response
                reasoning = "Answer generated from provided context documents."
            
            return {
                'answer': answer,
                'reasoning': reasoning
            }
            
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'reasoning': "Failed to process the question due to an error."
            }
    
    def _calculate_confidence(self, search_results: List[Tuple[Dict, float]]) -> float:
        """Calculate confidence score based on search results"""
        if not search_results:
            return 0.0
        
        # Average similarity score of top results
        scores = [score for _, score in search_results]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range (similarity scores should already be 0-1)
        confidence = max(0.0, min(1.0, avg_score))
        
        return round(confidence, 3)
    
    def get_multiple_answers(self, question: str, top_k: int = config.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """
        Get multiple potential answers from different document sources
        
        Returns top N answers with their source attribution
        """
        # Get more results than usual
        query_embedding = self.embedding_generator.generate_single_embedding(question)
        search_results = self.vector_store.search(query_embedding, top_k * 2)
        
        if not search_results:
            return []
        
        # Group results by source document
        source_groups = {}
        for doc, score in search_results:
            source_key = doc['metadata'].get('source', 'unknown')
            if source_key not in source_groups:
                source_groups[source_key] = []
            source_groups[source_key].append((doc, score))
        
        # Generate answer for each source group
        answers = []
        for source, docs in source_groups.items():
            # Take top chunks from this source
            top_docs = docs[:3]  # Top 3 chunks per source
            
            context_chunks = [doc['text'] for doc, _ in top_docs]
            context = "\n\n".join(context_chunks)
            
            # Generate response for this specific source
            source_info = [{
                'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'metadata': doc['metadata'],
                'similarity_score': score
            } for doc, score in top_docs]
            
            response = self._generate_llm_response(question, context, source_info)
            
            answers.append({
                'answer': response['answer'],
                'source': source,
                'sources': source_info,
                'confidence': self._calculate_confidence(top_docs),
                'reasoning': response['reasoning']
            })
        
        # Sort by confidence
        answers.sort(key=lambda x: x['confidence'], reverse=True)
        
        return answers[:top_k]  # Return top N answers