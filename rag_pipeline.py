import google.generativeai as genai
from typing import List, Dict, Any, Tuple, Optional
import config

class RAGPipeline:
    """Main RAG pipeline that combines retrieval and generation"""
    
    def __init__(self, vector_store_manager, embedding_generator):
        self.vector_store = vector_store_manager
        self.embedding_generator = embedding_generator
        
        # Initialize Google Generative AI client with validation
        if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your-google-api-key-here":
            raise ValueError("Google API key is required. Please set GOOGLE_API_KEY environment variable or update config.py. Get your key from: https://aistudio.google.com/app/apikey")
        
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.llm_model = config.LLM_MODEL
    
    def query(self, question: str, top_k: int = config.TOP_K_RESULTS, return_sources: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline with enhanced fallback
        
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
        
        # Calculate average confidence from search results
        avg_confidence = self._calculate_confidence(search_results)
        
        # Step 3: Determine response strategy based on results and confidence
        if not search_results or avg_confidence < config.FALLBACK_CONFIDENCE_THRESHOLD:
            if config.ENABLE_FALLBACK_LLM:
                return self._generate_llm_fallback_response(question, search_results, return_sources)
            else:
                return {
                    'answer': "I couldn't find sufficient relevant information in the uploaded documents to answer your question confidently.",
                    'sources': [],
                    'confidence': avg_confidence,
                    'reasoning': "No relevant documents found in the database or confidence too low.",
                    'response_type': 'insufficient_data'
                }
        
        # Step 4: Generate response using retrieval + LLM
        return self._generate_retrieval_response(question, search_results, return_sources)
    
    def _generate_llm_response(self, question: str, context: str, sources: List[Dict]) -> Dict[str, str]:
        """Generate enhanced response using LLM with retrieved context"""
        
        # Create enhanced system prompt
        system_prompt = """You are an expert AI assistant specializing in document analysis and question answering. Your task is to provide accurate, comprehensive answers based on the provided document context.

        Instructions:
        1. PRIMARILY use the information from the provided documents to answer the question
        2. Be specific about which documents/pages/sections you're referencing
        3. If the documents contain partial information, clearly state what is covered and what might be missing
        4. Provide comprehensive answers that synthesize information across multiple document sections when relevant
        5. If you need to make reasonable inferences based on the document content, clearly indicate this
        6. Structure your response clearly with proper organization
        7. Always maintain accuracy - don't make up information not present in the documents
        
        Response Format:
        - Provide a direct, comprehensive answer to the question
        - Use specific references to document sources (e.g., "According to page 3 of the financial report...")
        - If multiple documents contain relevant information, synthesize appropriately
        """
        
        # Create enhanced user prompt
        source_summary = []
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources for clarity
            metadata = source['metadata']
            source_desc = f"Source {i}: {metadata.get('source', 'Unknown')}"
            if 'page' in metadata:
                source_desc += f" (Page {metadata['page']})"
            elif 'sheet' in metadata and 'row' in metadata:
                source_desc += f" (Sheet: {metadata['sheet']}, Row: {metadata['row']})"
            source_summary.append(source_desc)
        
        user_prompt = f"""Document Context:
{context}

Source Summary:
{chr(10).join(source_summary)}

Question: {question}

Please provide a comprehensive answer that:
1. Directly addresses the question using the document content
2. References specific sources where information was found
3. Explains your reasoning process
4. Indicates if any important information might be missing from the provided documents

Your response should be informative, well-structured, and clearly cite the document sources used."""
        
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
                'reasoning': "Failed to process the question due to an error.",
                'response_type': 'error'
            }
    
    def _generate_llm_fallback_response(self, question: str, search_results: List[Tuple[Dict, float]], return_sources: bool) -> Dict[str, Any]:
        """Generate fallback response using only LLM when documents are insufficient"""
        
        # Create fallback prompt
        fallback_prompt = f"""You are a helpful AI assistant. A user has asked you a question, but the relevant documents in the database either don't contain sufficient information or have low relevance scores.

Question: {question}

Please provide a helpful answer based on your general knowledge. Be clear that this answer is generated from your training data rather than from the uploaded documents.

If you believe the question requires specific document content to answer accurately, explain that you would need more relevant documents to provide a complete answer.

Provide a clear, informative response while being transparent about your information source."""
        
        try:
            model = genai.GenerativeModel(self.llm_model)
            response = model.generate_content(
                fallback_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
            )
            
            answer = response.text
            reasoning = "LLM-generated response: No sufficiently relevant documents found in the database. Answer generated from AI model's training data."
            
            # Include low-confidence sources if any exist
            source_info = []
            if search_results and return_sources:
                source_info = [{
                    'content': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                    'filename': doc['metadata'].get('source', 'Unknown'),
                    'page': doc['metadata'].get('page', 'N/A'),
                    'similarity': score,
                    'word_count': doc['metadata'].get('word_count', doc['metadata'].get('chunk_word_count', 'N/A')),
                    'char_count': doc['metadata'].get('char_count', doc['metadata'].get('chunk_char_count', 'N/A')),
                    'metadata': doc['metadata']
                } for doc, score in search_results[:3]]  # Include top 3 even if low confidence
            
            return {
                'answer': answer,
                'sources': source_info,
                'confidence': self._calculate_confidence(search_results),
                'reasoning': reasoning,
                'response_type': 'llm_generated',
                'retrieved_chunks': len(search_results)
            }
            
        except Exception as e:
            return {
                'answer': f"I couldn't find relevant information in the documents, and encountered an error generating a general response: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'reasoning': "Error in fallback LLM response generation.",
                'response_type': 'error'
            }
    
    def _generate_retrieval_response(self, question: str, search_results: List[Tuple[Dict, float]], return_sources: bool) -> Dict[str, Any]:
        """Generate response using retrieved documents"""
        
        # Prepare context from retrieved chunks
        context_chunks = []
        source_info = []
        
        for doc, similarity_score in search_results:
            context_chunks.append(doc['text'])
            source_info.append({
                'content': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'filename': doc['metadata'].get('source', 'Unknown'),
                'page': doc['metadata'].get('page', 'N/A'),
                'similarity': similarity_score,
                'word_count': doc['metadata'].get('word_count', doc['metadata'].get('chunk_word_count', 'N/A')),
                'char_count': doc['metadata'].get('char_count', doc['metadata'].get('chunk_char_count', 'N/A')),
                'metadata': doc['metadata']
            })
        
        # Generate response using LLM with context
        context = "\n\n".join([f"[Document {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)])
        
        response = self._generate_llm_response(question, context, source_info)
        
        # Determine if we should combine with LLM knowledge
        confidence = self._calculate_confidence(search_results)
        if config.COMBINE_RETRIEVAL_AND_LLM and confidence < 0.8:
            response = self._enhance_with_llm_knowledge(question, response, context)
        
        # Format and return results
        result = {
            'answer': response['answer'],
            'sources': source_info if return_sources else [],
            'confidence': confidence,
            'reasoning': response['reasoning'],
            'response_type': 'document_based',
            'retrieved_chunks': len(search_results)
        }
        
        return result
    
    def _enhance_with_llm_knowledge(self, question: str, initial_response: Dict[str, str], context: str) -> Dict[str, str]:
        """Enhance document-based response with additional LLM knowledge"""
        
        enhancement_prompt = f"""You have generated an initial answer based on provided documents, but the confidence is moderate. Please enhance this answer with additional relevant information from your general knowledge while clearly distinguishing between document-based and general knowledge.

Question: {question}

Initial answer from documents: {initial_response['answer']}

Document context: {context[:1000]}...

Please provide:
1. An enhanced answer that combines the document information with additional relevant context
2. Clear indication of what comes from documents vs. general knowledge
3. Enhanced reasoning that explains both sources of information

Format your response as a comprehensive answer that's more helpful to the user."""

        try:
            model = genai.GenerativeModel(self.llm_model)
            response = model.generate_content(
                enhancement_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE * 0.8  # Slightly less creative for enhancement
                )
            )
            
            enhanced_answer = response.text
            enhanced_reasoning = f"Enhanced response: Combined information from uploaded documents with additional context from AI knowledge base. {initial_response['reasoning']}"
            
            return {
                'answer': enhanced_answer,
                'reasoning': enhanced_reasoning
            }
            
        except Exception as e:
            # Fallback to original response if enhancement fails
            return initial_response
    
    def _calculate_confidence(self, search_results: List[Tuple[Dict, float]]) -> float:
        """Calculate enhanced confidence score based on search results"""
        if not search_results:
            return 0.0
        
        scores = [score for _, score in search_results]
        
        # Calculate base confidence (average of top scores)
        avg_score = sum(scores) / len(scores)
        
        # Boost confidence if we have multiple high-quality results
        if len(scores) >= 3:
            top_3_avg = sum(scores[:3]) / 3
            if top_3_avg > 0.7:
                avg_score = min(1.0, avg_score * 1.1)  # 10% boost for multiple good results
        
        # Penalty for very few results
        if len(scores) == 1 and scores[0] < 0.6:
            avg_score *= 0.8  # 20% penalty for single low-confidence result
        
        # Normalize to 0-1 range
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
                'content': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'filename': doc['metadata'].get('source', 'Unknown'),
                'page': doc['metadata'].get('page', 'N/A'),
                'similarity': score,
                'word_count': doc['metadata'].get('word_count', doc['metadata'].get('chunk_word_count', 'N/A')),
                'char_count': doc['metadata'].get('char_count', doc['metadata'].get('chunk_char_count', 'N/A')),
                'metadata': doc['metadata']
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