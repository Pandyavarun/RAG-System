import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import config
from document_processor import DocumentProcessor, TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline

class RAGSystem:
    """Main RAG system that orchestrates all components"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.text_chunker = TextChunker()
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize vector store with correct embedding dimension
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        self.vector_store_manager = VectorStoreManager(
            db_type=config.VECTOR_DB_TYPE,
            embedding_dim=embedding_dim
        )
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(
            vector_store_manager=self.vector_store_manager,
            embedding_generator=self.embedding_generator
        )
    
    def ingest_documents(self, uploaded_files) -> Dict[str, Any]:
        """Process and ingest uploaded documents"""
        results = {
            'processed_files': [],
            'total_chunks': 0,
            'errors': []
        }
        
        all_documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process the file
                processed_data = self.document_processor.process_file(tmp_file_path, uploaded_file.name)
                documents = processed_data['documents']
                
                # Chunk the documents
                chunked_documents = self.text_chunker.chunk_documents(documents)
                all_documents.extend(chunked_documents)
                
                results['processed_files'].append({
                    'name': uploaded_file.name,
                    'type': processed_data['source_type'],
                    'chunks': len(chunked_documents)
                })
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
            except Exception as e:
                results['errors'].append(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if all_documents:
            # Generate embeddings for all chunks
            texts = [doc['text'] for doc in all_documents]
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Store in vector database
            self.vector_store_manager.add_documents(all_documents, embeddings)
            
            results['total_chunks'] = len(all_documents)
        
        return results
    
    def query_documents(self, question: str, top_k: int = config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Query the RAG system"""
        return self.rag_pipeline.query(question, top_k)
    
    def get_multiple_answers(self, question: str, top_k: int = config.TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Get multiple answers from different sources"""
        return self.rag_pipeline.get_multiple_answers(question, top_k)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database"""
        return self.vector_store_manager.get_info()
    
    def clear_database(self):
        """Clear all documents from the database"""
        self.vector_store_manager.clear_database()

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout="wide"
    )
    
    # Check API key before initializing
    if not config.GOOGLE_API_KEY or config.GOOGLE_API_KEY == "your-google-api-key-here":
        st.error("🔑 **Google API Key Required!**")
        st.markdown("""
        Please set your Google Gemini API key:
        
        **Option 1: Environment Variable (Recommended)**
        ```bash
        # Windows PowerShell
        $env:GOOGLE_API_KEY="your-api-key-here"
        
        # Windows Command Prompt  
        set GOOGLE_API_KEY=your-api-key-here
        
        # Linux/Mac
        export GOOGLE_API_KEY="your-api-key-here"
        ```
        
        **Option 2: Direct Configuration**
        Edit `config.py` and replace the placeholder with your actual API key.
        
        **Get your API key**: [Google AI Studio](https://aistudio.google.com/app/apikey)
        """)
        st.stop()
    
    # Initialize RAG system
    try:
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = RAGSystem()
    except ValueError as e:
        st.error(f"🔑 **Configuration Error**: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ **Initialization Error**: {str(e)}")
        st.stop()
    
    rag_system = st.session_state.rag_system
    
    # Main title
    st.title("🔍 RAG System - Document Q&A")
    st.markdown("Upload documents and ask questions to get AI-powered answers with source attribution!")
    
    # Sidebar for configuration and database info
    with st.sidebar:
        st.header("📊 Database Info")
        try:
            db_info = rag_system.get_database_info()
            st.metric("Documents", db_info['document_count'])
            st.metric("Database Type", db_info['type'].upper())
        except Exception as e:
            st.error(f"Error accessing database: {str(e)}")
            st.metric("Documents", "Error")
            st.metric("Database Type", "ChromaDB")
        
        if st.button("🗑️ Clear Database", type="secondary"):
            with st.spinner("Clearing database..."):
                rag_system.clear_database()
            st.success("Database cleared!")
            # Force rerun to update the UI immediately
            st.rerun()
        
        st.header("⚙️ Settings")
        
        # Enhanced settings
        top_k = st.slider("Number of results", min_value=1, max_value=15, value=config.TOP_K_RESULTS)
        
        enable_fallback = st.checkbox("Enable AI fallback for missing info", value=config.ENABLE_FALLBACK_LLM, 
                                    help="Allow AI to answer questions even when documents don't contain sufficient information")
        
        if enable_fallback:
            confidence_threshold = st.slider("Confidence threshold for fallback", 
                                            min_value=0.0, max_value=1.0, 
                                            value=config.FALLBACK_CONFIDENCE_THRESHOLD, step=0.1,
                                            help="Below this confidence, AI will provide general knowledge answers")
        
        combine_knowledge = st.checkbox("Enhance with AI knowledge", value=config.COMBINE_RETRIEVAL_AND_LLM,
                                      help="Combine document information with AI knowledge for richer answers")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "❓ Ask Questions", "📋 Multiple Answers"])
    
    with tab1:
        st.header("Document Upload & Ingestion")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'xlsx', 'xls', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Excel, Word, Text files"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    results = rag_system.ingest_documents(uploaded_files)
                
                # Display results
                if results['processed_files']:
                    st.success(f"Successfully processed {len(results['processed_files'])} files!")
                    
                    # Show processing details
                    for file_info in results['processed_files']:
                        st.info(f"✅ {file_info['name']} ({file_info['type']}) - {file_info['chunks']} chunks")
                    
                    st.metric("Total chunks created", results['total_chunks'])
                
                # Show errors if any
                if results['errors']:
                    st.error("Some files had errors:")
                    for error in results['errors']:
                        st.error(error)
    
    with tab2:
        st.header("Ask Questions")
        
        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main topic discussed in the documents?",
            help="Ask any question about your uploaded documents"
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🔍 Get Answer", type="primary", disabled=not question):
                if question:
                    # Update config dynamically based on user settings
                    config.ENABLE_FALLBACK_LLM = enable_fallback
                    if enable_fallback:
                        config.FALLBACK_CONFIDENCE_THRESHOLD = confidence_threshold
                    config.COMBINE_RETRIEVAL_AND_LLM = combine_knowledge
                    
                    with st.spinner("Searching and generating answer..."):
                        response = rag_system.query_documents(question, top_k)
                    
                    # Display answer with response type indicator
                    st.subheader("💡 Answer")
                    
                    # Show response type
                    response_type = response.get('response_type', 'document_based')
                    if response_type == 'llm_generated':
                        st.info("🤖 **AI-Generated Response**: This answer is generated from the AI model's knowledge as no sufficiently relevant documents were found.")
                    elif response_type == 'document_based':
                        st.success("📚 **Document-Based Response**: This answer is primarily based on your uploaded documents.")
                    elif 'enhanced' in response.get('reasoning', '').lower():
                        st.info("🔄 **Enhanced Response**: This answer combines document information with additional AI knowledge.")
                    
                    st.write(response['answer'])
                    
                    # Display confidence and reasoning
                    col_conf, col_chunks = st.columns(2)
                    with col_conf:
                        st.metric("Confidence", f"{response['confidence']:.1%}")
                    with col_chunks:
                        st.metric("Sources Used", response['retrieved_chunks'])
                    
                    # Display reasoning
                    with st.expander("🧠 Reasoning"):
                        st.write(response['reasoning'])
                    
                    # Display sources
                    if response['sources']:
                        st.subheader("📚 Sources")
                        
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"Source {i} - Similarity: {source['similarity_score']:.1%}"):
                                # Display metadata
                                metadata = source['metadata']
                                
                                cols = st.columns(4)
                                with cols[0]:
                                    st.write(f"**File:** {metadata.get('source', 'Unknown')}")
                                with cols[1]:
                                    if 'page' in metadata:
                                        st.write(f"**Page:** {metadata['page']}")
                                    elif 'sheet' in metadata:
                                        st.write(f"**Sheet:** {metadata['sheet']}")
                                        st.write(f"**Row:** {metadata['row']}")
                                    elif 'paragraph' in metadata:
                                        st.write(f"**Paragraph:** {metadata['paragraph']}")
                                with cols[2]:
                                    if 'chunk' in metadata:
                                        st.write(f"**Chunk:** {metadata['chunk']}/{metadata.get('total_chunks', '?')}")
                                    if 'word_count' in metadata:
                                        st.write(f"**Words:** {metadata['word_count']}")
                                with cols[3]:
                                    st.write(f"**Similarity:** {source['similarity_score']:.1%}")
                                    if 'chunk_word_count' in metadata:
                                        st.write(f"**Chunk Words:** {metadata['chunk_word_count']}")
                                
                                # Display text content
                                st.write("**Content:**")
                                st.text_area("", value=source['text'], height=100, disabled=True, key=f"source_{i}")
        
        with col2:
            show_multiple = st.checkbox("Show multiple answers")
    
    with tab3:
        st.header("Multiple Answers from Different Sources")
        st.markdown("Get answers from different document sources to see various perspectives.")
        
        question_multi = st.text_input(
            "Enter your question for multiple answers:",
            placeholder="Compare information across different documents...",
            key="multi_question"
        )
        
        if st.button("🔍 Get Multiple Answers", type="primary", disabled=not question_multi):
            if question_multi:
                with st.spinner("Searching across multiple sources..."):
                    multiple_answers = rag_system.get_multiple_answers(question_multi, top_k)
                
                if multiple_answers:
                    for i, answer_data in enumerate(multiple_answers, 1):
                        with st.container():
                            st.subheader(f"Answer {i} from: {answer_data['source']}")
                            
                            # Answer and confidence
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.write(answer_data['answer'])
                            with col2:
                                st.metric("Confidence", f"{answer_data['confidence']:.1%}")
                            
                            # Reasoning and sources in expander
                            with st.expander(f"Details for Answer {i}"):
                                st.write("**Reasoning:**")
                                st.write(answer_data['reasoning'])
                                
                                st.write("**Source Details:**")
                                for j, source in enumerate(answer_data['sources'], 1):
                                    st.write(f"**Chunk {j}:** {source['text']}")
                                    st.caption(f"Similarity: {source['similarity_score']:.1%}")
                            
                            st.divider()
                else:
                    st.warning("No answers found across the document sources.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with Streamlit, LangChain, and Google Gemini • "
        f"Embedding Model: {config.EMBEDDING_MODEL} • "
        f"Vector DB: {config.VECTOR_DB_TYPE}"
    )

if __name__ == "__main__":
    main()