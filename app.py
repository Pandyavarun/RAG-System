import streamlit as st
import os
import sys
from typing import Dict, Any, List
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import project modules
try:
    from config import *
    from document_processor import DocumentProcessor
    from embedding_generator import EmbeddingGenerator
    from vector_store import VectorStoreManager
    from rag_pipeline import RAGPipeline
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.info("Please ensure all required files are in the project directory.")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_sqlite_compatibility():
    """Check SQLite version and provide guidance"""
    try:
        import sqlite3
        sqlite_version = sqlite3.sqlite_version
        version_parts = [int(x) for x in sqlite_version.split('.')]
        
        # Check if SQLite version is >= 3.35.0
        if version_parts[0] > 3 or (version_parts[0] == 3 and version_parts[1] >= 35):
            return True, sqlite_version
        else:
            return False, sqlite_version
    except Exception as e:
        return False, f"Error: {e}"

def get_optimal_vector_db():
    """Determine optimal vector database based on environment"""
    # Check if we're running on Streamlit Cloud
    if os.getenv('STREAMLIT_SHARING_MODE') or os.getenv('STREAMLIT_SERVER_HEADLESS'):
        logger.info("Detected Streamlit Cloud deployment, using FAISS")
        return "faiss"
    
    # Check SQLite compatibility for ChromaDB
    is_compatible, version_info = check_sqlite_compatibility()
    if not is_compatible:
        logger.warning(f"SQLite version {version_info} is not compatible with ChromaDB. Using FAISS instead.")
        return "faiss"
    
    # Use configured default
    return VECTOR_DB_TYPE

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with optimal vector database"""
        try:
            # Determine optimal vector database
            self.vector_db_type = get_optimal_vector_db()
            
            # Initialize components
            self.document_processor = DocumentProcessor()
            self.embedding_generator = EmbeddingGenerator()
            self.vector_store_manager = VectorStoreManager(db_type=self.vector_db_type)
            self.rag_pipeline = RAGPipeline(
                embedding_generator=self.embedding_generator,
                vector_store_manager=self.vector_store_manager
            )
            
            st.success(f"✅ RAG System initialized with {self.vector_db_type.upper()} vector database")
            
        except Exception as e:
            st.error(f"❌ Error initializing RAG system: {e}")
            if "sqlite3" in str(e).lower():
                st.error("🔧 **SQLite Compatibility Issue Detected**")
                st.info("""
                **For Streamlit Cloud Deployment:**
                1. The system has automatically switched to FAISS vector database
                2. This provides better compatibility with cloud environments
                3. All functionality remains the same
                
                **For Local Development:**
                - Update your SQLite version to 3.35.0 or higher
                - Or continue using FAISS (recommended for cloud deployment)
                """)
            raise e

    def process_documents(self, uploaded_files) -> bool:
        """Process uploaded documents"""
        try:
            all_chunks = []
            for uploaded_file in uploaded_files:
                # Process each document
                chunks = self.document_processor.process_file(uploaded_file)
                all_chunks.extend(chunks)
            
            if all_chunks:
                # Generate embeddings and store
                embeddings = self.embedding_generator.generate_embeddings([chunk['content'] for chunk in all_chunks])
                self.vector_store_manager.add_documents(all_chunks, embeddings)
                return True
            return False
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            return False

    def query(self, question: str, similarity_threshold: float = None, top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            threshold = similarity_threshold or FALLBACK_CONFIDENCE_THRESHOLD
            k = top_k or TOP_K_RESULTS
            
            return self.rag_pipeline.query(
                question=question,
                top_k=k,
                similarity_threshold=threshold
            )
        except Exception as e:
            st.error(f"Error during query: {e}")
            return {"error": str(e)}

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database"""
        try:
            info = self.vector_store_manager.get_info()
            info['vector_db_type'] = self.vector_db_type
            return info
        except Exception as e:
            logger.warning(f"Could not get database info: {e}")
            return {
                'document_count': 0,
                'type': self.vector_db_type,
                'status': 'Error',
                'vector_db_type': self.vector_db_type
            }

    def clear_database(self):
        """Clear all documents from the database"""
        try:
            self.vector_store_manager.clear_all()
            return True
        except Exception as e:
            st.error(f"Error clearing database: {e}")
            return False

def main():
    """Main Streamlit application"""
    
    # Check API key
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == 'your-google-api-key-here':
        st.error("🔑 **Google API Key Required**")
        st.markdown("""
        **To use this application, you need a Google API Key:**
        
        1. **Get API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. **Set Environment Variable**: 
           ```bash
           # Method 1: Environment Variable (Recommended)
           export GOOGLE_API_KEY="your-actual-api-key"
           
           # Method 2: Streamlit Secrets (for deployment)
           # Add to .streamlit/secrets.toml:
           # GOOGLE_API_KEY = "your-actual-api-key"
           ```
        3. **For Streamlit Cloud**: Add your API key in the app's secrets management
        """)
        st.stop()

    # Initialize RAG system
    try:
        if 'rag_system' not in st.session_state:
            with st.spinner("Initializing RAG System..."):
                st.session_state.rag_system = RAGSystem()
        rag_system = st.session_state.rag_system
    except Exception as e:
        st.error(f"Failed to initialize RAG System: {e}")
        st.stop()

    # Main header
    st.title("🤖 Enhanced RAG System with AI Fallback")
    st.markdown("**Upload documents, ask questions, get AI-powered answers with source attribution**")

    # Sidebar for configuration and database info
    with st.sidebar:
        st.header("📊 Database Info")
        db_info = rag_system.get_database_info()
        st.metric("Documents", db_info.get('document_count', 0))
        st.metric("Database Type", db_info.get('vector_db_type', 'Unknown').upper())
        
        # Display SQLite compatibility status
        is_compatible, version_info = check_sqlite_compatibility()
        if not is_compatible:
            st.warning("⚠️ SQLite compatibility issue detected")
            st.info(f"SQLite version: {version_info}")
            st.success("✅ Using FAISS (cloud-compatible)")
        else:
            st.success(f"✅ SQLite {version_info} compatible")
        
        st.header("⚙️ Configuration")
        
        # Dynamic configuration
        similarity_threshold = st.slider(
            "Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=FALLBACK_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Lower values include more results, higher values are more selective"
        )
        
        top_k_results = st.slider(
            "Max Results", 
            min_value=1, 
            max_value=10, 
            value=TOP_K_RESULTS,
            help="Number of document chunks to retrieve"
        )
        
        enable_ai_fallback = st.checkbox(
            "Enable AI Fallback", 
            value=ENABLE_FALLBACK_LLM,
            help="Use AI knowledge when documents don't contain sufficient information"
        )
        
        # Database management
        st.header("🗃️ Database Management")
        if st.button("🗑️ Clear Database", use_container_width=True):
            with st.spinner("Clearing database..."):
                if rag_system.clear_database():
                    st.success("✅ Database cleared successfully!")
                    st.rerun()

    # Main content tabs
    tab1, tab2 = st.tabs(["📤 Upload & Query", "📋 System Status"])
    
    with tab1:
        # File upload section
        st.header("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'xlsx', 'xls', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Excel (.xlsx, .xls), Word (.docx), Text (.txt)"
        )
        
        if uploaded_files:
            if st.button("📊 Process Documents", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                    progress_bar = st.progress(0)
                    
                    success = rag_system.process_documents(uploaded_files)
                    progress_bar.progress(100)
                    
                    if success:
                        st.success(f"✅ Successfully processed {len(uploaded_files)} document(s)!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to process documents. Please check the files and try again.")

        # Query section
        st.header("❓ Ask Questions")
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            help="Ask any question about the uploaded documents"
        )
        
        if question and st.button("🔍 Search", use_container_width=True):
            with st.spinner("Searching for answers..."):
                response = rag_system.query(
                    question=question,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k_results
                )
                
                if "error" in response:
                    st.error(f"❌ Error: {response['error']}")
                else:
                    # Display response
                    st.subheader("💡 Answer")
                    
                    # Response type indicator
                    response_type = response.get('response_type', 'unknown')
                    if response_type == 'document_based':
                        st.success("📚 **Document-based Answer**")
                    elif response_type == 'ai_generated':
                        st.info("🤖 **AI-generated Answer** (information not found in documents)")
                    elif response_type == 'enhanced':
                        st.warning("🔄 **Enhanced Answer** (document + AI knowledge)")
                    
                    # Main answer
                    st.write(response['answer'])
                    
                    # Confidence and reasoning
                    if 'confidence' in response:
                        st.metric("Confidence", f"{response['confidence']:.1%}")
                    
                    if 'reasoning' in response:
                        with st.expander("🧠 Reasoning"):
                            st.write(response['reasoning'])
                    
                    # Sources
                    if response.get('sources'):
                        st.subheader("📖 Sources")
                        for i, source in enumerate(response['sources'], 1):
                            with st.expander(f"Source {i} - {source.get('filename', 'Unknown')}"):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Similarity", f"{source.get('similarity', 0):.1%}")
                                with col2:
                                    st.metric("Page", source.get('page', 'N/A'))
                                with col3:
                                    st.metric("Words", source.get('word_count', 'N/A'))
                                with col4:
                                    st.metric("Characters", source.get('char_count', 'N/A'))
                                
                                st.text_area("Content", source.get('content', ''), height=100, disabled=True)

    with tab2:
        st.header("📋 System Status")
        
        # Environment information
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔧 Environment")
            st.write(f"**Vector Database**: {rag_system.vector_db_type.upper()}")
            st.write(f"**LLM Model**: {LLM_MODEL}")
            st.write(f"**Embedding Model**: {EMBEDDING_MODEL}")
            
            # SQLite status
            is_compatible, version_info = check_sqlite_compatibility()
            st.write(f"**SQLite Version**: {version_info}")
            st.write(f"**ChromaDB Compatible**: {'✅ Yes' if is_compatible else '❌ No'}")
            
        with col2:
            st.subheader("⚙️ Configuration")
            st.write(f"**Chunk Size**: {CHUNK_SIZE} characters")
            st.write(f"**Chunk Overlap**: {CHUNK_OVERLAP} characters")
            st.write(f"**Default Top-K**: {TOP_K_RESULTS} results")
            st.write(f"**Similarity Threshold**: {SIMILARITY_THRESHOLD}")
        
        # Database statistics
        st.subheader("📊 Database Statistics")
        db_info = rag_system.get_database_info()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", db_info.get('document_count', 0))
        with col2:
            st.metric("Database Type", db_info.get('type', 'Unknown').upper())
        with col3:
            st.metric("Status", db_info.get('status', 'Unknown'))

    # Footer
    st.markdown("---")
    st.markdown(
        "**Enhanced RAG System** | "
        f"Powered by Google Gemini & {rag_system.vector_db_type.upper()} | "
        "🔗 [GitHub Repository](#)"
    )

if __name__ == "__main__":
    main()