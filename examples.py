"""
Example usage of the RAG system components
This file demonstrates how to use the RAG system programmatically
"""

import os
from app import RAGSystem
from document_processor import DocumentProcessor, TextChunker
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStoreManager
from rag_pipeline import RAGPipeline

def example_basic_usage():
    """Basic example of using the RAG system"""
    print("🚀 Basic RAG System Usage Example")
    print("=" * 50)
    
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    # Example: Process documents programmatically
    # (In practice, you'd have actual file paths)
    sample_documents = [
        {
            'text': "The company's revenue increased by 25% in Q3 2024, reaching $2.5 million.",
            'metadata': {
                'source': 'financial_report_q3.pdf',
                'page': 1,
                'type': 'pdf'
            }
        },
        {
            'text': "Our customer satisfaction score improved to 4.8/5.0 based on 1,200 survey responses.",
            'metadata': {
                'source': 'customer_survey_results.xlsx',
                'sheet': 'Summary',
                'row': 5,
                'type': 'excel'
            }
        }
    ]
    
    # Manually add documents (normally done through file upload)
    texts = [doc['text'] for doc in sample_documents]
    embeddings = rag_system.embedding_generator.generate_embeddings(texts)
    rag_system.vector_store_manager.add_documents(sample_documents, embeddings)
    
    print("✅ Documents added to vector store")
    
    # Query the system
    questions = [
        "What was the revenue increase in Q3?",
        "What is the customer satisfaction score?",
        "How many customers responded to the survey?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = rag_system.query_documents(question, top_k=3)
        
        print(f"💡 Answer: {response['answer']}")
        print(f"🎯 Confidence: {response['confidence']:.1%}")
        print(f"📚 Sources: {response['retrieved_chunks']} chunks")
        
        # Show source details
        for i, source in enumerate(response['sources'][:2], 1):
            metadata = source['metadata']
            print(f"   Source {i}: {metadata['source']} "
                  f"(Page {metadata.get('page', metadata.get('row', 'N/A'))})")

def example_component_usage():
    """Example of using individual components"""
    print("\n🔧 Individual Component Usage Example")
    print("=" * 50)
    
    # 1. Document Processing
    print("\n1. Document Processing:")
    processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    # Simulate document processing
    sample_text = """
    This is a sample document with multiple paragraphs.
    
    The first paragraph discusses the importance of data processing in modern applications.
    
    The second paragraph explains how machine learning models can benefit from proper data preparation.
    """
    
    mock_doc = {
        'text': sample_text,
        'metadata': {'source': 'sample.txt', 'type': 'txt', 'paragraph': 1}
    }
    
    chunked = chunker.chunk_documents([mock_doc])
    print(f"   Original text length: {len(sample_text)}")
    print(f"   Number of chunks: {len(chunked)}")
    
    # 2. Embedding Generation
    print("\n2. Embedding Generation:")
    embedding_gen = EmbeddingGenerator()
    
    texts = ["Sample text for embedding", "Another piece of text"]
    embeddings = embedding_gen.generate_embeddings(texts)
    
    print(f"   Generated embeddings for {len(texts)} texts")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    
    # 3. Vector Store Operations
    print("\n3. Vector Store Operations:")
    vector_store = VectorStoreManager(db_type="chromadb")
    
    # Add sample documents
    sample_docs = [
        {
            'text': 'Python is a programming language',
            'metadata': {'source': 'programming.txt', 'topic': 'python'}
        },
        {
            'text': 'Machine learning uses algorithms to find patterns',
            'metadata': {'source': 'ml_guide.pdf', 'topic': 'machine-learning'}
        }
    ]
    
    doc_texts = [doc['text'] for doc in sample_docs]
    doc_embeddings = embedding_gen.generate_embeddings(doc_texts)
    vector_store.add_documents(sample_docs, doc_embeddings)
    
    print(f"   Added {len(sample_docs)} documents to vector store")
    
    # Search the vector store
    query = "What is Python?"
    query_embedding = embedding_gen.generate_single_embedding(query)
    results = vector_store.search(query_embedding, top_k=2)
    
    print(f"   Search results for '{query}':")
    for i, (doc, score) in enumerate(results, 1):
        print(f"   {i}. Score: {score:.3f} - {doc['text'][:50]}...")

def example_advanced_usage():
    """Advanced usage patterns"""
    print("\n🎓 Advanced Usage Examples")
    print("=" * 50)
    
    # Initialize components with custom settings
    embedding_gen = EmbeddingGenerator("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = VectorStoreManager(db_type="chromadb", embedding_dim=384)
    rag_pipeline = RAGPipeline(vector_store, embedding_gen)
    
    # Example: Batch processing
    print("\n1. Batch Document Processing:")
    
    batch_documents = [
        {
            'text': f"This is document {i} with important information about topic {i}.",
            'metadata': {'source': f'doc_{i}.txt', 'topic': f'topic_{i}', 'batch': 1}
        }
        for i in range(1, 6)
    ]
    
    texts = [doc['text'] for doc in batch_documents]
    embeddings = embedding_gen.generate_embeddings(texts)
    vector_store.add_documents(batch_documents, embeddings)
    
    print(f"   Processed batch of {len(batch_documents)} documents")
    
    # Example: Multiple answer comparison
    print("\n2. Multiple Answer Comparison:")
    
    question = "What topics are covered?"
    multiple_answers = rag_pipeline.get_multiple_answers(question, top_k=3)
    
    print(f"   Question: {question}")
    for i, answer in enumerate(multiple_answers, 1):
        print(f"   Answer {i} (from {answer['source']}): {answer['answer'][:100]}...")
    
    # Example: Custom confidence thresholding
    print("\n3. Confidence-based Filtering:")
    
    response = rag_pipeline.query("Tell me about document 3")
    
    if response['confidence'] > 0.7:
        print(f"   High confidence answer ({response['confidence']:.1%}): {response['answer']}")
    elif response['confidence'] > 0.4:
        print(f"   Medium confidence answer ({response['confidence']:.1%}): {response['answer']}")
    else:
        print(f"   Low confidence ({response['confidence']:.1%}). Consider refining the question.")

def example_error_handling():
    """Example of proper error handling"""
    print("\n⚠️  Error Handling Examples")
    print("=" * 50)
    
    try:
        # Initialize with invalid configuration
        rag_system = RAGSystem()
        
        # Handle empty database queries
        response = rag_system.query_documents("What is the meaning of life?")
        
        if not response['sources']:
            print("   No relevant documents found in database")
            print(f"   Response: {response['answer']}")
        
        # Handle database info
        db_info = rag_system.get_database_info()
        print(f"   Database has {db_info['document_count']} documents")
        
        if db_info['document_count'] == 0:
            print("   💡 Tip: Upload some documents first!")
    
    except Exception as e:
        print(f"   Error occurred: {str(e)}")
        print("   💡 Check configuration and dependencies")

if __name__ == "__main__":
    """Run all examples"""
    
    print("🧪 RAG System - Usage Examples")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_component_usage() 
        example_advanced_usage()
        example_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("💡 Run 'streamlit run app.py' to start the web interface")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")