"""
Test script to verify query timing logging.
Run this script to test the complete pipeline and see timing logs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.logging_utils import setup_logger, log_query_start, log_query_complete
from src.retrieval import load_retriever_from_disk
from src.generation import create_conversational_chain
from src.query_processor import QueryProcessor
import time
import uuid

# Initialize logger
logger = setup_logger('test')

def test_query_processing():
    """Test query processor timing"""
    print("\n" + "="*80)
    print("TEST 1: Query Processing")
    print("="*80)
    
    processor = QueryProcessor()
    test_question = "What are the symptoms of late blight in potatoes?"
    chat_history = [
        ("What is potato disease?", "Potato diseases are pathological conditions..."),
        ("Tell me more", "Sure! Potatoes can be affected by various pathogens...")
    ]
    
    result = processor.preprocess_question(test_question, chat_history)
    print(f"\nOriginal: {result['original_question']}")
    print(f"Standalone: {result['standalone_question']}")
    print(f"Enhanced: {result['enhanced_question']}")
    print(f"Expanded queries: {result['expanded_queries']}")


def test_retrieval():
    """Test retrieval timing"""
    print("\n" + "="*80)
    print("TEST 2: Retrieval Pipeline")
    print("="*80)
    
    retriever = load_retriever_from_disk()
    test_query = "How to control late blight fungal disease?"
    
    print(f"\nRetrieving documents for: {test_query}")
    docs = retriever.invoke(test_query)
    
    print(f"\nRetrieved {len(docs)} documents")
    if docs:
        print(f"First document preview: {docs[0].page_content[:150]}...")


def test_full_pipeline():
    """Test full RAG pipeline timing"""
    print("\n" + "="*80)
    print("TEST 3: Full RAG Pipeline")
    print("="*80)
    
    request_id = str(uuid.uuid4())[:8]
    test_question = "What are the best practices for managing early blight?"
    
    log_query_start(logger, request_id, test_question)
    
    start_time = time.perf_counter()
    
    try:
        # Load components
        retriever = load_retriever_from_disk()
        qa_chain = create_conversational_chain(retriever)
        
        # Test invocation
        print(f"\nProcessing: {test_question}")
        result = qa_chain.invoke({
            "question": test_question,
            "chat_history": []
        })
        
        elapsed = time.perf_counter() - start_time
        
        print(f"\nAnswer length: {len(result.get('answer', ''))} characters")
        print(f"Source documents: {len(result.get('source_documents', []))}")
        
        log_query_complete(logger, request_id, elapsed, "SUCCESS")
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"Test failed: {e}")
        log_query_complete(logger, request_id, elapsed, f"FAILED: {type(e).__name__}")


if __name__ == "__main__":
    print("\n" + "🥔 " * 20)
    print("QUERY TIMING LOG TEST SUITE")
    print("🥔 " * 20)
    
    print("\n📝 Logs will be saved to: logs/query_timing.log")
    print("📝 Make sure your .env file is configured with OpenAI API key")
    
    try:
        print("\n⏱️  Running tests...")
        
        # Test 1: Query processing
        test_query_processing()
        
        # Test 2: Retrieval
        test_retrieval()
        
        # Test 3: Full pipeline (requires API keys)
        # Uncomment if you want to test the full pipeline
        # test_full_pipeline()
        
        print("\n" + "="*80)
        print("✅ Tests completed! Check logs/query_timing.log for detailed timing information")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test execution failed")
