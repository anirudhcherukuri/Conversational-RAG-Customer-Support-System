import os
import sys
import logging
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

load_dotenv()

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_workflow():
    logger.info("Initializing RAG test workflow...")
    
    # Import inside function to avoid loading imports before sys.path update
    from rag_graph import rag_graph
    
    # 1. Define query and initial state
    test_state = {
        "session_id": "test_session_123",
        "question": "What is the return policy and payment methods accepted?",
        "chat_history": [],
        "raw_documents": [],
        "reranked_documents": [],
        "generation": "",
        "faithfulness_score": 0.0,
        "faithfulness_reason": "",
        "attempts": 0,
        "max_attempts": 2,
        "confidence_threshold": 0.4,
        "logs": []
    }
    
    logger.info(f"Running LangGraph pipeline with query: '{test_state['question']}'")
    
    # 2. Invoke graph
    try:
        final_state = rag_graph.invoke(test_state)
        
        logger.info("\n=== EXECUTION LOGS ===")
        for log in final_state.get("logs", []):
            logger.info(f"Log: {log}")
            
        logger.info("\n=== DENSE & BM25 RETRIEVED DOCUMENTS ===")
        for idx, doc in enumerate(final_state.get("raw_documents", [])):
            logger.info(f"Raw Doc {idx+1}: {doc['text'][:100]}... | RRF: {doc.get('rrf_score', 0):.4f}")
            
        logger.info("\n=== RERANKED DOCUMENTS ===")
        for idx, doc in enumerate(final_state.get("reranked_documents", [])):
            logger.info(f"Reranked Doc {idx+1}: {doc['text'][:100]}... | Score: {doc.get('rerank_score', 0):.4f}")
            
        logger.info("\n=== GENERATED RESPONSE ===")
        logger.info(final_state.get("generation"))
        
        logger.info("\n=== HALLUCINATION GUARDRAIL EVALUATION ===")
        logger.info(f"Faithfulness Score: {final_state.get('faithfulness_score')}")
        logger.info(f"Evaluation Reason: {final_state.get('faithfulness_reason')}")
        logger.info(f"Total Attempts: {final_state.get('attempts')}")
        
        # Simple validations
        assert len(final_state.get("raw_documents")) > 0, "Should retrieve raw documents"
        assert len(final_state.get("reranked_documents")) > 0, "Should rerank documents"
        assert len(final_state.get("generation")) > 0, "Should generate answer"
        assert final_state.get("attempts") >= 1, "Should have run at least 1 attempt"
        
        logger.info("\nWorkflow verification: SUCCESS!")
        
    except Exception as e:
        logger.error(f"Workflow test failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    test_workflow()
