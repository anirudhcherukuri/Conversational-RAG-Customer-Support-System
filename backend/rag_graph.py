import os
import json
import logging
from typing import List, Dict, Any, TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Import Chroma and embeddings
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# Load .env file from the backend directory specifically
backend_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize ONNX-based embedding function (CPU-friendly, lightweight, no PyTorch)
try:
    logger.info("Initializing ONNX embedding model...")
    onnx_ef = ONNXMiniLM_L6_V2()
    logger.info("ONNX embedding model initialized.")
except Exception as e:
    logger.error(f"Failed to initialize ONNX embedding model: {e}")
    onnx_ef = None

# Initialize Groq LLM
# Make sure GROQ_API_KEY is in the environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.warning("GROQ_API_KEY not found in environment variables. Make sure to set it.")

# Default client for Chroma
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Retrieve default collection or create one
default_collection = chroma_client.get_or_create_collection(
    name="default_faq",
    embedding_function=onnx_ef
)

# Add some seed/default FAQ documents if default_collection is empty
if default_collection.count() == 0:
    logger.info("Initializing default FAQ collection...")
    default_faqs = [
        {
            "id": "faq_1",
            "text": "The returns policy allows customers to return any unused, unopened products within 30 days of purchase for a full refund. Shipping costs for returns are the responsibility of the customer unless the item was damaged or defective upon arrival.",
            "metadata": {"source": "returns_policy.txt", "category": "returns"}
        },
        {
            "id": "faq_2",
            "text": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Orders over $50 qualify for free standard shipping. Orders are processed within 24 hours on weekdays.",
            "metadata": {"source": "shipping_policy.txt", "category": "shipping"}
        },
        {
            "id": "faq_3",
            "text": "We accept major credit cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, and Google Pay. We do not accept cash, personal checks, or cash on delivery (COD).",
            "metadata": {"source": "payment_methods.txt", "category": "payment"}
        },
        {
            "id": "faq_4",
            "text": "Our customer support team is available Monday through Friday from 9 AM to 6 PM EST. You can contact support via email at support@company.com, by calling 1-800-555-0199, or via our live chat on the website.",
            "metadata": {"source": "contact_info.txt", "category": "support"}
        },
        {
            "id": "faq_5",
            "text": "All our products come with a 1-year limited warranty covering manufacturing defects. The warranty does not cover accidental damage, wear and tear, or unauthorized modifications. To file a claim, please contact support with your receipt.",
            "metadata": {"source": "warranty.txt", "category": "warranty"}
        }
    ]
    # Simple embedding extraction (we will use a standard model later; Chroma will automatically use its default sentence-transformers model if none specified)
    default_collection.add(
        documents=[faq["text"] for faq in default_faqs],
        metadatas=[faq["metadata"] for faq in default_faqs],
        ids=[faq["id"] for faq in default_faqs]
    )
    logger.info("Default FAQ collection initialized.")


class RAGState(TypedDict):
    session_id: str
    question: str
    chat_history: List[Dict[str, Any]]
    raw_documents: List[Dict[str, Any]]
    reranked_documents: List[Dict[str, Any]]
    generation: str
    faithfulness_score: float
    faithfulness_reason: str
    attempts: int
    max_attempts: int
    confidence_threshold: float
    logs: List[str]


# ----------------- Nodes -----------------

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """
    Retrieve documents from ChromaDB and BM25.
    Combines dense similarity search with BM25 keyword search using RRF.
    """
    question = state["question"]
    session_id = state["session_id"]
    logs = state.get("logs", [])
    logs.append(f"Starting retrieval for query: '{question}' in session {session_id}")
    
    # 1. Retrieve from session-specific Chroma collection
    # If session doesn't exist, we fallback to default collection
    session_collection_name = f"session_{session_id}"
    try:
        collection = chroma_client.get_collection(
            name=session_collection_name,
            embedding_function=onnx_ef
        )
        logs.append(f"Using session-specific collection: {session_collection_name}")
    except Exception:
        collection = default_collection
        logs.append("No session-specific collection found. Using default FAQ collection.")

    total_count = collection.count()
    if total_count == 0:
        logs.append("No documents available in collection.")
        return {"raw_documents": [], "logs": logs}

    # Dense query (ChromaDB similarity search)
    # We query up to 8 documents
    dense_results = collection.query(
        query_texts=[question],
        n_results=min(8, total_count),
        include=["documents", "metadatas", "distances"]
    )
    
    dense_docs = []
    if dense_results and dense_results["documents"] and len(dense_results["documents"][0]) > 0:
        for idx in range(len(dense_results["documents"][0])):
            doc_text = dense_results["documents"][0][idx]
            metadata = dense_results["metadatas"][0][idx] or {}
            # distances are L2 or Cosine distance; let's convert to rank index
            dense_docs.append({
                "text": doc_text,
                "metadata": metadata,
                "dense_rank": idx
            })
    
    logs.append(f"Dense search retrieved {len(dense_docs)} documents.")

    # Sparse query (BM25)
    # To run BM25 dynamically, we fetch all documents in the current collection
    all_db_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_db_data["documents"]
    all_metadatas = all_db_data["metadatas"]
    
    sparse_docs = []
    if len(all_docs) > 0:
        try:
            from rank_bm25 import BM25Okapi
            # Tokenize documents simple whitespace/lower
            tokenized_corpus = [doc.lower().split() for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            
            tokenized_query = question.lower().split()
            # Get top N doc scores
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Sort documents by BM25 score
            scored_indices = sorted(enumerate(doc_scores), key=lambda x: x[1], reverse=True)
            # Take top 8
            top_indices = [idx for idx, score in scored_indices if score > 0][:8]
            
            for rank, idx in enumerate(top_indices):
                sparse_docs.append({
                    "text": all_docs[idx],
                    "metadata": all_metadatas[idx] or {},
                    "sparse_rank": rank
                })
            logs.append(f"BM25 search retrieved {len(sparse_docs)} documents with score > 0.")
        except Exception as e:
            logger.error(f"BM25 failed: {e}")
            logs.append(f"BM25 retrieval failed: {e}")
    
    # Combine results using Reciprocal Rank Fusion (RRF)
    # RRF Score = sum(1 / (rank + 60))
    rrf_scores = {}
    
    # Process dense
    for doc in dense_docs:
        doc_text = doc["text"]
        dense_rank = doc["dense_rank"]
        rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1.0 / (dense_rank + 60))
        
    # Process sparse
    for doc in sparse_docs:
        doc_text = doc["text"]
        sparse_rank = doc["sparse_rank"]
        rrf_scores[doc_text] = rrf_scores.get(doc_text, 0) + (1.0 / (sparse_rank + 60))
    
    # Combine metadata
    metadata_map = {}
    for doc in dense_docs + sparse_docs:
        metadata_map[doc["text"]] = doc["metadata"]
        
    # Sort by RRF score
    sorted_docs_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    raw_documents = []
    for doc_text, rrf_score in sorted_docs_by_rrf[:8]:
        raw_documents.append({
            "text": doc_text,
            "metadata": metadata_map[doc_text],
            "rrf_score": rrf_score
        })
        
    logs.append(f"Hybrid retrieval merged {len(raw_documents)} unique documents using RRF.")
    return {"raw_documents": raw_documents, "logs": logs}


def rerank_node(state: RAGState) -> Dict[str, Any]:
    """
    Reranks documents using a lightweight TF-IDF cosine similarity scorer.
    """
    question = state["question"]
    raw_docs = state["raw_documents"]
    logs = state.get("logs", [])
    
    if not raw_docs:
        logs.append("Reranking skipped (no documents retrieved).")
        return {"reranked_documents": [], "logs": logs}
        
    logs.append(f"Running lightweight TF-IDF reranking for {len(raw_docs)} documents...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        texts = [doc["text"] for doc in raw_docs]
        # Fit vectorizer on documents and the question
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts + [question])
        
        # The last row is the query vector
        query_vector = tfidf_matrix[-1]
        doc_vectors = tfidf_matrix[:-1]
        
        # Cosine similarity
        similarities = cosine_similarity(doc_vectors, query_vector).flatten()
        
        scored_docs = []
        for idx, doc in enumerate(raw_docs):
            scored_docs.append({
                **doc,
                "rerank_score": float(similarities[idx])
            })
            
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        reranked_docs = scored_docs[:4]
        
        # Log scores
        for idx, doc in enumerate(reranked_docs):
            logs.append(f"Rank {idx+1}: Score {doc['rerank_score']:.4f} | Source: {doc['metadata'].get('source', 'Unknown')} | Snippet: {doc['text'][:60]}...")
            
        return {"reranked_documents": reranked_docs, "logs": logs}
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        logs.append(f"Reranking failed: {e}. Falling back to default order.")
        return {"reranked_documents": raw_docs[:4], "logs": logs}


def generate_node(state: RAGState) -> Dict[str, Any]:
    """
    Generate response using Groq Mixtral LLM.
    """
    question = state["question"]
    docs = state["reranked_documents"]
    chat_history = state.get("chat_history", [])
    attempts = state.get("attempts", 0)
    logs = state.get("logs", [])
    
    logs.append(f"Generating answer. Attempt: {attempts + 1}")
    
    # Construct context string
    context = "\n\n".join([f"--- DOCUMENT {idx+1} (Source: {doc['metadata'].get('source', 'Unknown')}) ---\n{doc['text']}" for idx, doc in enumerate(docs)])
    
    # Construct memory string
    history_str = ""
    for msg in chat_history[-6:]:  # Last 6 messages
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    # Define generation prompt
    system_prompt = (
        "You are an expert customer support agent. Answer the user's question accurately and helpfully.\n"
        "You MUST rely ONLY on the provided documents context. Do NOT make up facts or extrapolate beyond what is documented.\n"
        "If you do not know the answer or the context does not contain the answer, say: "
        "'I am sorry, but I cannot confidently answer that question based on the available documentation.'\n\n"
        f"--- CONTEXT ---\n{context}\n\n"
        f"--- CHAT HISTORY ---\n{history_str}\n"
    )
    
    user_prompt = f"Question: {question}\nAnswer:"
    
    # Add correction warning if this is a regeneration attempt due to hallucination
    if attempts > 0:
        system_prompt += (
            "\n\n[WARNING] Your previous generation failed the hallucination guardrail evaluation. "
            "Your output contained statements NOT directly supported by the context documents. "
            "Please regenerate. Adhere strictly to the facts in the context and do not make any unsupported claims."
        )
        logs.append("Applying hallucination correction instructions to generation prompt.")

    try:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.invoke(messages)
        generation = response.content
        logs.append("LLM generation completed successfully.")
        
        return {
            "generation": generation,
            "attempts": attempts + 1,
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logs.append(f"LLM generation failed: {e}")
        return {
            "generation": "Error: Failed to generate response from LLM.",
            "attempts": attempts + 1,
            "logs": logs
        }


def guardrail_node(state: RAGState) -> Dict[str, Any]:
    """
    RAGAS-style hallucination guardrail.
    Evaluates whether the generated answer is faithful to the context.
    """
    docs = state["reranked_documents"]
    generation = state["generation"]
    logs = state.get("logs", [])
    
    if "Error:" in generation:
        logs.append("Skipping guardrail check due to generation error.")
        return {"faithfulness_score": 0.0, "faithfulness_reason": "Generation error", "logs": logs}
        
    logs.append("Running Hallucination Guardrail Grader...")
    
    context = "\n\n".join([doc['text'] for doc in docs])
    
    # Construct Grader prompt
    grader_system = (
        "You are an expert evaluator checking for hallucinations in RAG systems.\n"
        "Your task is to perform a faithfulness assessment matching the RAGAS metrics.\n"
        "Analyze the GENERATED ANSWER against the RETRIEVED CONTEXT.\n"
        "Follow these steps:\n"
        "1. Identify the individual facts/claims expressed in the GENERATED ANSWER.\n"
        "2. For each claim, check if it can be directly inferred from the RETRIEVED CONTEXT.\n"
        "3. Count the number of supported claims vs total claims.\n"
        "4. Calculate the faithfulness score = (supported claims) / (total claims). If no claims are made, score is 1.0.\n\n"
        "You MUST return your output in JSON format with exactly the following keys:\n"
        "{\n"
        "  \"claims\": [\n"
        "     { \"claim\": \"string\", \"supported\": true/false, \"explanation\": \"string\" }\n"
        "  ],\n"
        "  \"faithfulness_score\": float,\n"
        "  \"reason\": \"string\"\n"
        "}\n"
        "Do NOT return any markdown wrapping (no ```json) and no conversational text, just the raw JSON."
    )
    
    grader_user = (
        f"--- RETRIEVED CONTEXT ---\n{context}\n\n"
        f"--- GENERATED ANSWER ---\n{generation}\n"
    )
    
    try:
        # Use a fast model like llama-3-8b for quick JSON structured extraction
        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0)
        # Enable structured JSON output if supported or prompt strictly
        response = llm.invoke([
            {"role": "system", "content": grader_system},
            {"role": "user", "content": grader_user}
        ])
        
        raw_output = response.content.strip()
        # Clean markdown wraps if the model included them anyway
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```json")[-1].split("```")[0].strip()
            
        data = json.loads(raw_output)
        score = float(data.get("faithfulness_score", 1.0))
        reason = data.get("reason", "Evaluated successfully.")
        
        logs.append(f"Guardrail evaluation: Faithfulness Score = {score:.2f} | Reason: {reason}")
        
        # Log individual claims for trace
        for idx, c in enumerate(data.get("claims", [])):
            status = "SUPPORTED" if c.get("supported") else "HALLUCINATION"
            logs.append(f"  Claim {idx+1} [{status}]: '{c.get('claim')}' | Reason: {c.get('explanation')}")
            
        return {
            "faithfulness_score": score,
            "faithfulness_reason": reason,
            "logs": logs
        }
    except Exception as e:
        logger.error(f"Guardrail grading failed: {e}")
        logs.append(f"Guardrail evaluation failed: {e}. Defaulting to faithfulness score 1.0 to bypass loop.")
        return {
            "faithfulness_score": 1.0,
            "faithfulness_reason": f"Evaluation error: {e}",
            "logs": logs
        }


# ----------------- Conditional Edges -----------------

def decide_next_step(state: RAGState) -> Literal["generate", "__end__"]:
    """
    Decides whether to proceed or loop back to regenerate if hallucination detected.
    """
    score = state["faithfulness_score"]
    attempts = state["attempts"]
    max_attempts = state.get("max_attempts", 2)
    threshold = state.get("confidence_threshold", 0.4)
    logs = state.get("logs", [])
    
    if score >= threshold:
        logs.append(f"Faithfulness score {score:.2f} satisfies the threshold of {threshold}. Proceeding to end.")
        return END
        
    if attempts >= max_attempts:
        logs.append(f"Faithfulness score {score:.2f} is below threshold {threshold}, but max attempts ({max_attempts}) reached. Proceeding to end with fallback.")
        return END
        
    logs.append(f"Faithfulness score {score:.2f} is below threshold {threshold}. Loop back to regenerate.")
    return "generate"


# ----------------- Graph Construction -----------------

def build_rag_graph():
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("guardrail", guardrail_node)
    
    # Connect graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", "guardrail")
    
    # Conditional route from guardrail
    workflow.add_conditional_edges(
        "guardrail",
        decide_next_step,
        {
            "generate": "generate",
            END: END
        }
    )
    
    return workflow.compile()

# Compile the graph
rag_graph = build_rag_graph()
