import os
import json
import logging
from typing import List, Dict, Any, TypedDict, Literal

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Import Chroma and embeddings
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

# Load .env file from the backend directory specifically
backend_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(backend_dir, ".env")

load_dotenv(dotenv_path=dotenv_path)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# ONNX EMBEDDING MODEL
# ============================================================

try:
    logger.info("Initializing ONNX embedding model...")

    onnx_ef = ONNXMiniLM_L6_V2()

    logger.info("ONNX embedding model initialized.")

except Exception as e:
    logger.exception("Failed to initialize ONNX embedding model")
    onnx_ef = None


# ============================================================
# GROQ API KEY
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    logger.warning(
        "GROQ_API_KEY not found in environment variables. "
        "Make sure to set it."
    )


# ============================================================
# CHROMADB CONFIGURATION
# ============================================================

CHROMA_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "chroma_db"
)

logger.info(
    f"Initializing ChromaDB PersistentClient at: {CHROMA_DATA_PATH}"
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DATA_PATH
)


# ============================================================
# DEFAULT FAQ COLLECTION
# ============================================================

default_collection = chroma_client.get_or_create_collection(
    name="default_faq",
    embedding_function=onnx_ef
)


# ============================================================
# DEFAULT FAQ DATA
# ============================================================

if default_collection.count() == 0:

    logger.info("Initializing default FAQ collection...")

    default_faqs = [

        {
            "id": "faq_1",
            "text": (
                "The returns policy allows customers to return any "
                "unused, unopened products within 30 days of purchase "
                "for a full refund. Shipping costs for returns are the "
                "responsibility of the customer unless the item was "
                "damaged or defective upon arrival."
            ),
            "metadata": {
                "source": "returns_policy.txt",
                "category": "returns"
            }
        },

        {
            "id": "faq_2",
            "text": (
                "Standard shipping takes 3-5 business days. Express "
                "shipping takes 1-2 business days. Orders over $50 "
                "qualify for free standard shipping. Orders are "
                "processed within 24 hours on weekdays."
            ),
            "metadata": {
                "source": "shipping_policy.txt",
                "category": "shipping"
            }
        },

        {
            "id": "faq_3",
            "text": (
                "We accept major credit cards (Visa, MasterCard, "
                "American Express, Discover), PayPal, Apple Pay, "
                "and Google Pay. We do not accept cash, personal "
                "checks, or cash on delivery (COD)."
            ),
            "metadata": {
                "source": "payment_methods.txt",
                "category": "payment"
            }
        },

        {
            "id": "faq_4",
            "text": (
                "Our customer support team is available Monday "
                "through Friday from 9 AM to 6 PM EST. You can "
                "contact support via email at support@company.com, "
                "by calling 1-800-555-0199, or via our live chat "
                "on the website."
            ),
            "metadata": {
                "source": "contact_info.txt",
                "category": "support"
            }
        },

        {
            "id": "faq_5",
            "text": (
                "All our products come with a 1-year limited warranty "
                "covering manufacturing defects. The warranty does "
                "not cover accidental damage, wear and tear, or "
                "unauthorized modifications. To file a claim, please "
                "contact support with your receipt."
            ),
            "metadata": {
                "source": "warranty.txt",
                "category": "warranty"
            }
        }
    ]

    default_collection.add(
        documents=[
            faq["text"]
            for faq in default_faqs
        ],
        metadatas=[
            faq["metadata"]
            for faq in default_faqs
        ],
        ids=[
            faq["id"]
            for faq in default_faqs
        ]
    )

    logger.info("Default FAQ collection initialized.")


# ============================================================
# RAG STATE
# ============================================================

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


# ============================================================
# RETRIEVAL NODE
# ============================================================

def retrieve_node(state: RAGState) -> Dict[str, Any]:

    """
    Retrieve documents from ChromaDB and BM25.

    Combines dense similarity search with BM25 keyword search
    using Reciprocal Rank Fusion (RRF).
    """

    question = state["question"]

    session_id = state["session_id"]

    logs = state.get("logs", [])

    logs.append(
        f"Starting retrieval for query: '{question}' "
        f"in session {session_id}"
    )


    # --------------------------------------------------------
    # SESSION COLLECTION
    # --------------------------------------------------------

    session_collection_name = f"session_{session_id}"

    try:

        collection = chroma_client.get_collection(
            name=session_collection_name,
            embedding_function=onnx_ef
        )

        logs.append(
            f"Using session-specific collection: "
            f"{session_collection_name}"
        )

    except Exception:

        collection = default_collection

        logs.append(
            "No session-specific collection found. "
            "Using default FAQ collection."
        )


    # --------------------------------------------------------
    # CHECK COLLECTION
    # --------------------------------------------------------

    total_count = collection.count()

    if total_count == 0:

        logs.append(
            "No documents available in collection."
        )

        return {
            "raw_documents": [],
            "logs": logs
        }


    # --------------------------------------------------------
    # DENSE RETRIEVAL
    # --------------------------------------------------------

    dense_results = collection.query(
        query_texts=[question],
        n_results=min(8, total_count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    dense_docs = []

    if (
        dense_results
        and dense_results["documents"]
        and len(dense_results["documents"][0]) > 0
    ):

        for idx in range(
            len(dense_results["documents"][0])
        ):

            doc_text = dense_results["documents"][0][idx]

            metadata = (
                dense_results["metadatas"][0][idx]
                or {}
            )

            dense_docs.append(
                {
                    "text": doc_text,
                    "metadata": metadata,
                    "dense_rank": idx
                }
            )


    logs.append(
        f"Dense search retrieved "
        f"{len(dense_docs)} documents."
    )


    # --------------------------------------------------------
    # BM25 SPARSE RETRIEVAL
    # --------------------------------------------------------

    all_db_data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    all_docs = all_db_data["documents"]

    all_metadatas = all_db_data["metadatas"]


    sparse_docs = []


    if len(all_docs) > 0:

        try:

            from rank_bm25 import BM25Okapi


            # Tokenize documents
            tokenized_corpus = [
                doc.lower().split()
                for doc in all_docs
            ]


            bm25 = BM25Okapi(
                tokenized_corpus
            )


            # Tokenize query
            tokenized_query = (
                question.lower().split()
            )


            # Calculate BM25 scores
            doc_scores = bm25.get_scores(
                tokenized_query
            )


            # Sort documents by BM25 score
            scored_indices = sorted(
                enumerate(doc_scores),
                key=lambda x: x[1],
                reverse=True
            )


            # Take top 8 documents with score > 0
            top_indices = [
                idx
                for idx, score in scored_indices
                if score > 0
            ][:8]


            for rank, idx in enumerate(
                top_indices
            ):

                sparse_docs.append(
                    {
                        "text": all_docs[idx],
                        "metadata": (
                            all_metadatas[idx]
                            or {}
                        ),
                        "sparse_rank": rank
                    }
                )


            logs.append(
                f"BM25 search retrieved "
                f"{len(sparse_docs)} documents "
                f"with score > 0."
            )


        except Exception as e:

            logger.exception(
                "BM25 retrieval failed"
            )

            logs.append(
                f"BM25 retrieval failed: {e}"
            )


    # ========================================================
    # RECIPROCAL RANK FUSION
    # ========================================================

    rrf_scores = {}


    # --------------------------------------------------------
    # Process dense results
    # --------------------------------------------------------

    for doc in dense_docs:

        doc_text = doc["text"]

        dense_rank = doc["dense_rank"]

        rrf_scores[doc_text] = (
            rrf_scores.get(doc_text, 0)
            + (
                1.0 /
                (dense_rank + 60)
            )
        )


    # --------------------------------------------------------
    # Process sparse results
    # --------------------------------------------------------

    for doc in sparse_docs:

        doc_text = doc["text"]

        sparse_rank = doc["sparse_rank"]

        rrf_scores[doc_text] = (
            rrf_scores.get(doc_text, 0)
            + (
                1.0 /
                (sparse_rank + 60)
            )
        )


    # --------------------------------------------------------
    # Combine metadata
    # --------------------------------------------------------

    metadata_map = {}


    for doc in dense_docs + sparse_docs:

        metadata_map[
            doc["text"]
        ] = doc["metadata"]


    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    sorted_docs_by_rrf = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # Create raw documents
    # --------------------------------------------------------

    raw_documents = []


    for doc_text, rrf_score in (
        sorted_docs_by_rrf[:8]
    ):

        raw_documents.append(
            {
                "text": doc_text,

                "metadata": metadata_map[
                    doc_text
                ],

                "rrf_score": rrf_score
            }
        )


    logs.append(
        f"Hybrid retrieval merged "
        f"{len(raw_documents)} unique documents "
        f"using RRF."
    )


    return {
        "raw_documents": raw_documents,
        "logs": logs
    }


# ============================================================
# TF-IDF RERANKING NODE
# ============================================================

def rerank_node(
    state: RAGState
) -> Dict[str, Any]:

    """
    Reranks documents using lightweight
    TF-IDF cosine similarity.
    """

    question = state["question"]

    raw_docs = state["raw_documents"]

    logs = state.get("logs", [])


    if not raw_docs:

        logs.append(
            "Reranking skipped "
            "(no documents retrieved)."
        )

        return {
            "reranked_documents": [],
            "logs": logs
        }


    logs.append(
        f"Running lightweight TF-IDF reranking "
        f"for {len(raw_docs)} documents..."
    )


    try:

        from sklearn.feature_extraction.text import (
            TfidfVectorizer
        )

        from sklearn.metrics.pairwise import (
            cosine_similarity
        )


        texts = [
            doc["text"]
            for doc in raw_docs
        ]


        # ----------------------------------------------------
        # Fit TF-IDF
        # ----------------------------------------------------

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )


        tfidf_matrix = vectorizer.fit_transform(
            texts + [question]
        )


        # Last row is query
        query_vector = tfidf_matrix[-1]


        # Remaining rows are documents
        doc_vectors = tfidf_matrix[:-1]


        # ----------------------------------------------------
        # Cosine similarity
        # ----------------------------------------------------

        similarities = cosine_similarity(
            doc_vectors,
            query_vector
        ).flatten()


        # ----------------------------------------------------
        # Attach scores
        # ----------------------------------------------------

        scored_docs = []


        for idx, doc in enumerate(
            raw_docs
        ):

            scored_docs.append(
                {
                    **doc,
                    "rerank_score": float(
                        similarities[idx]
                    )
                }
            )


        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        scored_docs.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )


        # Top 4 documents
        reranked_docs = scored_docs[:4]


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        for idx, doc in enumerate(
            reranked_docs
        ):

            logs.append(
                f"Rank {idx + 1}: "
                f"Score {doc['rerank_score']:.4f} | "
                f"Source: "
                f"{doc['metadata'].get('source', 'Unknown')} | "
                f"Snippet: "
                f"{doc['text'][:60]}..."
            )


        return {
            "reranked_documents": reranked_docs,
            "logs": logs
        }


    except Exception as e:

        logger.exception(
            "Reranking failed"
        )

        logs.append(
            f"Reranking failed: {e}. "
            f"Falling back to default order."
        )


        return {
            "reranked_documents": raw_docs[:4],
            "logs": logs
        }


# ============================================================
# GENERATION NODE
# ============================================================

def generate_node(
    state: RAGState
) -> Dict[str, Any]:

    """
    Generate response using Groq LLM.

    Uses OpenAI GPT-OSS 120B through Groq.
    """

    question = state["question"]

    docs = state["reranked_documents"]

    chat_history = state.get(
        "chat_history",
        []
    )

    attempts = state.get(
        "attempts",
        0
    )

    logs = state.get(
        "logs",
        [])


    logs.append(
        f"Generating answer. "
        f"Attempt: {attempts + 1}"
    )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = "\n\n".join(
        [
            (
                f"--- DOCUMENT {idx + 1} "
                f"(Source: "
                f"{doc['metadata'].get('source', 'Unknown')}) ---\n"
                f"{doc['text']}"
            )

            for idx, doc in enumerate(docs)
        ]
    )


    # ========================================================
    # BUILD CHAT HISTORY
    # ========================================================

    history_str = ""


    for msg in chat_history[-6:]:

        role = (
            "User"
            if msg["role"] == "user"
            else "Assistant"
        )

        history_str += (
            f"{role}: "
            f"{msg['content']}\n"
        )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = (

        "You are an expert customer support agent. "
        "Answer the user's question accurately and helpfully.\n"

        "You MUST rely ONLY on the provided documents context. "
        "Do NOT make up facts or extrapolate beyond what is documented.\n"

        "If you do not know the answer or the context does not "
        "contain the answer, say: "
        "'I am sorry, but I cannot confidently answer that question "
        "based on the available documentation.'\n\n"

        f"--- CONTEXT ---\n"
        f"{context}\n\n"

        f"--- CHAT HISTORY ---\n"
        f"{history_str}\n"
    )


    user_prompt = (
        f"Question: {question}\n"
        f"Answer:"
    )


    # ========================================================
    # REGENERATION WARNING
    # ========================================================

    if attempts > 0:

        system_prompt += (

            "\n\n[WARNING] Your previous generation failed "
            "the hallucination guardrail evaluation. "

            "Your output contained statements NOT directly "
            "supported by the context documents. "

            "Please regenerate. Adhere strictly to the facts "
            "in the context and do not make any unsupported claims."
        )

        logs.append(
            "Applying hallucination correction instructions "
            "to generation prompt."
        )


    # ========================================================
    # GROQ GENERATION
    # ========================================================

    try:

        # ----------------------------------------------------
        # UPDATED GROQ MODEL
        # ----------------------------------------------------
        llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            temperature=0.1
        )


        messages = [

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ]


        response = llm.invoke(
            messages
        )


        generation = response.content


        logs.append(
            "LLM generation completed successfully."
        )


        return {

            "generation": generation,

            "attempts": attempts + 1,

            "logs": logs
        }


    except Exception as e:

        # IMPORTANT:
        # logger.exception prints the complete traceback
        # in Render logs.

        logger.exception(
            "Generation failed"
        )


        logs.append(
            f"LLM generation failed: {e}"
        )


        return {

            "generation":
                "Error: Failed to generate response from LLM.",

            "attempts":
                attempts + 1,

            "logs":
                logs
        }


# ============================================================
# HALLUCINATION GUARDRAIL NODE
# ============================================================

def guardrail_node(
    state: RAGState
) -> Dict[str, Any]:

    """
    RAGAS-style hallucination guardrail.

    Evaluates whether the generated answer
    is faithful to the retrieved context.
    """

    docs = state["reranked_documents"]

    generation = state["generation"]

    logs = state.get(
        "logs",
        []
    )


    # ========================================================
    # GENERATION ERROR CHECK
    # ========================================================

    if "Error:" in generation:

        logs.append(
            "Skipping guardrail check "
            "due to generation error."
        )

        return {

            "faithfulness_score": 0.0,

            "faithfulness_reason":
                "Generation error",

            "logs":
                logs
        }


    logs.append(
        "Running Hallucination Guardrail Grader..."
    )


    # ========================================================
    # CONTEXT
    # ========================================================

    context = "\n\n".join(
        [
            doc["text"]
            for doc in docs
        ]
    )


    # ========================================================
    # GRADER SYSTEM PROMPT
    # ========================================================

    grader_system = (

        "You are an expert evaluator checking for hallucinations "
        "in RAG systems.\n"

        "Your task is to perform a faithfulness assessment "
        "matching the RAGAS metrics.\n"

        "Analyze the GENERATED ANSWER against the RETRIEVED CONTEXT.\n"

        "Follow these steps:\n"

        "1. Identify the individual facts/claims expressed "
        "in the GENERATED ANSWER.\n"

        "2. For each claim, check if it can be directly inferred "
        "from the RETRIEVED CONTEXT.\n"

        "3. Count the number of supported claims vs total claims.\n"

        "4. Calculate the faithfulness score = "
        "(supported claims) / (total claims). "
        "If no claims are made, score is 1.0.\n\n"

        "You MUST return your output in JSON format with exactly "
        "the following keys:\n"

        "{\n"

        '  "claims": [\n'

        '     { "claim": "string", '
        '"supported": true/false, '
        '"explanation": "string" }\n'

        "  ],\n"

        '  "faithfulness_score": float,\n'

        '  "reason": "string"\n'

        "}\n"

        "Do NOT return any markdown wrapping "
        "(no ```json) and no conversational text, "
        "just the raw JSON."
    )


    # ========================================================
    # GRADER USER PROMPT
    # ========================================================

    grader_user = (

        f"--- RETRIEVED CONTEXT ---\n"
        f"{context}\n\n"

        f"--- GENERATED ANSWER ---\n"
        f"{generation}\n"
    )


    # ========================================================
    # GUARDRAIL LLM
    # ========================================================

    try:

        # ----------------------------------------------------
        # UPDATED GROQ MODEL
        # ----------------------------------------------------
        llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0.0
        )


        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": grader_system
                },

                {
                    "role": "user",
                    "content": grader_user
                }
            ]
        )


        raw_output = response.content.strip()


        # ====================================================
        # CLEAN MARKDOWN JSON
        # ====================================================

        if raw_output.startswith("```"):

            raw_output = (
                raw_output
                .split("```json")[-1]
                .split("```")[0]
                .strip()
            )


        # ====================================================
        # PARSE JSON
        # ====================================================

        data = json.loads(
            raw_output
        )


        score = float(
            data.get(
                "faithfulness_score",
                1.0
            )
        )


        reason = data.get(
            "reason",
            "Evaluated successfully."
        )


        logs.append(
            f"Guardrail evaluation: "
            f"Faithfulness Score = {score:.2f} | "
            f"Reason: {reason}"
        )


        # ====================================================
        # LOG INDIVIDUAL CLAIMS
        # ====================================================

        for idx, c in enumerate(
            data.get("claims", [])
        ):

            status = (
                "SUPPORTED"
                if c.get("supported")
                else "HALLUCINATION"
            )


            logs.append(
                f"  Claim {idx + 1} "
                f"[{status}]: "
                f"'{c.get('claim')}' | "
                f"Reason: "
                f"{c.get('explanation')}"
            )


        return {

            "faithfulness_score":
                score,

            "faithfulness_reason":
                reason,

            "logs":
                logs
        }


    except Exception as e:

        # IMPORTANT:
        # logger.exception prints full traceback
        # in Render.

        logger.exception(
            "Guardrail grading failed"
        )


        logs.append(
            f"Guardrail evaluation failed: {e}. "
            f"Defaulting to faithfulness score "
            f"1.0 to bypass loop."
        )


        return {

            "faithfulness_score":
                1.0,

            "faithfulness_reason":
                f"Evaluation error: {e}",

            "logs":
                logs
        }


# ============================================================
# CONDITIONAL EDGE
# ============================================================

def decide_next_step(
    state: RAGState
) -> Literal["generate", "__end__"]:

    """
    Decide whether to proceed or regenerate
    if hallucination is detected.
    """

    score = state[
        "faithfulness_score"
    ]

    attempts = state[
        "attempts"
    ]

    max_attempts = state.get(
        "max_attempts",
        2
    )

    threshold = state.get(
        "confidence_threshold",
        0.4
    )

    logs = state.get(
        "logs",
        []
    )


    # ========================================================
    # PASS
    # ========================================================

    if score >= threshold:

        logs.append(
            f"Faithfulness score "
            f"{score:.2f} satisfies the threshold "
            f"of {threshold}. "
            f"Proceeding to end."
        )

        return END


    # ========================================================
    # MAX ATTEMPTS
    # ========================================================

    if attempts >= max_attempts:

        logs.append(
            f"Faithfulness score "
            f"{score:.2f} is below threshold "
            f"{threshold}, but max attempts "
            f"({max_attempts}) reached. "
            f"Proceeding to end with fallback."
        )

        return END


    # ========================================================
    # REGENERATE
    # ========================================================

    logs.append(
        f"Faithfulness score "
        f"{score:.2f} is below threshold "
        f"{threshold}. "
        f"Loop back to regenerate."
    )


    return "generate"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_rag_graph():

    workflow = StateGraph(
        RAGState
    )


    # ========================================================
    # ADD NODES
    # ========================================================

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "rerank",
        rerank_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_node(
        "guardrail",
        guardrail_node
    )


    # ========================================================
    # CONNECT GRAPH
    # ========================================================

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "rerank"
    )

    workflow.add_edge(
        "rerank",
        "generate"
    )

    workflow.add_edge(
        "generate",
        "guardrail"
    )


    # ========================================================
    # CONDITIONAL ROUTING
    # ========================================================

    workflow.add_conditional_edges(

        "guardrail",

        decide_next_step,

        {
            "generate": "generate",
            END: END
        }
    )


    return workflow.compile()


# ============================================================
# COMPILE RAG GRAPH
# ============================================================

rag_graph = build_rag_graph()
