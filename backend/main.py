import os
import shutil
import asyncio
import json
import logging
import uuid
from typing import List, Dict, Any

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import the RAG graph and Chroma client
from rag_graph import (
    rag_graph,
    chroma_client,
    onnx_ef
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Conversational RAG Customer Support System API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    session_id: str

    question: str

    chat_history: List[Dict[str, Any]] = []

    confidence_threshold: float = 0.4


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy"
    }


# ============================================================
# CREATE INITIAL RAG STATE
# ============================================================

def create_initial_state(
    request: ChatRequest
) -> Dict[str, Any]:

    return {

        "session_id":
            request.session_id,

        "question":
            request.question,

        "chat_history":
            request.chat_history,

        "raw_documents":
            [],

        "reranked_documents":
            [],

        "generation":
            "",

        "faithfulness_score":
            0.0,

        "faithfulness_reason":
            "",

        "attempts":
            0,

        "max_attempts":
            2,

        "confidence_threshold":
            request.confidence_threshold,

        "logs":
            []
    }


# ============================================================
# NORMAL CHAT ENDPOINT
# ============================================================

@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest
):

    """
    Run the LangGraph Conversational RAG pipeline.

    Returns the complete structured JSON response.
    """

    logger.info(
        f"Chat request received: "
        f"'{request.question}' "
        f"| session={request.session_id}"
    )

    try:

        initial_state = create_initial_state(
            request
        )


        # ----------------------------------------------------
        # RUN GRAPH ONCE
        # ----------------------------------------------------

        result = rag_graph.invoke(
            initial_state
        )


        generation = result.get(
            "generation",
            ""
        )

        score = float(
            result.get(
                "faithfulness_score",
                0.0
            )
        )

        threshold = request.confidence_threshold


        # ----------------------------------------------------
        # SAFE FALLBACK
        # ----------------------------------------------------

        if (
            score < threshold
            and "Error:" not in generation
        ):

            generation = (
                "I am sorry, but I cannot confidently "
                "answer that question based on the "
                "available documentation. "
                "Would you like me to connect you "
                "to a live support agent?"
            )

            result["generation"] = generation

            result.setdefault(
                "logs",
                []
            ).append(
                "Faithfulness score below threshold "
                "after final attempt. "
                "Replaced response with safe fallback message."
            )


        return {

            "generation":
                result.get(
                    "generation",
                    ""
                ),

            "raw_documents":
                result.get(
                    "raw_documents",
                    []
                ),

            "reranked_documents":
                result.get(
                    "reranked_documents",
                    []
                ),

            "faithfulness_score":
                score,

            "faithfulness_reason":
                result.get(
                    "faithfulness_reason",
                    ""
                ),

            "attempts":
                result.get(
                    "attempts",
                    0
                ),

            "logs":
                result.get(
                    "logs",
                    []
                )
        }


    except Exception as e:

        logger.exception(
            "Error in chat endpoint"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# STREAMING CHAT ENDPOINT
# ============================================================

@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest
):
    """
    Run the synchronous LangGraph RAG pipeline in a worker thread
    and return the result through Server-Sent Events.

    The graph uses synchronous nodes, so we intentionally use
    rag_graph.invoke() inside asyncio.to_thread() instead of
    rag_graph.astream(). This avoids the async-executor
    CancelledError seen in the deployed Render service.
    """

    logger.info(
        f"Streaming request received: "
        f"'{request.question}' "
        f"| session={request.session_id}"
    )

    async def event_generator():

        initial_state = create_initial_state(request)

        try:
            # Send an immediate event so the browser/proxy knows
            # the stream has started.
            yield (
                "data: "
                + json.dumps({
                    "node": "start",
                    "generation": "",
                    "faithfulness_score": 0.0,
                    "faithfulness_reason": "",
                    "raw_documents": [],
                    "reranked_documents": [],
                    "logs": ["RAG pipeline started."]
                })
                + "\n\n"
            )

            # Run the complete synchronous graph ONCE in a
            # worker thread. Do not call rag_graph.astream().
            final_state = await asyncio.to_thread(
                rag_graph.invoke,
                initial_state
            )

            all_logs = final_state.get("logs", [])

            # Retrieval update
            retrieval_logs = [
                log for log in all_logs
                if (
                    "retriev" in log.lower()
                    or "collection" in log.lower()
                    or "dense" in log.lower()
                    or "bm25" in log.lower()
                    or "hybrid" in log.lower()
                )
            ]

            yield (
                "data: "
                + json.dumps({
                    "node": "retrieve",
                    "generation": "",
                    "faithfulness_score": 0.0,
                    "faithfulness_reason": "",
                    "raw_documents":
                        final_state.get("raw_documents", []),
                    "reranked_documents": [],
                    "logs": retrieval_logs
                })
                + "\n\n"
            )

            # Reranking update
            rerank_logs = [
                log for log in all_logs
                if (
                    "rerank" in log.lower()
                    or "rank " in log.lower()
                )
            ]

            yield (
                "data: "
                + json.dumps({
                    "node": "rerank",
                    "generation": "",
                    "faithfulness_score": 0.0,
                    "faithfulness_reason": "",
                    "raw_documents":
                        final_state.get("raw_documents", []),
                    "reranked_documents":
                        final_state.get("reranked_documents", []),
                    "logs": rerank_logs
                })
                + "\n\n"
            )

            # Generation update
            generation_logs = [
                log for log in all_logs
                if (
                    "generat" in log.lower()
                    or "llm" in log.lower()
                )
            ]

            yield (
                "data: "
                + json.dumps({
                    "node": "generate",
                    "generation":
                        final_state.get("generation", ""),
                    "faithfulness_score": 0.0,
                    "faithfulness_reason": "",
                    "raw_documents":
                        final_state.get("raw_documents", []),
                    "reranked_documents":
                        final_state.get("reranked_documents", []),
                    "logs": generation_logs
                })
                + "\n\n"
            )

            # Guardrail update
            score = float(
                final_state.get(
                    "faithfulness_score",
                    0.0
                )
            )

            guardrail_logs = [
                log for log in all_logs
                if (
                    "faith" in log.lower()
                    or "guardrail" in log.lower()
                    or "hallucination" in log.lower()
                )
            ]

            yield (
                "data: "
                + json.dumps({
                    "node": "guardrail",
                    "generation":
                        final_state.get("generation", ""),
                    "faithfulness_score": score,
                    "faithfulness_reason":
                        final_state.get(
                            "faithfulness_reason",
                            ""
                        ),
                    "raw_documents":
                        final_state.get("raw_documents", []),
                    "reranked_documents":
                        final_state.get("reranked_documents", []),
                    "logs": guardrail_logs
                })
                + "\n\n"
            )

            # Safe fallback if the final faithfulness score is
            # below the user's configured threshold.
            generation = final_state.get(
                "generation",
                ""
            )

            threshold = float(
                request.confidence_threshold
            )

            if (
                score < threshold
                and "Error:" not in generation
            ):
                generation = (
                    "I am sorry, but I cannot confidently "
                    "answer that question based on the "
                    "available documentation. "
                    "Would you like me to connect you "
                    "to a live support agent?"
                )

                final_state["generation"] = generation

                final_state.setdefault(
                    "logs",
                    []
                ).append(
                    "Final answer failed guardrail. "
                    "Safe fallback activated."
                )

            # Final event consumed by the React frontend.
            final_payload = {
                "node": "complete",
                "generation": generation,
                "faithfulness_score": score,
                "faithfulness_reason":
                    final_state.get(
                        "faithfulness_reason",
                        ""
                    ),
                "attempts":
                    final_state.get(
                        "attempts",
                        0
                    ),
                "raw_documents":
                    final_state.get(
                        "raw_documents",
                        []
                    ),
                "reranked_documents":
                    final_state.get(
                        "reranked_documents",
                        []
                    ),
                "logs":
                    final_state.get(
                        "logs",
                        []
                    )
            }

            yield (
                "data: "
                + json.dumps(final_payload)
                + "\n\n"
            )

            logger.info(
                "Streaming RAG request completed successfully."
            )

        except asyncio.CancelledError:
            logger.warning(
                "Chat stream request was cancelled by "
                "the client or upstream connection."
            )
            return

        except Exception as e:
            logger.exception(
                "Error in stream endpoint"
            )

            error_payload = {
                "node": "error",
                "message": str(e),
                "generation":
                    "Error: Failed to generate response from LLM.",
                "faithfulness_score": 0.0,
                "faithfulness_reason":
                    "Generation or pipeline error.",
                "raw_documents":
                    initial_state.get(
                        "raw_documents",
                        []
                    ),
                "reranked_documents":
                    initial_state.get(
                        "reranked_documents",
                        []
                    ),
                "logs":
                    initial_state.get(
                        "logs",
                        []
                    ) + [
                        f"Stream error: {str(e)}"
                    ]
            }

            yield (
                "data: "
                + json.dumps(error_payload)
                + "\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/api/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):

    """
    Upload a TXT, MD, or PDF document.

    The document is extracted, chunked, embedded,
    and stored in the session-specific ChromaDB collection.
    """

    if not file:

        raise HTTPException(
            status_code=400,
            detail="No file uploaded."
        )


    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename."
        )


    # ========================================================
    # FILE VALIDATION
    # ========================================================

    file_ext = os.path.splitext(
        file.filename
    )[1].lower()


    if file_ext not in [
        ".txt",
        ".md",
        ".pdf"
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only .txt, .md, and .pdf "
                "files are supported."
            )
        )


    logger.info(
        f"Uploading file '{file.filename}' "
        f"for session '{session_id}'"
    )


    try:

        # ====================================================
        # READ FILE
        # ====================================================

        content = ""


        # ----------------------------------------------------
        # TXT / MD
        # ----------------------------------------------------

        if file_ext in [
            ".txt",
            ".md"
        ]:

            content_bytes = await file.read()

            content = content_bytes.decode(
                "utf-8",
                errors="ignore"
            )


        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        elif file_ext == ".pdf":

            import tempfile


            tmp_path = None


            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp_path = tmp.name

                    shutil.copyfileobj(
                        file.file,
                        tmp
                    )


                try:

                    import pypdf

                except ImportError:

                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "PDF parsing package "
                            "(pypdf) is not installed "
                            "on the server."
                        )
                    )


                reader = pypdf.PdfReader(
                    tmp_path
                )


                text_list = []


                for page in reader.pages:

                    page_text = (
                        page.extract_text()
                        or ""
                    )

                    text_list.append(
                        page_text
                    )


                content = "\n".join(
                    text_list
                )


            finally:

                if (
                    tmp_path
                    and os.path.exists(tmp_path)
                ):

                    os.remove(
                        tmp_path
                    )


        # ====================================================
        # CONTENT VALIDATION
        # ====================================================

        if not content.strip():

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )


        # ====================================================
        # TEXT CHUNKING
        # ====================================================

        chunk_size = 600

        overlap = 120

        chunks = []


        words = content.split()

        current_chunk = []

        current_len = 0


        for word in words:

            current_chunk.append(
                word
            )

            current_len += (
                len(word) + 1
            )


            if current_len >= chunk_size:

                chunks.append(
                    " ".join(
                        current_chunk
                    )
                )


                # Approximate overlap
                # using the last 15 words

                current_chunk = (
                    current_chunk[-15:]
                )


                current_len = sum(
                    len(w) + 1
                    for w in current_chunk
                )


        # Add remaining text

        if current_chunk:

            chunks.append(
                " ".join(
                    current_chunk
                )
            )


        if not chunks:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not create document chunks."
                )
            )


        logger.info(
            f"Split document into "
            f"{len(chunks)} chunks."
        )


        # ====================================================
        # SESSION COLLECTION
        # ====================================================

        session_collection_name = (
            f"session_{session_id}"
        )


        collection = (
            chroma_client.get_or_create_collection(
                name=session_collection_name,
                embedding_function=onnx_ef
            )
        )


        # ====================================================
        # UNIQUE DOCUMENT ID
        # ====================================================

        upload_id = uuid.uuid4().hex


        ids = [

            (
                f"{session_id}_"
                f"{upload_id}_"
                f"chunk_{idx}"
            )

            for idx in range(
                len(chunks)
            )
        ]


        metadatas = [

            {
                "source":
                    file.filename,

                "session_id":
                    session_id
            }

            for _ in chunks
        ]


        # ====================================================
        # ADD TO CHROMADB
        # ====================================================

        collection.add(

            documents=chunks,

            metadatas=metadatas,

            ids=ids
        )


        logger.info(
            f"Successfully added "
            f"{len(chunks)} chunks to "
            f"collection "
            f"{session_collection_name}"
        )


        # ====================================================
        # SUCCESS RESPONSE
        # ====================================================

        return {

            "filename":
                file.filename,

            "chunks_count":
                len(chunks),

            "collection":
                session_collection_name,

            "message":
                "File processed and indexed successfully."
        }


    except HTTPException:

        raise


    except Exception as e:

        logger.exception(
            "Error uploading file"
        )


        raise HTTPException(
            status_code=500,
            detail=(
                f"File processing failed: {str(e)}"
            )
        )


# ============================================================
# LIST SESSION DOCUMENTS
# ============================================================

@app.get(
    "/api/sessions/{session_id}/documents"
)
def list_session_documents(
    session_id: str
):

    """
    List all documents uploaded to a session.
    """

    session_collection_name = (
        f"session_{session_id}"
    )


    try:

        collection = (
            chroma_client.get_collection(
                name=session_collection_name
            )
        )


        data = collection.get(
            include=[
                "metadatas"
            ]
        )


        filenames = set()


        if (
            data
            and data.get("metadatas")
        ):

            for meta in data["metadatas"]:

                if (
                    meta
                    and "source" in meta
                ):

                    filenames.add(
                        meta["source"]
                    )


        return {
            "documents":
                list(filenames)
        }


    except Exception:

        return {
            "documents":
                []
        }


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete(
    "/api/sessions/{session_id}"
)
def clear_session(
    session_id: str
):

    """
    Delete session-specific vector storage.
    """

    session_collection_name = (
        f"session_{session_id}"
    )


    try:

        chroma_client.delete_collection(
            name=session_collection_name
        )


        return {

            "message":
                (
                    f"Session collection "
                    f"{session_collection_name} "
                    f"deleted successfully."
                )
        }


    except Exception as e:

        return {

            "message":
                (
                    f"No collection found or "
                    f"error deleting collection: "
                    f"{str(e)}"
                )
        }
