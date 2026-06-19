import os
import shutil
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

# Import the RAG graph and chroma client
from rag_graph import rag_graph, chroma_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Conversational RAG Customer Support System API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to Netlify domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    question: str
    chat_history: List[Dict[str, Any]] = []
    confidence_threshold: float = 0.4

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Run the LangGraph Conversational RAG pipeline.
    Returns the complete structured JSON response containing the final generation,
    retrieved documents, reranked documents, and execution logs.
    """
    try:
        initial_state = {
            "session_id": request.session_id,
            "question": request.question,
            "chat_history": request.chat_history,
            "raw_documents": [],
            "reranked_documents": [],
            "generation": "",
            "faithfulness_score": 0.0,
            "faithfulness_reason": "",
            "attempts": 0,
            "max_attempts": 2,
            "confidence_threshold": request.confidence_threshold,
            "logs": []
        }
        
        # Run the graph synchronously/asynchronously
        result = rag_graph.invoke(initial_state)
        
        # Check for fallback case (e.g. if final answer was below threshold and max attempts reached)
        generation = result.get("generation", "")
        score = result.get("faithfulness_score", 0.0)
        threshold = request.confidence_threshold
        
        # If faithfulness is too low, overwrite with a safe agent message
        if score < threshold and "Error:" not in generation:
            generation = (
                "I am sorry, but I cannot confidently answer that question based on the available documentation. "
                "Would you like me to connect you to a live support agent?"
            )
            result["generation"] = generation
            result["logs"].append("Faithfulness score below threshold after final attempt. Replaced response with safe fallback message.")
            
        return {
            "generation": result.get("generation", ""),
            "raw_documents": result.get("raw_documents", []),
            "reranked_documents": result.get("reranked_documents", []),
            "faithfulness_score": score,
            "faithfulness_reason": result.get("faithfulness_reason", ""),
            "attempts": result.get("attempts", 0),
            "logs": result.get("logs", [])
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streams the LangGraph execution steps using Server-Sent Events (SSE).
    This allows the frontend to show logs and documents as they are retrieved,
    reranked, and checked by the guardrail.
    """
    async def event_generator():
        initial_state = {
            "session_id": request.session_id,
            "question": request.question,
            "chat_history": request.chat_history,
            "raw_documents": [],
            "reranked_documents": [],
            "generation": "",
            "faithfulness_score": 0.0,
            "faithfulness_reason": "",
            "attempts": 0,
            "max_attempts": 2,
            "confidence_threshold": request.confidence_threshold,
            "logs": []
        }
        
        try:
            # We iterate through state updates from the graph execution
            # use astream to stream updates node-by-node
            async for event in rag_graph.astream(initial_state, stream_mode="updates"):
                # event is a dictionary, e.g. {"retrieve": {...}} or {"generate": {...}}
                node_name = list(event.keys())[0]
                node_data = event[node_name]
                
                # Format payload
                payload = {
                    "node": node_name,
                    "logs": node_data.get("logs", []),
                    "raw_documents": node_data.get("raw_documents", []),
                    "reranked_documents": node_data.get("reranked_documents", []),
                    "generation": node_data.get("generation", ""),
                    "faithfulness_score": node_data.get("faithfulness_score", 0.0),
                    "faithfulness_reason": node_data.get("faithfulness_reason", "")
                }
                
                yield f"data: {json.dumps(payload)}\n\n"
                
            # Final Event with complete results & fallback adjustment
            # Let's run a final check to verify if we need to apply fallback
            # We fetch the final state from the last iteration
            final_res = rag_graph.invoke(initial_state)
            generation = final_res.get("generation", "")
            score = final_res.get("faithfulness_score", 0.0)
            
            if score < request.confidence_threshold and "Error:" not in generation:
                generation = (
                    "I am sorry, but I cannot confidently answer that question based on the available documentation. "
                    "Would you like me to connect you to a live support agent?"
                )
                final_res["generation"] = generation
                final_res["logs"].append("Final Answer failed guardrail. Safe fallback activated.")

            final_payload = {
                "node": "complete",
                "generation": generation,
                "faithfulness_score": score,
                "faithfulness_reason": final_res.get("faithfulness_reason", ""),
                "logs": final_res.get("logs", []),
                "raw_documents": final_res.get("raw_documents", []),
                "reranked_documents": final_res.get("reranked_documents", [])
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            error_payload = {"node": "error", "message": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a file, chunk it, embed it, and add to session-specific Chroma collection.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".txt", ".md", ".pdf"]:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf files are supported.")
        
    try:
        # Read file contents
        content = ""
        if file_ext in [".txt", ".md"]:
            content_bytes = await file.read()
            content = content_bytes.decode("utf-8", errors="ignore")
        elif file_ext == ".pdf":
            # Read PDF using pypdf if installed
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
                
            try:
                import pypdf
                reader = pypdf.PdfReader(tmp_path)
                text_list = []
                for page in reader.pages:
                    text_list.append(page.extract_text() or "")
                content = "\n".join(text_list)
            except ImportError:
                # If pypdf is not installed, we raise an error
                raise HTTPException(
                    status_code=500, 
                    detail="PDF parsing package (pypdf) is not installed on the server."
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if not content.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Simple text splitter: Chunk size 600, overlap 120
        chunk_size = 600
        overlap = 120
        chunks = []
        
        # Basic chunking logic
        words = content.split()
        current_chunk = []
        current_len = 0
        
        for word in words:
            current_chunk.append(word)
            current_len += len(word) + 1
            if current_len >= chunk_size:
                chunks.append(" ".join(current_chunk))
                # keep some overlap (approx last 15 words)
                current_chunk = current_chunk[-15:]
                current_len = sum(len(w) + 1 for w in current_chunk)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(f"Split document into {len(chunks)} chunks.")
        
        # Save to session collection in Chroma
        session_collection_name = f"session_{session_id}"
        collection = chroma_client.get_or_create_collection(name=session_collection_name)
        
        # Prepare IDs and Metadatas
        ids = [f"{session_id}_{file.filename}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [{"source": file.filename, "session_id": session_id} for _ in range(len(chunks))]
        
        # Add to ChromaDB
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks to collection {session_collection_name}")
        
        return {
            "filename": file.filename,
            "chunks_count": len(chunks),
            "collection": session_collection_name,
            "message": "File processed and indexed successfully."
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


@app.get("/api/sessions/{session_id}/documents")
def list_session_documents(session_id: str):
    """
    List all documents uploaded to a session.
    """
    session_collection_name = f"session_{session_id}"
    try:
        collection = chroma_client.get_collection(name=session_collection_name)
        data = collection.get(include=["metadatas"])
        
        # Get unique filenames
        filenames = set()
        if data and data["metadatas"]:
            for meta in data["metadatas"]:
                if meta and "source" in meta:
                    filenames.add(meta["source"])
                    
        return {"documents": list(filenames)}
    except Exception:
        # Collection does not exist
        return {"documents": []}


@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    """
    Delete session-specific vector storage.
    """
    session_collection_name = f"session_{session_id}"
    try:
        chroma_client.delete_collection(name=session_collection_name)
        return {"message": f"Session collection {session_collection_name} deleted successfully."}
    except Exception as e:
        return {"message": f"No collection found or error deleting collection: {str(e)}"}
