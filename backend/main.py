import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
from typing import List
import asyncio
import threading
import time
import logging
import json

from backend.config import CORS_ORIGINS, API_HOST, API_PORT, DATABASE_TYPE, DEBUG
from backend.schemas import (
    ChatCreate, ChatRename, ChatResponse, ChatDetailResponse,
    MessageCreate, MessageResponse, StreamMessage, HealthResponse
)
from src.chat_db import (
    init_db, add_chat, get_chats, get_messages, 
    add_message, rename_chat as db_rename_chat, 
    delete_chat as db_delete_chat, get_chat_by_id
)
from src.generation import create_conversational_chain
from src.retrieval import load_retriever_from_disk
from src.logging_utils import setup_logger, log_timing, log_query_start, log_query_complete

# Image analysis (CLIP-based, runs locally — zero API cost)
try:
    from src.image_analyzer import CLIPDiseaseAnalyzer
    IMAGE_ANALYSIS_AVAILABLE = True
except ImportError:
    print("CLIP image analysis module not available. Install: pip install transformers torch")
    IMAGE_ANALYSIS_AVAILABLE = False

# Initialize logger
api_logger = setup_logger('api')

# Initialize FastAPI app
app = FastAPI(
    title="Aloo Sahayak API",
    description="Backend API for Potato Disease Assistant with RAG",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if isinstance(CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from src.image_analyzer import REFERENCE_IMAGES_DIR
import os
os.makedirs(REFERENCE_IMAGES_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=REFERENCE_IMAGES_DIR), name="images")

# Global instances — loaded lazily on first request to reduce startup RAM
qa_chain = None
retriever = None
image_analyzer = None
_chain_lock = threading.Lock()
_image_lock = threading.Lock()

def get_qa_chain():
    """Lazy-load RAG chain on first request. Thread-safe."""
    global qa_chain, retriever
    if qa_chain is None:
        with _chain_lock:
            if qa_chain is None:
                retriever = load_retriever_from_disk()
                qa_chain = create_conversational_chain(retriever)
                print(" RAG chain loaded (lazy)")
    return qa_chain

def get_image_analyzer():
    """Lazy-load CLIP analyzer on first image request. Thread-safe."""
    global image_analyzer
    if image_analyzer is None:
        with _image_lock:
            if image_analyzer is None:
                if not IMAGE_ANALYSIS_AVAILABLE:
                    return None
                try:
                    image_analyzer = CLIPDiseaseAnalyzer(load_faiss_index=True)
                    print(" CLIP image analyzer loaded (lazy)")
                except Exception as e:
                    print(f" CLIP image analyzer failed: {e}")
                    return None
    return image_analyzer

# Startup and Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup. RAG chain and CLIP load lazily on first request."""
    print(" Starting Aloo Sahayak API...")

    # Initialize database
    try:
        init_db()
        print(" Database initialized successfully")
    except Exception as e:
        print(f" Database initialization failed: {e}")
        raise

    print(" API startup complete! (RAG chain and CLIP will load on first request)")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print(" Shutting down Aloo Sahayak API...")

# ==================== Health Check ====================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        database=DATABASE_TYPE
    )

# ==================== Chat Endpoints ====================

@app.get("/api/chats", response_model=List[ChatResponse])
async def list_chats(request: Request):
    """Get all chats for the authenticated user"""
    try:
        user_id = request.headers.get("x-user-id")
        chats_data = get_chats(user_id=user_id)
        chats = []
        
        for chat_id, name, created_at in chats_data:
            messages = get_messages(chat_id)
            chats.append(ChatResponse(
                id=chat_id,
                name=name,
                created_at=created_at,
                message_count=len(messages) // 2  # Divide by 2 (user + assistant)
            ))
        
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats", response_model=ChatResponse)
async def create_chat(chat: ChatCreate, request: Request):
    """Create a new chat for the authenticated user"""
    try:
        chat_id = str(uuid.uuid4())
        # Use user_id from header, fallback to body field
        user_id = request.headers.get("x-user-id") or chat.user_id
        add_chat(chat_id, chat.name, user_id=user_id)
        chat_row = get_chat_by_id(chat_id)
        
        return ChatResponse(
            id=chat_id,
            name=chat.name,
            created_at=chat_row[2],
            message_count=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str):
    """Get chat details with messages"""
    try:
        chat_row = get_chat_by_id(chat_id)
        if not chat_row:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat_name = chat_row[1]
        created_at = chat_row[2]

        messages = get_messages(chat_id)
        message_list = []
        for i, msg in enumerate(messages):
            sender = msg[0]
            content = msg[1]
            timestamp = msg[2]
            metadata = msg[3] if len(msg) > 3 else None
            message_list.append(
                MessageResponse(
                    id=i,
                    sender=sender,
                    content=content,
                    timestamp=timestamp,
                    metadata=metadata
                )
            )
        
        return ChatDetailResponse(
            id=chat_id,
            name=chat_name,
            created_at=created_at,
            messages=message_list
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/chats/{chat_id}", response_model=ChatResponse)
async def update_chat(chat_id: str, chat: ChatRename):
    """Rename a chat"""
    try:
        chat_row = get_chat_by_id(chat_id)
        if not chat_row:
            raise HTTPException(status_code=404, detail="Chat not found")

        created_at = chat_row[2]
        db_rename_chat(chat_id, chat.new_name)
        
        return ChatResponse(
            id=chat_id,
            name=chat.new_name,
            created_at=created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    try:
        db_delete_chat(chat_id)
        return {"status": "success", "message": "Chat deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Message Endpoints ====================

@app.get("/api/chats/{chat_id}/messages", response_model=List[MessageResponse])
async def get_chat_messages(chat_id: str):
    """Get all messages for a chat"""
    try:
        messages = get_messages(chat_id)
        
        result = []
        for i, msg in enumerate(messages):
            sender = msg[0]
            content = msg[1]
            timestamp = msg[2]
            metadata = msg[3] if len(msg) > 3 else None
            result.append(MessageResponse(id=i, sender=sender, content=content, timestamp=timestamp, metadata=metadata))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, message: MessageCreate):
    """Send message and get AI response (non-streaming)"""
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    
    try:
        log_query_start(api_logger, request_id, message.content[:100])
        
        if get_qa_chain() is None:
            raise HTTPException(status_code=500, detail="RAG chain not loaded")
        
        user_query = message.content
        
        # Add user message to database
        db_start = time.perf_counter()
        add_message(chat_id, "user", user_query)
        db_elapsed = time.perf_counter() - db_start
        
        log_timing(api_logger, "DB_ADD_USER_MESSAGE", {
            'duration_ms': round(db_elapsed * 1000, 2)
        })
        
        # Get chat history for context
        hist_start = time.perf_counter()
        messages = get_messages(chat_id)
        chat_history = [
            (msg[0], msg[1]) 
            for msg in messages[:-1]  # Exclude latest user message
            if msg[0] in ["user", "assistant"]
        ]
        
        # Reconstruct alternating pairs for context
        reconstructed_history = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                reconstructed_history.append((chat_history[i][1], chat_history[i+1][1]))
        
        hist_elapsed = time.perf_counter() - hist_start
        log_timing(api_logger, "HISTORY_RETRIEVAL", {
            'duration_ms': round(hist_elapsed * 1000, 2),
            'history_items': len(reconstructed_history)
        })
        
        # Add language instruction
        if message.language == "Hindi":
            query = user_query + "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
        else:
            query = user_query
        
        # Get AI response (measure time)
        chain_start = time.perf_counter()
        result = get_qa_chain().invoke({
            "question": query,
            "chat_history": reconstructed_history
        })
        chain_elapsed = time.perf_counter() - chain_start
        
        log_timing(api_logger, "AI_CHAIN_INVOCATION", {
            'duration_ms': round(chain_elapsed * 1000, 2)
        })
        
        ai_response = result.get("answer", "")
        source_docs = result.get("source_documents", [])
        
        # Serialize source documents to JSON-friendly structure (force JSON-safe types)
        serialized_sources = []
        for doc in source_docs:
            try:
                raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                src = raw_meta.get('source', None)
                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                if src is not None:
                    src = str(src)
                safe_meta = {}
                for k, v in (raw_meta or {}).items():
                    try:
                        json.dumps(v)
                        safe_meta[str(k)] = v
                    except (TypeError, ValueError):
                        safe_meta[str(k)] = str(v)
            except Exception:
                src = None
                safe_meta = {}
                page_content = str(doc)

            serialized_sources.append({
                "source": src,
                "page_content": page_content,
                "metadata": safe_meta
            })

        # Add AI message to database
        db_ai_start = time.perf_counter()
        add_message(chat_id, "assistant", ai_response, metadata={"source_documents": serialized_sources})
        db_ai_elapsed = time.perf_counter() - db_ai_start
        
        log_timing(api_logger, "DB_ADD_AI_MESSAGE", {
            'duration_ms': round(db_ai_elapsed * 1000, 2),
            'response_length': len(ai_response)
        })

        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, "SUCCESS")

        return {
            "user_message": user_query,
            "ai_response": ai_response,
            "sources_count": len(serialized_sources),
            "source_documents": serialized_sources,
            "language": message.language,
            "timings": {"total_ms": round(total_time * 1000, 2)}
        }
    except HTTPException:
        raise
    except Exception as e:
        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, f"FAILED: {type(e).__name__}")
        api_logger.error(f"Error in send_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WebSocket for Streaming ====================

@app.websocket("/api/ws/chats/{chat_id}/stream")
async def websocket_stream(websocket: WebSocket, chat_id: str):
    """WebSocket endpoint for real-time message streaming"""
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            user_query = data.get("message", "")
            language = data.get("language", "English")
            
            if not user_query:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message content is required"
                })
                continue
            
            log_query_start(api_logger, request_id, user_query[:100])
            
            # Add user message to database
            db_start = time.perf_counter()
            add_message(chat_id, "user", user_query)
            db_elapsed = time.perf_counter() - db_start
            
            log_timing(api_logger, "DB_ADD_USER_WS", {
                'duration_ms': round(db_elapsed * 1000, 2)
            })
            
            # Notify client that user message received
            await websocket.send_json({
                "type": "user_received",
                "message": user_query
            })

            # FIX 6: Immediately signal the frontend that processing has started.
            # This eliminates the blank-wait UX during the condense + retrieval phase
            # (which can take 1-6 seconds before the first answer token arrives).
            await websocket.send_json({
                "type": "thinking",
                "message": "Searching knowledge base..."
            })

            # Get chat history
            hist_start = time.perf_counter()
            messages = get_messages(chat_id)
            chat_history = [
                (msg[0], msg[1])
                for msg in messages[:-1]  # Exclude latest user message
                if msg[0] in ["user", "assistant"]
            ]
            
            # Reconstruct alternating pairs
            reconstructed_history = []
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    reconstructed_history.append((chat_history[i][1], chat_history[i+1][1]))
            
            hist_elapsed = time.perf_counter() - hist_start
            log_timing(api_logger, "HISTORY_RETRIEVAL_WS", {
                'duration_ms': round(hist_elapsed * 1000, 2),
                'history_items': len(reconstructed_history)
            })
            
            # Add language instruction
            if language == "Hindi":
                query = user_query + "\n\nIMPORTANT: Respond to this question in Hindi (हिंदी में उत्तर दें)."
            else:
                query = user_query
            
            # Stream the response and measure timings
            stream_start = time.perf_counter()
            full_response = ""
            source_docs = []
            serialized_sources = []  # initialize here — avoids UnboundLocalError on error path
            first_chunk_time = None
            chunk_count = 0

            # Use an asyncio.Queue to receive chunks from a background thread.
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            cancelled = threading.Event()

            def run_blocking_stream():
                try:
                    for stream_output in get_qa_chain().stream({
                        "question": query,
                        "chat_history": reconstructed_history
                    }):
                        if cancelled.is_set():
                            break
                        # Push each chunk into the asyncio queue from this thread
                        loop.call_soon_threadsafe(queue.put_nowait, stream_output)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": str(e)})

            # Start background thread for blocking streaming work
            thread = threading.Thread(target=run_blocking_stream, daemon=True)
            thread.start()

            try:
                while True:
                    # Wait for the next chunk from the background thread
                    stream_output = await queue.get()

                    if stream_output.get('type') == 'chunk':
                        chunk_content = stream_output['content']
                        chunk_count += 1
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                        full_response += chunk_content
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk_content
                        })

                    elif stream_output.get('type') == 'complete':
                        full_response = stream_output.get('answer', full_response)
                        source_docs = stream_output.get('source_documents', [])
                        total_stream_elapsed = time.perf_counter() - stream_start

                        # Serialize source documents to JSON-friendly structure
                        serialized_sources = []
                        for doc in source_docs:
                            try:
                                raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                                src = raw_meta.get('source', None)
                                page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                if src is not None:
                                    src = str(src)
                                safe_meta = {}
                                for k, v in (raw_meta or {}).items():
                                    try:
                                        json.dumps(v)
                                        safe_meta[str(k)] = v
                                    except (TypeError, ValueError):
                                        safe_meta[str(k)] = str(v)
                            except Exception:
                                src = None
                                safe_meta = {}
                                page_content = str(doc)

                            serialized_sources.append({
                                "source": src,
                                "page_content": page_content,
                                "metadata": safe_meta
                            })

                        log_timing(api_logger, "STREAMING_COMPLETE", {
                            'duration_ms': round(total_stream_elapsed * 1000, 2),
                            'chunks': chunk_count,
                            'response_length': len(full_response),
                            'first_chunk_ms': round(first_chunk_time * 1000, 2) if first_chunk_time else None
                        })

                        await websocket.send_json({
                            "type": "complete",
                            "content": full_response,
                            "sources_count": len(serialized_sources),
                            "source_documents": serialized_sources,
                            "timings": {
                                "first_chunk_ms": round(first_chunk_time * 1000, 2) if first_chunk_time else None,
                                "total_ms": round(total_stream_elapsed * 1000, 2)
                            }
                        })
                        break

                    elif stream_output.get('type') == 'error':
                        await websocket.send_json({
                            "type": "error",
                            "error": stream_output.get('content')
                        })
                        break

                # Add AI response to database after streaming completes (include sources metadata)
                db_ai_start = time.perf_counter()
                add_message(chat_id, "assistant", full_response, metadata={"source_documents": serialized_sources})
                db_ai_elapsed = time.perf_counter() - db_ai_start
                
                log_timing(api_logger, "DB_ADD_AI_WS", {
                    'duration_ms': round(db_ai_elapsed * 1000, 2),
                    'response_length': len(full_response)
                })
                
                total_elapsed = time.perf_counter() - request_start
                log_query_complete(api_logger, request_id, total_elapsed, "SUCCESS")

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Error generating response: {str(e)}"
                })
                total_elapsed = time.perf_counter() - request_start
                log_query_complete(api_logger, request_id, total_elapsed, f"FAILED: {type(e).__name__}")
    
    except WebSocketDisconnect:
        cancelled.set()
        api_logger.info(f"WebSocket client disconnected - request_id={request_id}")
    except Exception as e:
        api_logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

# ==================== Image Analysis Endpoint ====================



from fastapi import File, UploadFile, Form
from PIL import Image as PILImage
import io

@app.post("/api/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    chat_id: str = Form(...),
    language: str = Form("English"),
    trigger_rag: bool = Form(True),
):
    """
    Analyze an uploaded potato image using CLIP reference matching.
    Returns disease prediction + confidence + optional RAG explanation.
    Classification runs locally (zero API cost). Only the RAG answer uses LLM.
    """
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()

    try:
        analyzer = get_image_analyzer()
        if not IMAGE_ANALYSIS_AVAILABLE or analyzer is None:
            raise HTTPException(
                status_code=503,
                detail="Image analysis not available. Install transformers+torch and run build_reference_index."
            )

        # Read and validate image
        image_bytes = await image.read()
        try:
            pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        log_query_start(api_logger, request_id, f"IMAGE_ANALYSIS: {image.filename}")

        # CLIP analysis (runs locally — $0)
        analysis_start = time.perf_counter()
        analysis = analyzer.analyze_image(pil_image, top_k_diseases=5, top_k_ref_images=10)
        analysis_elapsed = time.perf_counter() - analysis_start

        log_timing(api_logger, "CLIP_IMAGE_ANALYSIS", {
            'duration_ms': round(analysis_elapsed * 1000, 2),
            'prediction': analysis['display_name'],
            'confidence': analysis['confidence'],
        })

        # Save image for future training
        from src.image_analyzer import REFERENCE_IMAGES_DIR
        import os
        
        # Create safe directory name from prediction
        folder_name = analysis['prediction'].replace(" ", "_")
        upload_dir = os.path.join(REFERENCE_IMAGES_DIR, folder_name)
        os.makedirs(upload_dir, exist_ok=True)
        
        saved_image_name = f"{folder_name}/image_{request_id}.jpg"
        saved_image_path = os.path.join(REFERENCE_IMAGES_DIR, saved_image_name)
        
        pil_image.save(saved_image_path, "JPEG")
        image_url = f"/images/{saved_image_name}"

        # Build response
        response_data = {
            "prediction": analysis['display_name'],
            "confidence": analysis['confidence'],
            "image_url": image_url,
            "top_candidates": analysis['all_candidates'],
            "matched_ref_images": [
                {
                    "image_path": img['image_path'],
                    "disease": img['display_name'],
                    "similarity_score": round(img['similarity_score'], 4),
                }
                for img in analysis['matched_ref_images'][:5]
            ],
            "rag_query": analysis['rag_query'],
            "rag_response": None,
            "source_documents": [],
        }

        # Optionally trigger RAG for detailed explanation (1 LLM call)
        if trigger_rag and get_qa_chain() is not None:
            rag_query = analysis['rag_query']
            if language == "Hindi":
                rag_query += "\n\nIMPORTANT: Respond in Hindi (\u0939\u093f\u0902\u0926\u0940 \u092e\u0947\u0902 \u0909\u0924\u094d\u0924\u0930 \u0926\u0947\u0902)."

            # Save user message — include base64 image + CLIP analysis in metadata
            import base64 as _b64
            image_b64 = _b64.b64encode(image_bytes).decode('utf-8')
            user_msg = f"[Image uploaded: {image.filename}]\n{analysis['rag_query']}"
            user_meta = {
                "image_analysis": True,
                "image_b64": image_b64,
                "image_filename": image.filename,
            }
            add_message(chat_id, "user", user_msg, metadata=user_meta)

            # Get chat history
            messages_db = get_messages(chat_id)
            chat_history_raw = [
                (msg[0], msg[1])
                for msg in messages_db[:-1]
                if msg[0] in ["user", "assistant"]
            ]
            reconstructed_history = []
            for i in range(0, len(chat_history_raw), 2):
                if i + 1 < len(chat_history_raw):
                    reconstructed_history.append((chat_history_raw[i][1], chat_history_raw[i+1][1]))

            # Invoke RAG chain
            chain_start = time.perf_counter()
            result = get_qa_chain().invoke({
                "question": rag_query,
                "chat_history": reconstructed_history
            })
            log_timing(api_logger, "IMAGE_RAG_CHAIN", {
                'duration_ms': round((time.perf_counter() - chain_start) * 1000, 2)
            })

            ai_response = result.get("answer", "")
            source_docs = result.get("source_documents", [])

            # Serialize sources
            serialized_sources = []
            for doc in source_docs:
                try:
                    raw_meta = doc.metadata if hasattr(doc, 'metadata') else {}
                    src = raw_meta.get('source', None)
                    page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    safe_meta = {}
                    for k, v in (raw_meta or {}).items():
                        try:
                            json.dumps(v)
                            safe_meta[str(k)] = v
                        except (TypeError, ValueError):
                            safe_meta[str(k)] = str(v)
                except Exception:
                    src = None
                    safe_meta = {}
                    page_content = str(doc)
                serialized_sources.append({
                    "source": src,
                    "page_content": page_content,
                    "metadata": safe_meta
                })

            assistant_meta = {
                "source_documents": serialized_sources,
                "image_analysis": True,
                "prediction": analysis['display_name'],
                "confidence": analysis['confidence'],
                "top_candidates": analysis['all_candidates'],
                "matched_ref_images": [
                    {
                        "image_path": img['image_path'],
                        "disease": img['display_name'],
                        "similarity_score": round(img['similarity_score'], 4),
                    }
                    for img in analysis['matched_ref_images'][:5]
                ],
            }
            add_message(chat_id, "assistant", ai_response, metadata=assistant_meta)

            response_data["rag_response"] = ai_response
            response_data["source_documents"] = serialized_sources

        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, "SUCCESS")
        response_data["timings"] = {"total_ms": round(total_time * 1000, 2)}
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, f"FAILED: {type(e).__name__}")
        api_logger.error(f"Error in analyze_image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Debug / Monitoring ====================

@app.get("/api/debug/memory")
async def memory_debug():
    """Check current RAM usage of this process (useful on memory-constrained hosts)"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info()
        return {
            "rss_mb": round(mem.rss / 1024 / 1024, 1),
            "vms_mb": round(mem.vms / 1024 / 1024, 1),
            "chain_loaded": qa_chain is not None,
            "clip_loaded": image_analyzer is not None,
            "bm25_enabled": os.getenv("ENABLE_BM25", "true").lower() == "true",
        }
    except ImportError:
        return {"error": "psutil not installed. Add psutil to requirements.txt"}

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error"
        },
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Aloo Sahayak API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "status": "running"
    }

if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting API on {API_HOST}:{API_PORT}")
    print(f"Database: {DATABASE_TYPE}")
    print(f"Debug: {DEBUG}")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/api/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="debug" if DEBUG else "info",
        reload=DEBUG
    )