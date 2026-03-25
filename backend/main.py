import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
    MessageCreate, MessageResponse, HealthResponse
)
from src.chat_db import (
    init_db, add_chat, get_chats, get_messages, 
    add_message, rename_chat as db_rename_chat, 
    delete_chat as db_delete_chat
)
from src.generation import create_conversational_chain
from src.retrieval import load_retriever_from_disk
from src.logging_utils import setup_logger, log_timing, log_query_start, log_query_complete

# Initialize logger
api_logger = setup_logger('api')

# FastAPI app
app = FastAPI(
    title="Aloo Sahayak API",
    version="1.0.0",
    docs_url="/api/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if isinstance(CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBALS (lazy loaded)
qa_chain = None
retriever = None

# ==================== LAZY LOADER ====================
def get_chain():
    global qa_chain, retriever

    if qa_chain is None:
        print("⚡ Lazy loading RAG chain...")

        if not os.path.exists("faiss_index_multimodal"):
            raise Exception("FAISS index missing. Build locally before deploy.")

        retriever = load_retriever_from_disk()
        qa_chain = create_conversational_chain(retriever)

    return qa_chain

# ==================== STARTUP ====================
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting API...")

    try:
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ DB Error: {e}")
        raise

    print("⚠️ RAG will load lazily (first request)")
    print("⚠️ Image analysis disabled for low memory mode")

# ==================== HEALTH ====================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="API running",
        database=DATABASE_TYPE
    )

# ==================== CHAT ====================
@app.get("/api/chats", response_model=List[ChatResponse])
async def list_chats():
    chats_data = get_chats()
    chats = []

    for chat_id, name, created_at in chats_data:
        messages = get_messages(chat_id)
        chats.append(ChatResponse(
            id=chat_id,
            name=name,
            created_at=created_at,
            message_count=len(messages)//2
        ))

    return chats

@app.post("/api/chats", response_model=ChatResponse)
async def create_chat(chat: ChatCreate):
    chat_id = str(uuid.uuid4())
    add_chat(chat_id, chat.name)

    return ChatResponse(
        id=chat_id,
        name=chat.name,
        created_at=get_chats()[0][2],
        message_count=0
    )

# ==================== MESSAGE ====================
@app.post("/api/chats/{chat_id}/messages")
async def send_message(chat_id: str, message: MessageCreate):
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()

    try:
        log_query_start(api_logger, request_id, message.content[:100])

        qa = get_chain()

        user_query = message.content
        add_message(chat_id, "user", user_query)

        messages = get_messages(chat_id)
        chat_history = [
            (msg[0], msg[1]) for msg in messages[:-1]
            if msg[0] in ["user", "assistant"]
        ]

        reconstructed = []
        for i in range(0, len(chat_history), 2):
            if i+1 < len(chat_history):
                reconstructed.append((chat_history[i][1], chat_history[i+1][1]))

        # LLM call
        result = qa.invoke({
            "question": user_query,
            "chat_history": reconstructed
        })

        ai_response = result.get("answer", "")

        add_message(chat_id, "assistant", ai_response)

        total_time = time.perf_counter() - request_start
        log_query_complete(api_logger, request_id, total_time, "SUCCESS")

        return {
            "user_message": user_query,
            "ai_response": ai_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== DISABLED IMAGE ====================
@app.post("/api/analyze-image")
async def analyze_image():
    return {
        "message": "Image analysis disabled in low-memory deployment"
    }

# ==================== ROOT ====================
@app.get("/")
async def root():
    return {
        "message": "Aloo Sahayak API",
        "status": "running",
        "mode": "low-memory"
    }

# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG
    )