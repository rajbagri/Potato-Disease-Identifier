"""
Aloo Sahayak API Client

Simple client library for interacting with the FastAPI backend.
Use this in your Streamlit app or other frontends.
"""

import requests
import json
from typing import List, Dict, Optional, Generator
import asyncio
import websockets

class AlooPotatoClient:
    """Client for Aloo Sahayak API"""
   
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        self.session = requests.Session()
   
    # ==================== Health ====================
   
    def health_check(self) -> Dict:
        """Check API health status"""
        response = self.session.get(f"{self.api_url}/health")
        response.raise_for_status()
        return response.json()
   
    # ==================== Chats ====================
   
    def list_chats(self) -> List[Dict]:
        """Get all chats"""
        response = self.session.get(f"{self.api_url}/chats")
        response.raise_for_status()
        return response.json()
   
    def create_chat(self, name: str = "New Chat") -> Dict:
        """Create a new chat"""
        response = self.session.post(
            f"{self.api_url}/chats",
            json={"name": name}
        )
        response.raise_for_status()
        return response.json()
   
    def get_chat(self, chat_id: str) -> Dict:
        """Get chat details with messages"""
        response = self.session.get(f"{self.api_url}/chats/{chat_id}")
        response.raise_for_status()
        return response.json()
   
    def rename_chat(self, chat_id: str, new_name: str) -> Dict:
        """Rename a chat"""
        response = self.session.put(
            f"{self.api_url}/chats/{chat_id}",
            json={"new_name": new_name}
        )
        response.raise_for_status()
        return response.json()
   
    def delete_chat(self, chat_id: str) -> Dict:
        """Delete a chat"""
        response = self.session.delete(f"{self.api_url}/chats/{chat_id}")
        response.raise_for_status()
        return response.json()
   
    # ==================== Messages ====================
   
    def get_messages(self, chat_id: str) -> List[Dict]:
        """Get all messages for a chat"""
        response = self.session.get(f"{self.api_url}/chats/{chat_id}/messages")
        response.raise_for_status()
        return response.json()
   
    def send_message(
        self,
        chat_id: str,
        content: str,
        language: str = "English"
    ) -> Dict:
        """Send a message and get AI response (blocking)"""
        response = self.session.post(
            f"{self.api_url}/chats/{chat_id}/messages",
            json={
                "content": content,
                "language": language
            }
        )
        response.raise_for_status()
        return response.json()
   
    # ==================== Image Analysis ====================
   
    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        chat_id: str,
        language: str = "English",
        trigger_rag: bool = True,
    ) -> Dict:
        """
        Upload an image for CLIP disease analysis.
        Classification is free (runs locally). RAG explanation costs 1 LLM call.
        """
        response = self.session.post(
            f"{self.api_url}/analyze-image",
            files={"image": (filename, image_bytes, "image/jpeg")},
            data={
                "chat_id": chat_id,
                "language": language,
                "trigger_rag": str(trigger_rag).lower(),
            },
            timeout=120,  # CLIP + RAG can take time on first load
        )
        response.raise_for_status()
        return response.json()
   
    # ==================== WebSocket Streaming ====================
   
    async def stream_message(
        self,
        chat_id: str,
        content: str,
        language: str = "English"
    ) -> Generator:
        """Stream message response in real-time"""
       
        ws_url = f"ws://127.0.0.1:8000/api/ws/chats/{chat_id}/stream"
       
        try:
            async with websockets.connect(ws_url) as websocket:
                # Send message
                await websocket.send(json.dumps({
                    "message": content,
                    "language": language
                }))
               
                # Receive chunks
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    yield data
                   
                    if data.get("type") == "complete":
                        break
       
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
   
    def stream_message_sync(
        self,
        chat_id: str,
        content: str,
        language: str = "English"
    ) -> Generator:
        """Synchronous wrapper for streaming (use in Streamlit)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_generator = None

        try:
            async_generator = self.stream_message(chat_id, content, language)
            while True:
                try:
                    yield loop.run_until_complete(async_generator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            # Ensure async generator is properly closed to avoid GeneratorExit and pending tasks
            try:
                if async_generator is not None:
                    loop.run_until_complete(async_generator.aclose())
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass


# Example usage in Streamlit

"""
import streamlit as st
from api_client import AlooPotatoClient

client = AlooPotatoClient("http://127.0.0.1:8000")

# List chats
chats = client.list_chats()

# Create new chat
new_chat = client.create_chat("Potato Diseases")

# Send message (blocking)
response = client.send_message(new_chat["id"], "What is rust?")
st.write(response["ai_response"])

# Stream message (real-time)
for chunk in client.stream_message_sync(new_chat["id"], "What is blight?"):
    if chunk.get("type") == "chunk":
        st.write(chunk["content"], end="")
    elif chunk.get("type") == "complete":
        st.write("Response complete!")
    elif chunk.get("type") == "error":
        st.error(chunk["error"])
"""