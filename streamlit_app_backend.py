# --- Python Path Setup ---
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

import streamlit as st
import requests
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
import time

# Import API client
from backend.api_client import AlooPotatoClient

# Configure page
st.set_page_config(
    page_title="Aloo Sahayak 🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 500;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"

# Initialize API client
@st.cache_resource
def get_api_client():
    return AlooPotatoClient(API_BASE_URL)

api_client = get_api_client()

# ==================== PAGE SETUP ====================

st.title("💬 Aloo Sahayak: Your Potato Disease Assistant")
st.caption("Powered by FastAPI Backend 🚀")

# ==================== SESSION STATE INITIALIZATION ====================

if "chats" not in st.session_state:
    try:
        # Load chats from backend
        chats_data = api_client.list_chats()
        st.session_state.chats = {}
       
        for chat in chats_data:
            chat_id = chat["id"]
            st.session_state.chats[chat_id] = {
                "id": chat_id,
                "name": chat["name"],
                "created_at": chat["created_at"],
                "message_count": chat.get("message_count", 0)
            }
       
        # Set active chat
        if st.session_state.chats:
            st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
        else:
            st.session_state.active_chat_id = None
    except Exception as e:
        st.error(f"Failed to load chats: {str(e)}")
        st.session_state.chats = {}

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

if "api_error" not in st.session_state:
    st.session_state.api_error = None

# Persist source documents for the most recent response across st.rerun() calls.
# Keyed by chat_id so switching chats doesn't show stale sources.
if "pending_sources" not in st.session_state:
    st.session_state.pending_sources = []
if "pending_sources_chat" not in st.session_state:
    st.session_state.pending_sources_chat = None
if "image_analysis_result" not in st.session_state:
    st.session_state.image_analysis_result = None
if "image_analysis_chat" not in st.session_state:
    st.session_state.image_analysis_chat = None
if "image_analysis_uploaded_bytes" not in st.session_state:
    st.session_state.image_analysis_uploaded_bytes = None
if "image_analysis_uploaded_name" not in st.session_state:
    st.session_state.image_analysis_uploaded_name = None
if "upload_key_counter" not in st.session_state:
    st.session_state.upload_key_counter = 0

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("💬 Conversations")
   
    # New Chat Button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            try:
                new_chat = api_client.create_chat("New Chat")
                chat_id = new_chat["id"]
                st.session_state.chats[chat_id] = {
                    "id": chat_id,
                    "name": new_chat["name"],
                    "created_at": new_chat["created_at"],
                    "message_count": 0
                }
                st.session_state.active_chat_id = chat_id
                st.success("✅ New chat created!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create chat: {str(e)}")
   
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            try:
                chats_data = api_client.list_chats()
                st.session_state.chats = {}
                for chat in chats_data:
                    chat_id = chat["id"]
                    st.session_state.chats[chat_id] = {
                        "id": chat_id,
                        "name": chat["name"],
                        "created_at": chat["created_at"],
                        "message_count": chat.get("message_count", 0)
                    }
                if st.session_state.chats and not st.session_state.active_chat_id:
                    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
                st.success("✅ Chats refreshed!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to refresh chats: {str(e)}")
   
    st.divider()
   
    # Chat List
    st.subheader("📋 Chat History")
   
    if st.session_state.chats:
        # Sort chats by creation time (newest first)
        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
       
        for chat_id, chat_data in sorted_chats:
            col1, col2 = st.columns([5, 1])
           
            with col1:
                is_active = chat_id == st.session_state.active_chat_id
                chat_label = f"{'🟢 ' if is_active else '⚪ '}{chat_data['name']}"
               
                if chat_data.get("message_count", 0) > 0:
                    chat_label += f" ({chat_data['message_count']})"
               
                if st.button(
                    chat_label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()
           
            with col2:
                # Three-dot menu
                with st.popover("⋮", use_container_width=True):
                    st.markdown(f"**{chat_data['name'][:25]}...**")
                    st.divider()
                   
                    # Rename
                    new_name = st.text_input(
                        "Rename",
                        value=chat_data['name'],
                        key=f"rename_input_{chat_id}",
                        label_visibility="collapsed"
                    )
                   
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✏️ Rename", key=f"rename_{chat_id}", use_container_width=True):
                            try:
                                if new_name and new_name != chat_data['name']:
                                    api_client.rename_chat(chat_id, new_name)
                                    st.session_state.chats[chat_id]["name"] = new_name
                                    st.success("✅ Chat renamed!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Failed to rename: {str(e)}")
                   
                    # Delete
                    if len(st.session_state.chats) > 1:
                        st.divider()
                        if st.button("🗑️ Delete", key=f"delete_{chat_id}", use_container_width=True, type="secondary"):
                            try:
                                api_client.delete_chat(chat_id)
                                del st.session_state.chats[chat_id]
                                if st.session_state.active_chat_id == chat_id:
                                    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
                                st.success("✅ Chat deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {str(e)}")
                    else:
                        st.divider()
                        st.caption("⚠️ Cannot delete last chat")
    else:
        st.info("No chats yet. Create one to start!")
   
    st.divider()
   
    # Language Toggle
    st.subheader("🌐 Response Language")
   
    col_lang1, col_lang2 = st.columns(2)
    with col_lang1:
        if st.button(
            "🇬🇧 English",
            key="lang_en",
            use_container_width=True,
            type="primary" if st.session_state.response_language == "English" else "secondary"
        ):
            st.session_state.response_language = "English"
            st.rerun()
   
    with col_lang2:
        if st.button(
            "🇮🇳 हिंदी",
            key="lang_hi",
            use_container_width=True,
            type="primary" if st.session_state.response_language == "Hindi" else "secondary"
        ):
            st.session_state.response_language = "Hindi"
            st.rerun()
   
    st.caption(f"💬 **{st.session_state.response_language}**")
   
    st.divider()
   
    # System Status
    st.subheader("📊 System Status")
   
    try:
        health = api_client.health_check()
        st.success(f"✅ API Running")
        st.caption(f"Database: {health['database']}")
    except:
        st.error("❌ API Offline")
        st.caption("Cannot reach backend")
   
    st.divider()
   
    # API Configuration
    st.subheader("⚙️ API Settings")
    st.caption(f"Backend: {API_BASE_URL}")
   
    if st.button("🔌 Test Connection", use_container_width=True):
        try:
            health = api_client.health_check()
            st.success(f"✅ Connected! Status: {health['status']}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")

# ==================== MAIN CHAT AREA ====================

if st.session_state.active_chat_id:
    active_chat_id = st.session_state.active_chat_id
    active_chat_name = st.session_state.chats[active_chat_id]["name"]
   
    st.subheader(f"📝 {active_chat_name}")
   
    # Load and display messages
    try:
        messages = api_client.get_messages(active_chat_id)
       
        if messages:
            # Display messages in a scrollable container
            for msg in messages:
                sender = msg["sender"]
                content = msg["content"]
                timestamp = msg["timestamp"]
               
                if sender == "user":
                    # Check if this is an image analysis message
                    msg_meta = msg.get('metadata') or {}
                    if isinstance(msg_meta, dict) and msg_meta.get('image_analysis'):
                        import base64 as _b64
                        with st.chat_message("user"):
                            image_b64 = msg_meta.get('image_b64')
                            if image_b64:
                                try:
                                    img_data = _b64.b64decode(image_b64)
                                    st.image(img_data, caption=f"Uploaded: {msg_meta.get('image_filename', 'image')}", width=300)
                                except Exception:
                                    pass
                            st.markdown(content)
                            st.caption(timestamp)
                    else:
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 You:</strong><br/>
                            {content}
                            <br/><small style="color:gray;">{timestamp}</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Check if this is an image analysis assistant message
                    a_meta = msg.get('metadata') or {}
                    is_img_analysis = isinstance(a_meta, dict) and a_meta.get('image_analysis')

                    if is_img_analysis:
                        # Render rich CLIP analysis panel within chat message
                        with st.chat_message("assistant"):
                            st.markdown("**🔬 Disease Analysis Results**")
                            conf = a_meta.get('confidence', 0)
                            pred = a_meta.get('prediction', 'Unknown')
                            conf_emoji = "🟢" if conf >= 0.5 else ("🟡" if conf >= 0.3 else "🔴")
                            st.markdown(f"**Predicted:** {pred} {conf_emoji} ({conf:.0%} confidence)")

                            with st.expander("🏥 All Candidate Predictions", expanded=False):
                                for c in a_meta.get('top_candidates', []):
                                    pct = c.get('score', 0) * 100
                                    name = c.get('display_name', c.get('disease', 'Unknown'))
                                    st.markdown(f"- **{name}**: {pct:.1f}%")

                            ref_imgs = a_meta.get('matched_ref_images', [])
                            if ref_imgs:
                                with st.expander(f"📸 Similar Reference Images ({len(ref_imgs)} found)", expanded=False):
                                    for ri in ref_imgs:
                                        img_path = ri.get('image_path', '')
                                        if os.path.exists(img_path):
                                            st.image(
                                                img_path,
                                                caption=f"{ri.get('disease', 'Unknown')} — similarity: {ri.get('similarity_score', 0):.2f}",
                                                use_container_width=True
                                            )
                                        else:
                                            st.caption(f"🖼️ {ri.get('disease', 'Unknown')} (similarity: {ri.get('similarity_score', 0):.2f})")

                            st.markdown("**📖 Detailed Explanation**")
                            st.markdown(content)

                            # Sources
                            source_documents = a_meta.get('source_documents', [])
                            if source_documents:
                                with st.expander(f"📚 View Sources ({len(source_documents)} found)"):
                                    for i, doc in enumerate(source_documents):
                                        source_name = doc.get('source') or (doc.get('metadata', {}).get('source') if doc.get('metadata') else None) or f"Source {i+1}"
                                        st.markdown(f"**Source {i+1}:** {source_name}")
                                        content_preview = (doc.get('page_content') or '')[:300].replace('\n', ' ')
                                        if len(doc.get('page_content', '')) > 300:
                                            content_preview += "..."
                                        st.markdown(f"> {content_preview}")
                                        if i < len(source_documents) - 1:
                                            st.divider()

                            st.caption(f"💡 Ask follow-up questions below. • {timestamp}")
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🥔 Aloo Sahayak:</strong><br/>
                            {content}
                            <br/><small style="color:gray;">{timestamp}</small>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show persisted source documents if saved in message metadata
                        try:
                            source_documents = a_meta.get('source_documents', []) if isinstance(a_meta, dict) else []
                            if source_documents:
                                st.markdown('---')
                                with st.expander(f"📚 View Sources ({len(source_documents)} found)"):
                                    for i, doc in enumerate(source_documents):
                                        source_name = None
                                        try:
                                            source_name = doc.get('source') or (doc.get('metadata', {}).get('source') if doc.get('metadata') else None)
                                        except Exception:
                                            source_name = None

                                        if not source_name:
                                            source_name = f"Source {i+1}"

                                        st.markdown(f"**Source {i+1}:** {source_name}")
                                        content_preview = (doc.get('page_content') or '')[:300].replace('\n', ' ')
                                        if len(doc.get('page_content', '')) > 300:
                                            content_preview += "..."
                                        st.markdown(f"> {content_preview}")
                                        if i < len(source_documents) - 1:
                                            st.divider()
                        except Exception:
                            pass
        else:
            st.info("💬 No messages yet. Start by asking a question!")
   
    except Exception as e:
        st.error(f"Failed to load messages: {str(e)}")

    st.divider()

    # ========== INLINE IMAGE UPLOAD (ChatGPT-style) ==========
    uploaded_image = st.file_uploader(
        "📷 Attach a potato image for disease analysis",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key=f"inline_image_{active_chat_id}_{st.session_state.upload_key_counter}",
    )

    if uploaded_image is not None:
        img_cols = st.columns([1, 3])
        with img_cols[0]:
            st.image(uploaded_image, width=150)
        with img_cols[1]:
            st.caption(f"📄 {uploaded_image.name}")
            if st.button("🔬 Analyze Disease", type="primary", use_container_width=True):
                with st.spinner("Analyzing with CLIP + RAG..."):
                    try:
                        image_bytes = uploaded_image.getvalue()
                        result = api_client.analyze_image(
                            image_bytes=image_bytes,
                            filename=uploaded_image.name,
                            chat_id=active_chat_id,
                            language=st.session_state.response_language,
                            trigger_rag=True,
                        )
                        st.session_state.image_analysis_result = result
                        st.session_state.image_analysis_chat = active_chat_id
                        st.session_state.image_analysis_uploaded_bytes = image_bytes
                        st.session_state.image_analysis_uploaded_name = uploaded_image.name
                        st.session_state.upload_key_counter += 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

    # Chat Input
    col1, col2 = st.columns([5, 1])
   
    with col1:
        user_query = st.chat_input(
            "Ask about potato diseases, fertilizers, cultivation...",
            key=f"chat_input_{active_chat_id}"
        )
   
    with col2:
        stream_toggle = st.checkbox("🌊 Stream", value=True, help="Enable real-time streaming")
   
    if user_query:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)
       
        # Clear sources from prior response so stale docs don't persist
        # if the new response happens to find no relevant sources.
        st.session_state.pending_sources = []
        st.session_state.pending_sources_chat = active_chat_id

        # Get AI response
        with st.chat_message("assistant"):
            st.write("Thinking...")
            response_placeholder = st.empty()
            status_placeholder = st.empty()
           
            try:
                if stream_toggle:
                    # Stream via backend WebSocket using synchronous wrapper
                    response_placeholder.empty()
                    displayed_text = ""

                    for chunk in api_client.stream_message_sync(
                        active_chat_id,
                        user_query,
                        language=st.session_state.response_language
                    ):
                        if chunk.get("type") == "chunk":
                            displayed_text += chunk.get("content", "")
                            response_placeholder.markdown(displayed_text + "▌")
                        elif chunk.get("type") == "complete":
                            ai_response = chunk.get("content", "")
                            sources_count = chunk.get("sources_count", 0)
                            response_placeholder.markdown(ai_response)

                            # Save sources to session_state so the persistent
                            # "Latest Response Sources" panel below survives st.rerun().
                            source_documents = chunk.get("source_documents", [])
                            if source_documents:
                                st.session_state.pending_sources = source_documents
                                st.session_state.pending_sources_chat = active_chat_id

                            status_placeholder.success(f"✅ Response generated ({sources_count} sources)")
                            break
                        elif chunk.get("type") == "error":
                            response_placeholder.error(chunk.get("error"))
                            status_placeholder.error("Failed to get response")
                            break
                else:
                    # Non-streaming response (fallback)
                    response = api_client.send_message(
                        active_chat_id,
                        user_query,
                        language=st.session_state.response_language
                    )

                    ai_response = response["ai_response"]
                    sources_count = response.get("sources_count", 0)

                    response_placeholder.markdown(ai_response)
                    # Save sources to session_state so the persistent
                    # "Latest Response Sources" panel below survives st.rerun().
                    source_documents = response.get("source_documents", [])
                    if source_documents:
                        st.session_state.pending_sources = source_documents
                        st.session_state.pending_sources_chat = active_chat_id

                    status_placeholder.success(f"✅ Response generated ({sources_count} sources)")

                # Refresh chat to show new message
                time.sleep(1)
                st.rerun()

            except Exception as e:
                response_placeholder.error(f"Error: {str(e)}")
                status_placeholder.error(f"Failed to get response")

else:
    st.info("👈 Create a new chat from the sidebar to start!")

# ==================== FOOTER ====================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("💡 **Tip**: Use specific disease names (e.g., 'late blight', 'early blight')")

with col2:
    st.info("🌐 **Languages**: Ask in English, get responses in English or Hindi")

with col3:
    st.info("📚 **Sources**: Each response includes relevant sources from knowledge base")