# --- Python Path Setup ---
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

import streamlit as st
from src.retrieval import load_retriever_from_disk
from src.generation import create_conversational_chain
import datetime
import uuid

# --- 1. Import Multimodal Generator ---
try:
    from src.multimodal_generation import create_multimodal_generator
    MULTIMODAL_AVAILABLE = True
except ImportError:
    print("Multimodal module not found. Running in text-only mode.")
    MULTIMODAL_AVAILABLE = False

# --- 2. Import Enhanced Processing ---
try:
    from src.query_processor import QueryProcessor, ContextFilter
    ENHANCED_PROCESSING = True
except ImportError:
    print("Enhanced processing not available, using basic mode")
    ENHANCED_PROCESSING = False

# --- 3. Import Database Functions ---
try:
    from src.chat_db import (
        init_db, add_chat, get_chats, get_messages, 
        add_message, rename_chat as db_rename_chat, delete_chat as db_delete_chat
    )
    DB_AVAILABLE = True
except ImportError:
    print("Database module not found. Please ensure src/chat_db.py exists.")
    DB_AVAILABLE = False

# --- Page Setup ---
st.set_page_config(page_title="Aloo Sahayak 🥔", layout="wide")
st.title("🥔 Aloo Sahayak: Your Potato Disease Assistant")
st.caption("Ask me about potato diseases based on the provided documents!")

# Initialize DB
if DB_AVAILABLE:
    init_db()

# Initialize response language preference
if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

# --- Chat History Management (SQL Based) ---
def load_chats_from_db():
    """Load all chats from database"""
    if not DB_AVAILABLE: return None
    try:
        chats_data = get_chats()
        chats = {}
        
        for chat_id, name, created_at in chats_data:
            messages = get_messages(chat_id)
            # Convert database messages to chat format
            message_list = [(sender, content) for sender, content, _ in messages]
            
            # Create LangChain style history (User/AI pairs)
            chat_history = []
            for i in range(0, len(message_list) - 1, 2):
                if message_list[i][0] == 'user' and message_list[i+1][0] == 'assistant':
                    chat_history.append((message_list[i][1], message_list[i+1][1]))
            
            chats[chat_id] = {
                "name": name,
                "messages": message_list,
                "chat_history": chat_history,
                "created_at": datetime.datetime.fromisoformat(created_at)
            }
        
        return chats if chats else None
    except Exception as e:
        print(f"Error loading chats from database: {e}")
        return None

if "chats" not in st.session_state:
    loaded_chats = load_chats_from_db()
    if loaded_chats:
        st.session_state.chats = loaded_chats
        st.session_state.active_chat_id = list(loaded_chats.keys())[0]
    else:
        # Create default chat
        default_chat_id = str(uuid.uuid4())
        if DB_AVAILABLE: add_chat(default_chat_id, "New Chat")
        st.session_state.chats = {
            default_chat_id: {
                "name": "New Chat",
                "messages": [],
                "chat_history": [],
                "created_at": datetime.datetime.now()
            }
        }
        st.session_state.active_chat_id = default_chat_id

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]

# --- Helper Functions ---
def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    if DB_AVAILABLE: add_chat(new_chat_id, "New Chat")
    st.session_state.chats[new_chat_id] = {
        "name": "New Chat",
        "messages": [],
        "chat_history": [],
        "created_at": datetime.datetime.now()
    }
    st.session_state.active_chat_id = new_chat_id

def delete_chat(chat_id):
    if len(st.session_state.chats) > 1 and chat_id in st.session_state.chats:
        if DB_AVAILABLE: db_delete_chat(chat_id)
        del st.session_state.chats[chat_id]
        st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]

def auto_rename_chat(chat_id, first_message):
    if st.session_state.chats[chat_id]["name"] == "New Chat":
        name = first_message[:50] + ("..." if len(first_message) > 50 else "")
        st.session_state.chats[chat_id]["name"] = name
        if DB_AVAILABLE: db_rename_chat(chat_id, name)

def rename_chat(chat_id, new_name):
    if chat_id in st.session_state.chats and new_name.strip():
        new_name_stripped = new_name.strip()
        st.session_state.chats[chat_id]["name"] = new_name_stripped
        if DB_AVAILABLE: db_rename_chat(chat_id, new_name_stripped)

# Get current active chat
active_chat = st.session_state.chats[st.session_state.active_chat_id]

# --- Sidebar UI ---
with st.sidebar:
    st.header("💬 Conversations")
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()
    
    st.divider()
    st.subheader("📋 Chat History")
    
    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    for chat_id, chat_data in sorted_chats:
        col1, col2 = st.columns([5, 1])
        with col1:
            is_active = chat_id == st.session_state.active_chat_id
            msg_count = len(chat_data["messages"]) // 2
            chat_label = f"{'🟢 ' if is_active else ''}{chat_data['name']}"
            if msg_count > 0: chat_label += f" ({msg_count})"
            
            if st.button(chat_label, key=f"chat_{chat_id}", use_container_width=True):
                if not is_active:
                    st.session_state.active_chat_id = chat_id
                    st.rerun()
        
        with col2:
            with st.popover("⋮", use_container_width=True):
                st.markdown(f"**Options**")
                new_name = st.text_input("Rename", value=chat_data['name'], key=f"rename_{chat_id}")
                if st.button("Save Name", key=f"ren_btn_{chat_id}"):
                    rename_chat(chat_id, new_name)
                    st.rerun()
                if len(st.session_state.chats) > 1:
                    if st.button("Delete", key=f"del_{chat_id}", type="primary"):
                        delete_chat(chat_id)
                        st.rerun()

    st.divider()
    
    # Language Toggle
    col_lang1, col_lang2 = st.columns(2)
    with col_lang1:
        if st.button("🇬🇧 English", key="lang_en", type="primary" if st.session_state.response_language == "English" else "secondary"):
            st.session_state.response_language = "English"
            st.rerun()
    with col_lang2:
        if st.button("🇮🇳 हिंदी", key="lang_hi", type="primary" if st.session_state.response_language == "Hindi" else "secondary"):
            st.session_state.response_language = "Hindi"
            st.rerun()
    
    st.divider()
    # System Status
    st.subheader("📊 System Status")
    if ENHANCED_PROCESSING: st.success("✅ Enhanced Processing")
    if MULTIMODAL_AVAILABLE: st.success("✅ Multimodal Vision")
    if DB_AVAILABLE: st.success("✅ Database Connected")

# --- Logic Initialization ---
@st.cache_resource
def load_chain():
    retriever = load_retriever_from_disk()
    chain = create_conversational_chain(retriever)
    return chain

@st.cache_resource 
def load_processors():
    if ENHANCED_PROCESSING:
        return QueryProcessor(), ContextFilter()
    return None, None

@st.cache_resource
def load_multimodal():
    if MULTIMODAL_AVAILABLE:
        # Assumes FAISS index is at this path
        return create_multimodal_generator("faiss_index_multimodal")
    return None

qa_chain = load_chain()
query_processor, context_filter = load_processors()
multimodal_generator = load_multimodal()

# --- Display Messages ---
for role, content in active_chat["messages"]:
    with st.chat_message(role):
        st.markdown(content)

# --- Handle Input ---
user_query = st.chat_input("Ask about potato diseases...")

if user_query:
    if len(active_chat["messages"]) == 0:
        auto_rename_chat(st.session_state.active_chat_id, user_query)
    
    active_chat["messages"].append(("user", user_query))
    if DB_AVAILABLE: add_message(st.session_state.active_chat_id, "user", user_query)
    
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        try:
            # 1. Enhance Query
            if ENHANCED_PROCESSING and query_processor:
                processed_query = query_processor.preprocess_question(user_query, active_chat["chat_history"])
                query_to_use = processed_query['primary_query']
            else:
                query_to_use = user_query
            
            # 2. Add Language Instruction
            if st.session_state.response_language == "Hindi":
                query_with_language = query_to_use + "\n\nIMPORTANT: Respond in Hindi (हिंदी में उत्तर दें)."
            else:
                query_with_language = query_to_use
            
            # 3. Retrieve Documents First (Required for Multimodal check)
            retrieval_result = qa_chain.invoke({
                "question": query_with_language,
                "chat_history": active_chat["chat_history"]
            })
            source_documents = retrieval_result.get("source_documents", [])
            
            # 4. Filter Contexts
            if ENHANCED_PROCESSING and context_filter and source_documents:
                source_documents = context_filter.filter_contexts(source_documents, user_query, max_contexts=6)

            # 5. Stream Answer
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # Stream the response
                for chunk in qa_chain.stream({
                    "question": query_with_language,
                    "chat_history": active_chat["chat_history"]
                }):
                    if 'answer' in chunk:
                        full_response += chunk['answer']
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                ai_response = full_response

                # 6. Multimodal Image Display (Post-generation check)
                if multimodal_generator:
                    multimodal_result = multimodal_generator.generate_multimodal_response(
                        question=user_query,
                        text_answer=ai_response,
                        retrieved_contexts=source_documents
                    )
                    
                    if multimodal_result['has_images']:
                        st.markdown("### 📸 Visual References" if st.session_state.response_language == "English" else "### 📸 दृश्य संदर्भ (Visual References)")
                        
                        images_to_display = multimodal_result['images']
                        cols = st.columns(min(len(images_to_display), 3))
                        
                        for idx, img in enumerate(images_to_display):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                if os.path.exists(img['path']):
                                    st.image(
                                        img['path'], 
                                        caption=f"{img.get('image_name', 'Image')}",
                                        use_container_width=True
                                    )
                                    with st.expander("Details"):
                                        st.caption(img['caption'])
                                else:
                                    st.error(f"Image not found: {img['path']}")

                # 7. Display Sources
                if source_documents:
                    st.markdown("---")
                    with st.expander(f"📚 View Sources ({len(source_documents)} found)"):
                        for i, doc in enumerate(source_documents):
                            source_name = doc.metadata.get('source', 'Unknown Source')
                            doc_type = doc.metadata.get('type', 'text')
                            icon = "🖼️" if doc_type == 'image_description' else "📄"
                            st.markdown(f"**{icon} Source {i+1}:** `{source_name}`")
                            st.caption(doc.page_content[:200] + "...")
                            st.divider()

            # Save Assistant Message
            active_chat["messages"].append(("assistant", ai_response))
            active_chat["chat_history"].append((user_query, ai_response))
            if DB_AVAILABLE:
                # Serialize source_documents if available
                serialized_sources = []
                try:
                    for doc in source_documents:
                        try:
                            src = doc.metadata.get('source', None) if hasattr(doc, 'metadata') else None
                            meta = doc.metadata if hasattr(doc, 'metadata') else {}
                            page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        except Exception:
                            src = None
                            meta = {}
                            page_content = str(doc)
                        serialized_sources.append({
                            "source": src,
                            "page_content": page_content,
                            "metadata": meta
                        })
                except Exception:
                    serialized_sources = []

                add_message(st.session_state.active_chat_id, "assistant", ai_response, metadata={"source_documents": serialized_sources})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.text(traceback.format_exc())