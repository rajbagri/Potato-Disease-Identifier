import os
from dotenv import load_dotenv  # <-- Add this import
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()  # <-- Add this line

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index_multimodal" 
EMBEDDING_MODEL = "text-embedding-3-small"

def inspect_faiss_index():
    """
    Loads a saved FAISS index and prints its key statistics.
    """
    # ... (the rest of your script remains exactly the same) ...

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Error: FAISS index not found at '{FAISS_INDEX_PATH}'")
        return

    print("Loading FAISS index for inspection...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    db = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("Index loaded successfully.")
    print("-" * 30)

    total_vectors = db.index.ntotal
    print(f"Total number of chunks (vectors): {total_vectors}")

    vector_size = db.index.d
    print(f"Embedding vector size (dimensions): {vector_size}")

    print("-" * 30)
    print("Inspecting a sample document (ID 0):")
    
    docstore = db.docstore
    if len(db.index_to_docstore_id) > 0:
        first_doc_id = db.index_to_docstore_id[0]
        first_document = docstore.search(first_doc_id)
        
        print("\n--- METADATA ---")
        print(first_document.metadata)
        
        print("\n--- CONTENT (first 100 chars) ---")
        print(first_document.page_content[:100] + "...")
    else:
        print("Index appears to be empty.")

if __name__ == "__main__":
    inspect_faiss_index()