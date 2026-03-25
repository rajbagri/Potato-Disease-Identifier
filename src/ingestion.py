

import os
import fitz  # PyMuPDF
import base64
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

load_dotenv()

# --- Configuration ---
PDF_DIRECTORY = "data"
FAISS_INDEX_PATH = "faiss_index_multimodal"
IMAGE_SAVE_DIRECTORY = "extracted_images"

def extract_text_and_images(pdf_path: str):
    print(f"Processing {pdf_path}...")
    
    doc = fitz.open(pdf_path)
    all_text = ""
    image_paths = []
    os.makedirs(IMAGE_SAVE_DIRECTORY, exist_ok=True)
    for page_num, page in enumerate(doc):
        all_text += page.get_text() + "\n"
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{os.path.basename(pdf_path)}_p{page_num+1}_img{img_index}.{image_ext}"
            image_path = os.path.join(IMAGE_SAVE_DIRECTORY, image_filename)
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)
    return all_text, image_paths

def is_image_relevant(image_path: str, llm: ChatOpenAI) -> bool:
    """
    Uses gpt-4o to quickly classify if an image is relevant.
    """
    print(f"  -> Classifying '{os.path.basename(image_path)}'...")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        Analyze the following image. Is the primary subject a potato tuber, a potato plant, a potato leaf, a microscopic view of a pathogen (like spores or fungi), or a graph/chart related to agriculture?
        Answer with only a single word: "Yes" or "No".
        """
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        )
        response = llm.invoke([message])
        answer = response.content.strip().lower()
        print(f"  -> Classification result: {answer}")
        return "yes" in answer
    except Exception as e:
        print(f"  -> Error during classification: {e}")
        return False

def describe_image_with_openai(image_path: str, llm: ChatOpenAI) -> str:
    """
    Uses gpt-4o to generate a detailed description for a relevant image.
    """
    print(f"  -> Generating detailed description for '{os.path.basename(image_path)}'...")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        You are an expert plant pathologist. Describe this image from an agricultural research paper in detail. Focus on any visible potato disease symptoms, their color, texture, and shape. Be precise and informative.
        """ 
        
        message = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}])
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"  -> Error during description: {e}")
        return "No description generated."

def run_ingestion():
    print("Starting multi-modal ingestion pipeline with AI filtering...")
    all_chunks = []
    
    # Initialize the LLM once to reuse it for all API calls
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    for filename in os.listdir(PDF_DIRECTORY):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIRECTORY, filename)
            text, image_paths = extract_text_and_images(pdf_path)
            
            # Improved chunking strategy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Slightly larger chunks
                chunk_overlap=200,  # Increased overlap for better context
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],  # Better separators
                length_function=len,
                is_separator_regex=False
            )
            text_chunks = text_splitter.create_documents([text], metadatas=[{"source": filename, "type": "text"}])
            all_chunks.extend(text_chunks)
            print(f"Created {len(text_chunks)} text chunks for {filename}.")
            
            relevant_image_count = 0
            for image_path in image_paths:
                # STAGE 1: CLASSIFY AND FILTER
                if is_image_relevant(image_path, llm):
                    # STAGE 2: DESCRIBE (only if relevant)
                    description = describe_image_with_openai(image_path, llm)
                    # Enhanced image document with better context
                    enhanced_description = f"Image Description for {os.path.basename(image_path)}: {description}\n\nThis image is from document: {filename}. Visual content related to potato diseases, symptoms, or agricultural practices."
                    image_doc = Document(
                        page_content=enhanced_description,
                        metadata={
                            "source": filename, 
                            "image_path": image_path, 
                            "type": "image_description",
                            "image_name": os.path.basename(image_path)
                        }
                    )
                    all_chunks.append(image_doc)
                    relevant_image_count += 1
                else:
                    print(f"  -> Skipping irrelevant image: {os.path.basename(image_path)}")
            
            print(f"Created descriptions for {relevant_image_count} relevant images in {filename}.")

    if all_chunks:
        print("\nInitializing OpenAI embedding model for multimodal data...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        print(f"Creating FAISS index for {len(all_chunks)} total chunks...")
        vector_store = FAISS.from_documents(all_chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        print(f"Multimodal FAISS index created and saved at {FAISS_INDEX_PATH}")
    else:
        print("No documents or images processed.")

if __name__ == "__main__":
    run_ingestion()


    

