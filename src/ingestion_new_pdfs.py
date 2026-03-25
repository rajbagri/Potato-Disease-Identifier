"""
Incremental Ingestion Script
Processes only new PDFs and merges them into existing FAISS index
"""

import os
import fitz  # PyMuPDF
import base64
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage

load_dotenv()

# --- Configuration ---
NEW_PDF_DIRECTORY = "data\Fertilization_data"  # Folder with 9 OCR'd PDFs
EXISTING_INDEX_PATH = "faiss_index_multimodal"  # Your current index
IMAGE_SAVE_DIRECTORY = "extracted_images"

def extract_text_and_images(pdf_path: str):
    """Extract text and content images (skip full-page backgrounds)"""
    print(f"Processing {pdf_path}...")
    
    doc = fitz.open(pdf_path)
    all_text = ""
    image_paths = []
    os.makedirs(IMAGE_SAVE_DIRECTORY, exist_ok=True)
    
    for page_num, page in enumerate(doc):
        # Extract text from page
        all_text += page.get_text() + "\n"
        
        # Get page dimensions for filtering full-page images
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        page_area = page_width * page_height
        
        # Extract images from page
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            
            # Get image position and size on the page
            img_rects = page.get_image_rects(xref)
            
            # Skip if we can't get image dimensions
            if not img_rects:
                continue
            
            # Calculate what % of the page this image covers
            img_rect = img_rects[0]
            img_area = abs((img_rect.x1 - img_rect.x0) * (img_rect.y1 - img_rect.y0))
            coverage_ratio = img_area / page_area if page_area > 0 else 0
            
            # Skip full-page images (>85% coverage) - OCR page backgrounds
            if coverage_ratio > 0.85:
                print(f"  ⚠ Skipping full-page background on page {page_num+1} ({coverage_ratio*100:.1f}%)")
                continue
            
            # Extract actual content images only
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Filter by image dimensions (skip tiny images)
            image_width = base_image.get("width", 0)
            image_height = base_image.get("height", 0)
            
            if image_width < 100 or image_height < 100:
                print(f"  ⚠ Skipping tiny image on page {page_num+1} ({image_width}x{image_height}px)")
                continue
            
            # Save the content image
            image_filename = f"{os.path.basename(pdf_path)}_p{page_num+1}_img{img_index}.{image_ext}"
            image_path = os.path.join(IMAGE_SAVE_DIRECTORY, image_filename)
            
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            image_paths.append(image_path)
            print(f"  ✓ Content image: {image_filename} ({image_width}x{image_height}px)")
    
    doc.close()
    return all_text, image_paths

def is_image_relevant(image_path: str, llm: ChatOpenAI) -> bool:
    """Use GPT-4o to classify if image is relevant"""
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
    """Generate detailed description for relevant image"""
    print(f"  -> Generating description for '{os.path.basename(image_path)}'...")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = """
        You are an expert plant pathologist. Describe this image from an agricultural research paper in detail. Focus on any visible potato disease symptoms, their color, texture, and shape. Be precise and informative.
        """ 
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"  -> Error during description: {e}")
        return "No description generated."

def run_incremental_ingestion():
    """Process new PDFs and merge into existing index"""
    
    print("\n" + "="*70)
    print("INCREMENTAL INGESTION - New PDFs Only")
    print("="*70)
    
    # Check if new PDF directory exists
    if not os.path.exists(NEW_PDF_DIRECTORY):
        print(f"\n❌ Directory not found: {NEW_PDF_DIRECTORY}")
        return
    
    # Get list of new PDFs
    pdf_files = [f for f in os.listdir(NEW_PDF_DIRECTORY) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"\n✓ No new PDFs found in {NEW_PDF_DIRECTORY}")
        return
    
    print(f"\nFound {len(pdf_files)} new PDF(s) to process")
    for pdf in pdf_files:
        print(f"  - {pdf}")
    
    # Initialize LLM for image classification/description
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    
    # Process each new PDF
    all_new_chunks = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(NEW_PDF_DIRECTORY, pdf_file)
        
        print(f"\n{'='*70}")
        print(f"Processing: {pdf_file}")
        print(f"{'='*70}")
        
        # Extract text and images
        text, image_paths = extract_text_and_images(pdf_path)
        
        # Chunk text
        print(f"\n  Chunking text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200
        )
        text_chunks = text_splitter.split_text(text)
        
        # Create text documents
        for chunk in text_chunks:
            doc = Document(
                page_content=chunk,
                metadata={'source': pdf_file, 'type': 'text'}
            )
            all_new_chunks.append(doc)
        
        print(f"  ✓ Created {len(text_chunks)} text chunks")
        
        # Process images with AI
        print(f"\n  Processing {len(image_paths)} image(s) with AI...")
        relevant_images = 0
        
        for image_path in image_paths:
            # Classify relevance
            is_relevant = is_image_relevant(image_path, llm)
            
            if is_relevant:
                # Generate description
                description = describe_image_with_openai(image_path, llm)
                
                # Create image document
                enhanced_description = f"Image Description for {os.path.basename(image_path)}: {description}. This image is from the document {pdf_file}."
                
                image_doc = Document(
                    page_content=enhanced_description,
                    metadata={
                        'source': pdf_file,
                        'type': 'image_description',
                        'image_path': image_path,
                        'image_name': os.path.basename(image_path)
                    }
                )
                all_new_chunks.append(image_doc)
                relevant_images += 1
            else:
                # Delete irrelevant image
                try:
                    os.remove(image_path)
                    print(f"  ✗ Removed irrelevant image")
                except:
                    pass
        
        print(f"  ✓ Kept {relevant_images} relevant image(s)")
    
    # Summary of new content
    print(f"\n{'='*70}")
    print("New Content Summary")
    print(f"{'='*70}")
    
    text_docs = [d for d in all_new_chunks if d.metadata.get('type') == 'text']
    image_docs = [d for d in all_new_chunks if d.metadata.get('type') == 'image_description']
    
    print(f"\nTotal new documents: {len(all_new_chunks)}")
    print(f"  - Text chunks: {len(text_docs)}")
    print(f"  - Image descriptions: {len(image_docs)}")
    
    if len(all_new_chunks) == 0:
        print("\n⚠ No content to add to index!")
        return
    
    # Load existing index
    print(f"\n{'='*70}")
    print("Merging into Existing Index")
    print(f"{'='*70}")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if os.path.exists(EXISTING_INDEX_PATH):
        print(f"\nLoading existing index from {EXISTING_INDEX_PATH}...")
        existing_index = FAISS.load_local(
            EXISTING_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Get count before merge
        old_count = existing_index.index.ntotal
        print(f"  Existing documents: {old_count}")
        
        # Create new index from new documents
        print(f"\nGenerating embeddings for {len(all_new_chunks)} new documents...")
        new_index = FAISS.from_documents(all_new_chunks, embeddings)
        
        # Merge indexes
        print(f"Merging indexes...")
        existing_index.merge_from(new_index)
        
        # Get count after merge
        new_count = existing_index.index.ntotal
        print(f"  ✓ Merged! Total documents: {new_count} (+{new_count - old_count})")
        
        # Save updated index
        print(f"\nSaving updated index to {EXISTING_INDEX_PATH}...")
        existing_index.save_local(EXISTING_INDEX_PATH)
        
    else:
        # No existing index - create new one
        print(f"\n⚠ No existing index found. Creating new index...")
        print(f"Generating embeddings...")
        vector_store = FAISS.from_documents(all_new_chunks, embeddings)
        
        print(f"Saving to {EXISTING_INDEX_PATH}...")
        vector_store.save_local(EXISTING_INDEX_PATH)
        print(f"  ✓ Created with {len(all_new_chunks)} documents")
    
    print(f"\n{'='*70}")
    print("✓ INCREMENTAL INGESTION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n✅ Successfully added content from {len(pdf_files)} PDF(s)")
    print(f"✅ Index updated: {EXISTING_INDEX_PATH}")
    print(f"\nYou can now use the updated index in your RAG system!")
    print()

if __name__ == "__main__":
    run_incremental_ingestion()