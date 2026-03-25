import os
import json
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Configuration ---
PDF_DIRECTORY = "data"
OUTPUT_FILE = "evaluation_dataset_v2.json"
QUESTIONS_PER_DOC = 10
MAX_DOCS_TO_PROCESS = 0
MIN_DOC_LENGTH_CHARS = 500

def load_full_text_docs(pdf_directory: str) -> list[tuple[str, str]]:
    """Loads all PDFs and returns a list of (filename, full_text) tuples."""
    print(f"Loading documents from {pdf_directory}...")
    documents = []
    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist.")
        return []

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                doc = fitz.open(pdf_path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text() + "\n"
                doc.close()
                
                if len(full_text) < MIN_DOC_LENGTH_CHARS:
                    print(f"  - SKIPPED {filename} (too short: {len(full_text)} chars)")
                    continue
                    
                documents.append((filename, full_text))
                print(f"  - Loaded {filename} ({len(full_text)} chars)")
            except Exception as e:
                print(f"  - FAILED to load {filename}: {e}")
    
    print(f"Loaded {len(documents)} valid documents.")
    return documents

def get_qa_generation_chain():
    """Returns a LangChain chain for generating a *list* of Q&A pairs."""
    
    prompt_template = """
    You are an expert at creating evaluation datasets for a RAG system about potato diseases.
    Based ONLY on the full document text provided below, generate a list of {num_questions} high-quality, complex question-and-answer pairs.

    Guidelines:
    1.  The questions must be answerable using ONLY the provided document.
    2.  The answers must be factual, concise, and directly extracted or synthesized from the text.
    3.  Generate questions that cover a range of topics (e.g., symptoms, control, causes) from the document.
    4.  Return your response as a single JSON object with ONE key: "qa_pairs".
    5.  The value of "qa_pairs" must be a list of JSON objects, where each object has two keys: "question" and "ground_truth".
    
    EXAMPLE FORMAT:
    {{
      "qa_pairs": [
        {{
          "question": "What is the primary cause of Late Blight?",
          "ground_truth": "Late Blight is caused by the oomycete Phytophthora infestans, which thrives in cool, moist conditions."
        }},
        {{
          "question": "What are the visible symptoms of Late Blight on potato leaves?",
          "ground_truth": "Early symptoms appear as water-soaked spots on leaves, typically starting at the leaf margins. As the disease progresses, these spots become brown and necrotic, with a white, cottony fungal growth visible on the underside of leaves during humid conditions."
        }},
        {{
          "question": "How can Late Blight be managed in potato cultivation?",
          "ground_truth": "Late Blight can be managed through multiple strategies: removing infected plant material promptly, ensuring proper crop rotation (minimum 3-year gap), maintaining good air circulation through appropriate plant spacing, using resistant potato varieties, and applying fungicide sprays when environmental conditions favor disease development, particularly during periods of high humidity and moderate temperatures."
        }},
        {{
          "question": "Under what weather conditions does Late Blight spread most rapidly?",
          "ground_truth": "Late Blight spreads most rapidly in cool, moist conditions, typically when temperatures range between 12-25°C with high humidity (>90%) and frequent leaf wetness from rain or dew. The pathogen can complete its life cycle in as little as 3-5 days under optimal conditions."
        }},
        {{
          "question": "What is the economic impact of Late Blight on potato production?",
          "ground_truth": "Late Blight is one of the most economically significant potato diseases worldwide, causing severe yield losses that can reach 50-100% of the crop in severe outbreaks. The disease also requires substantial investment in fungicide applications and resistant variety development to manage effectively."
        }}
      ]
    }}

    DOCUMENT TEXT:
    ---
    {context}
    ---
    
    JSON:
    """
    
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
        max_tokens=4096 
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    parser = StrOutputParser()
    
    return prompt | llm | parser

def generate_dataset():
    """Main function to generate and save the dataset."""
    documents = load_full_text_docs(PDF_DIRECTORY)
    if not documents:
        print("No documents found. Exiting.")
        return

    qa_chain = get_qa_generation_chain()
    
    # Sort documents by length (descending) to identify top 7 longest docs
    documents_sorted_by_length = sorted(documents, key=lambda x: len(x[1]), reverse=True)
    
    # Identify top 7 documents
    top_7_docs = set(doc[0] for doc in documents_sorted_by_length[:7])
    
    dataset = []
    
    docs_to_process = documents
    if MAX_DOCS_TO_PROCESS > 0:
        print(f"\n--- WARNING: Limiting generation to {MAX_DOCS_TO_PROCESS} documents for this run. ---")
        docs_to_process = documents[:MAX_DOCS_TO_PROCESS]

    print(f"\nStarting Q&A pair generation for {len(docs_to_process)} documents...")
    print(f"Top 7 longest documents will get {QUESTIONS_PER_DOC} Q&A pairs each.")
    print(f"Remaining documents will get {5} Q&A pairs each.")
    
    for i, (filename, doc_text) in enumerate(docs_to_process):
        # Determine number of questions based on whether doc is in top 7
        num_questions = QUESTIONS_PER_DOC if filename in top_7_docs else 4
        
        print(f"\nProcessing document {i+1}/{len(docs_to_process)}: {filename} ({len(doc_text)} chars) - Generating {num_questions} Q&A pairs...")
        try:
            qa_list_str = qa_chain.invoke({
                "context": doc_text,
                "num_questions": num_questions
            })
            
            # --- THIS IS THE FIX ---
            # Find the first '{' and the last '}' to extract the JSON object
            start_index = qa_list_str.find('{')
            end_index = qa_list_str.rfind('}')
            
            if start_index == -1 or end_index == -1:
                raise json.JSONDecodeError("Could not find JSON object in LLM output.", qa_list_str, 0)
                
            clean_json_str = qa_list_str[start_index : end_index + 1]
            response_json = json.loads(clean_json_str)
            # --- END OF FIX ---
            
            if "qa_pairs" in response_json and isinstance(response_json["qa_pairs"], list):
                for pair in response_json["qa_pairs"]:
                    pair["source_document"] = filename
                
                dataset.extend(response_json["qa_pairs"])
                print(f"  -> Successfully generated {len(response_json['qa_pairs'])} pairs from {filename}.")
            else:
                print(f"  -> SKIPPED: Invalid JSON structure from LLM for {filename} (missing 'qa_pairs' list).")

        except json.JSONDecodeError as e:
            print(f"  -> SKIPPED: Failed to decode JSON from LLM output for {filename}.")
            print(f"  -> Error: {e}")
            print("  -> RAW LLM OUTPUT RECEIVED:", qa_list_str) # Keep this for debugging
        except Exception as e:
            print(f"  -> SKIPPED: An error occurred for {filename}: {e}")
            
    print(f"\nSuccessfully generated a total of {len(dataset)} Q&A pairs.")
    
    if dataset:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Evaluation dataset saved to {OUTPUT_FILE}")
    else:
        print("No Q&A pairs were generated, file not saved.")

if __name__ == "__main__":
    generate_dataset()