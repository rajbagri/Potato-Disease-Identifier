# --- Python Path Setup ---
import sys
import os
# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

import json
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.retrieval import load_retriever_from_disk
from src.generation import create_conversational_chain
from langchain_openai import ChatOpenAI

# --- Configuration ---
EVAL_DATASET_FILE = "evaluation_dataset_v2.json"
EVAL_RESULTS_FILE = "evaluation_results.csv"
PIPELINE_CHECKPOINT_FILE = "pipeline_results_checkpoint.json"
JUDGE_MODEL = "gpt-4o"
NUM_TEST_QUESTIONS = 2  # <-- Set the number of questions to test (0 for all)
FILTER_BY_SOURCE = None  # Set to "document_name.pdf" to filter by source
SAVE_INTERMEDIATE_RESULTS = True
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N questions

def load_eval_dataset(filepath: str) -> list:
    """Loads the evaluation dataset from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Evaluation dataset not found at {filepath}. Run generate_eval_dataset.py first.")
    
    print(f"Loading evaluation dataset from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} Q&A pairs.")
    return dataset

def run_rag_pipeline(qa_chain, eval_dataset: list, checkpoint_file: str = PIPELINE_CHECKPOINT_FILE) -> list:
    """Runs the full ConversationalRetrievalChain for each question with checkpoint support."""
    results = []
    start_idx = 0
    
    # Always clear checkpoint at the start to ensure fresh evaluation
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"Running RAG pipeline for {len(eval_dataset)} evaluation questions...")
    
    for i in tqdm(range(start_idx, len(eval_dataset)), total=len(eval_dataset), initial=start_idx, desc="Processing RAG queries"):
        item = eval_dataset[i]
        question = item['question']
        ground_truth = item['ground_truth']
        
        try:
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
            def invoke_chain():
                chat_history = [] 
                return qa_chain.invoke({
                    "question": question,
                    "chat_history": chat_history
                })
            
            result = invoke_chain()
            answer = result["answer"]
            retrieved_docs = result.get("source_documents", [])
            contexts_list = [doc.page_content for doc in retrieved_docs]
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts_list,
                "num_contexts": len(contexts_list)
            })
        except Exception as e:
            print(f"\n⚠️  FAILED on question {i+1}: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": f"Error: {str(e)[:100]}",
                "contexts": [],
                "num_contexts": 0
            })
        
        # Save checkpoint at intervals
        if SAVE_INTERMEDIATE_RESULTS and (i + 1 - start_idx) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(results, f)
    
    print("✓ Pipeline run complete.")
    return results

def main():
    start_time = time.time()
    
    try:
        # 1. Load the evaluation dataset
        eval_dataset = load_eval_dataset(EVAL_DATASET_FILE)
        
        # Filter by source if specified
        if FILTER_BY_SOURCE:
            eval_dataset = [q for q in eval_dataset if q.get('source_document') == FILTER_BY_SOURCE]
            print(f"📄 Filtered to {len(eval_dataset)} questions from {FILTER_BY_SOURCE}")
        
        # Limit the dataset to the first N questions for testing
        if NUM_TEST_QUESTIONS > 0:
            eval_dataset = eval_dataset[:NUM_TEST_QUESTIONS]
            print(f"⚙️  RUNNING TEST: Limiting evaluation to first {NUM_TEST_QUESTIONS} questions.")

        # 2. Load the RAG pipeline
        print("\n📂 Loading RAG pipeline...")
        retriever = load_retriever_from_disk()
        qa_chain = create_conversational_chain(retriever)
        print("✓ RAG pipeline loaded.")
        
        # 3. Run the pipeline and collect results
        pipeline_results = run_rag_pipeline(qa_chain, eval_dataset)
        
        # Check for questions with no retrieved contexts
        questions_no_context = [r for r in pipeline_results if r['num_contexts'] == 0]
        if questions_no_context:
            print(f"\n⚠️  WARNING: {len(questions_no_context)} questions had NO retrieved contexts!")
            print("   These may affect evaluation metrics.")
        
        # 4. Convert results to a Hugging Face Dataset
        results_dataset = Dataset.from_list(pipeline_results)
        
        # Define the LLM to be used as the "Judge"
        judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0.0)
        
        # 5. Run the RAGas evaluation
        print(f"\n🔍 Starting RAGas evaluation for {len(pipeline_results)} items...")
        
        result = evaluate(
            dataset=results_dataset,
            llm=judge_llm, 
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
        )
        
        print("✓ RAGas evaluation complete.")
        
        # 6. Save and display results
        df = result.to_pandas()
        
        # Add original questions and source documents back to dataframe
        questions = [r['question'] for r in pipeline_results]
        df['question'] = questions
        
        if pipeline_results and 'source_document' in pipeline_results[0]:
            df['source_document'] = [pipeline_results[i].get('source_document', 'Unknown') for i in range(len(df))]
        
        # Save with retry in case file is locked
        max_retries = 3
        for attempt in range(max_retries):
            try:
                df.to_csv(EVAL_RESULTS_FILE, index=False)
                print(f"✓ Results saved successfully to {EVAL_RESULTS_FILE}")
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    print(f"⚠️  File is locked, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                else:
                    print(f"❌ Could not save {EVAL_RESULTS_FILE} - file may be open in another application")
                    print("   Please close it and try again")
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS SUMMARY")
        print("="*70)
        display_cols = ['question', 'faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
        if all(col in df.columns for col in display_cols):
            print(df[display_cols].to_string(index=False))
        else:
            print(df.to_string())
        
        # Print the average scores
        print("\n" + "="*70)
        print("AVERAGE SCORES")
        print("="*70)
        avg_scores = {
            'Faithfulness': df['faithfulness'].mean(),
            'Answer Relevancy': df['answer_relevancy'].mean(),
            'Context Precision': df['context_precision'].mean(),
            'Context Recall': df['context_recall'].mean(),
        }
        
        for metric, score in avg_scores.items():
            status = "✓" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
            print(f"{status} {metric:.<40} {score:.4f}")
        
        # Detailed statistics
        print("\n" + "="*70)
        print("DETAILED STATISTICS")
        print("="*70)
        for col in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            print(f"\n{col.upper().replace('_', ' ')}:")
            print(f"  Mean:   {df[col].mean():.4f}")
            print(f"  Median: {df[col].median():.4f}")
            print(f"  Std:    {df[col].std():.4f}")
            print(f"  Min:    {df[col].min():.4f}")
            print(f"  Max:    {df[col].max():.4f}")
        
        # Identify worst performing questions
        print("\n" + "="*70)
        print("BOTTOM 3 PERFORMING QUESTIONS")
        print("="*70)
        worst_idx = df['faithfulness'].nsmallest(3).index
        for rank, idx in enumerate(worst_idx, 1):
            q = df.loc[idx, 'question'] if 'question' in df.columns else "N/A"
            q_display = q[:70] + "..." if len(str(q)) > 70 else q
            print(f"\n{rank}. Question: {q_display}")
            print(f"   Faithfulness: {df.loc[idx, 'faithfulness']:.4f}")
            print(f"   Answer Relevancy: {df.loc[idx, 'answer_relevancy']:.4f}")
            print(f"   Context Precision: {df.loc[idx, 'context_precision']:.4f}")
        
        # Timing and cost estimation
        elapsed = time.time() - start_time
        estimated_cost = len(eval_dataset) * 0.003  # Rough estimate for evaluation
        
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print(f"📊 Questions evaluated: {len(eval_dataset)}")
        print(f"💰 Estimated OpenAI cost: ${estimated_cost:.2f}")
        print(f"💾 Results saved to: {EVAL_RESULTS_FILE}")
        print(f"🔖 Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clean up checkpoint after successful completion
        if SAVE_INTERMEDIATE_RESULTS and os.path.exists(PIPELINE_CHECKPOINT_FILE):
            os.remove(PIPELINE_CHECKPOINT_FILE)
            print(f"🗑️  Cleaned up checkpoint file")
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        print(f"⏱️  Elapsed time: {(time.time() - start_time)/60:.1f} minutes")
        raise

if __name__ == "__main__":
    main()