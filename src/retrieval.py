import os
import pickle
import time
import uuid
from typing import List, Any, Optional
from pydantic import Field, PrivateAttr

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# --- UPDATED IMPORTS ---
from langchain_classic.retrievers import EnsembleRetriever 
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
# -----------------------

from src.logging_utils import setup_logger, timer, log_timing, log_retrieval_metrics

# Point this to the new multimodal index directory
FAISS_INDEX_PATH = "faiss_index_multimodal"

# Initialize logger
logger = setup_logger('retrieval')

class EnhancedRetriever(BaseRetriever):
    """Enhanced retriever that inherits from BaseRetriever for LangChain compatibility"""
    
    # Pydantic configuration to allow complex objects like FAISS and ChatOpenAI
    model_config = {"arbitrary_types_allowed": True}
    
    # Public fields (can be passed in init)
    faiss_index_path: str = FAISS_INDEX_PATH
    
    # Private attributes (internal state, not exposed to Pydantic validation)
    _embeddings: Any = PrivateAttr()
    _llm: Any = PrivateAttr()
    _vector_store: Any = PrivateAttr()
    _semantic_retriever: Any = PrivateAttr()
    _bm25_retriever: Optional[Any] = PrivateAttr(default=None)
    _ensemble_retriever: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize components using the path defined in self.faiss_index_path
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Load FAISS vector store
        self._vector_store = FAISS.load_local(
            self.faiss_index_path, 
            self._embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Create semantic retriever with more candidates
        self._semantic_retriever = self._vector_store.as_retriever(
            search_kwargs={"k": 9}
        )
        
        # Try to create BM25 retriever for hybrid search
        try:
            self._setup_bm25_retriever()
            if self._bm25_retriever:
                self._ensemble_retriever = EnsembleRetriever(
                    retrievers=[self._semantic_retriever, self._bm25_retriever],
                    weights=[0.7, 0.3]
                )
            else:
                self._ensemble_retriever = self._semantic_retriever
        except Exception:
            self._ensemble_retriever = self._semantic_retriever
            self._bm25_retriever = None

    # method to define BM25 retriever
    def _setup_bm25_retriever(self):
        """
        Setup BM25 retriever for keyword search.

        FIX 3: The BM25 index is now persisted to disk as a pickle file
        (faiss_index_multimodal/bm25_cache.pkl). On subsequent server restarts
        the cache is loaded in ~50ms instead of rebuilding from scratch (~3.5s).

        Cache invalidation: delete bm25_cache.pkl manually whenever the FAISS
        index is rebuilt via ingestion.py.
        """
        try:
            cache_path = os.path.join(self.faiss_index_path, "bm25_cache.pkl")

            # --- Try loading from disk cache first ---
            if os.path.exists(cache_path):
                cache_start = time.perf_counter()
                with open(cache_path, "rb") as f:
                    self._bm25_retriever = pickle.load(f)
                log_timing(logger, "BM25_CACHE_LOAD", {
                    'duration_ms': round((time.perf_counter() - cache_start) * 1000, 2),
                    'source': 'disk'
                })
                return

            # --- Build BM25 index from scratch ---
            all_docs = []
            docstore = self._vector_store.docstore
            for doc_id in self._vector_store.index_to_docstore_id.values():
                doc = docstore.search(doc_id)
                all_docs.append(doc)

            build_start = time.perf_counter()
            self._bm25_retriever = BM25Retriever.from_documents(all_docs, k=6)
            log_timing(logger, "BM25_INDEX_BUILD", {
                'duration_ms': round((time.perf_counter() - build_start) * 1000, 2),
                'num_docs': len(all_docs)
            })

            # --- Persist to disk for fast future restarts ---
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self._bm25_retriever, f)
                logger.info(f"BM25 cache saved to {cache_path}")
            except Exception as cache_err:
                logger.warning(f"Could not save BM25 cache: {cache_err}")

        except Exception as e:
            print(f"BM25 setup failed, using semantic only: {e}")
            self._bm25_retriever = None
    
    def preprocess_query(self, question: str) -> str:
        """Enhance query with domain knowledge"""
        start_time = time.perf_counter()
        question_lower = question.lower()
        
        # Disease name expansions
        expansions = {
            'late blight': 'late blight phytophthora infestans',
            'early blight': 'early blight alternaria solani',
            'blackleg': 'blackleg soft rot pectobacterium bacterial',
            'ring rot': 'ring rot bacterial clavibacter',
            'scab': 'scab streptomyces bacterial',
            'dry rot': 'dry rot fusarium fungal',
            'soft rot': 'soft rot bacterial pectobacterium',
            'wilt': 'wilt verticillium bacterial fungal'
        }
        
        enhanced_question = question
        enhancements_applied = []
        
        for disease, expansion in expansions.items():
            if disease in question_lower:
                enhanced_question += f" {expansion}"
                enhancements_applied.append(disease)
        
        elapsed = time.perf_counter() - start_time
        if enhancements_applied:
            log_timing(logger, "QUERY_PREPROCESS", {
                'duration_ms': round(elapsed * 1000, 2),
                'enhancements': len(enhancements_applied),
                'applied_domains': ','.join(enhancements_applied[:3])
            })
        
        return enhanced_question
    
    def rerank_documents(self, docs: List[Document], question: str) -> List[Document]:
        """Rerank documents by relevance"""
        start_time = time.perf_counter()
        
        if not docs:
            return docs
        
        question_tokens = set(question.lower().split())
        scored_docs = []
        
        for doc in docs:
            content_lower = doc.page_content.lower()
            
            # Token overlap score
            content_tokens = set(content_lower.split())
            overlap = len(question_tokens.intersection(content_tokens))
            overlap_score = overlap / max(len(question_tokens), 1)
            
            # Keyword presence boost
            keyword_boost = 0
            important_keywords = ['disease', 'symptom', 'treatment', 'management', 'control', 'pathogen']
            for keyword in important_keywords:
                if keyword in content_lower:
                    keyword_boost += 0.05
            
            # Disease name exact match boost
            for token in question_tokens:
                if len(token) > 4 and token in content_lower:
                    keyword_boost += 0.1
            
            final_score = overlap_score + keyword_boost
            scored_docs.append((doc, final_score))
        
        # Sort by relevance and return top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        result_docs = [doc for doc, _ in scored_docs[:8]]  # Return top 8
        
        elapsed = time.perf_counter() - start_time
        log_timing(logger, "DOCUMENT_RERANKING", {
            'duration_ms': round(elapsed * 1000, 2),
            'docs_processed': len(docs),
            'docs_returned': len(result_docs)
        })
        
        return result_docs
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """
        Main retrieval method - required by BaseRetriever.
        Note: The argument name must be 'query' to match BaseRetriever signature.
        """
        total_start = time.perf_counter()
        retrieval_methods = []
        
        try:
            # Preprocess the query
            enhanced_question = self.preprocess_query(query)
            
            # Retrieve documents using hybrid search if available
            if self._bm25_retriever:
                sem_start = time.perf_counter()
                try:
                    docs = self._ensemble_retriever.invoke(enhanced_question)
                    retrieval_methods = ['ensemble']
                except Exception as e:
                    logger.warning(f"Ensemble retrieval failed, using semantic: {e}")
                    sem_start = time.perf_counter()
                    docs = self._semantic_retriever.invoke(enhanced_question)
                    retrieval_methods = ['semantic_fallback']
                
                sem_elapsed = time.perf_counter() - sem_start
                log_timing(logger, "ENSEMBLE_RETRIEVAL", {
                    'duration_ms': round(sem_elapsed * 1000, 2),
                    'docs_retrieved': len(docs)
                })
            else:
                sem_start = time.perf_counter()
                docs = self._semantic_retriever.invoke(enhanced_question)
                sem_elapsed = time.perf_counter() - sem_start
                retrieval_methods = ['semantic']
                
                log_timing(logger, "SEMANTIC_RETRIEVAL", {
                    'duration_ms': round(sem_elapsed * 1000, 2),
                    'docs_retrieved': len(docs)
                })
            
            # Also try original question if enhanced didn't work well
            if len(docs) < 5:
                orig_start = time.perf_counter()
                additional_docs = self._semantic_retriever.invoke(query)
                orig_elapsed = time.perf_counter() - orig_start
                
                log_timing(logger, "FALLBACK_RETRIEVAL", {
                    'duration_ms': round(orig_elapsed * 1000, 2),
                    'docs_retrieved': len(additional_docs),
                    'reason': 'insufficient_results'
                })
                
                # Combine and deduplicate
                all_docs = docs + additional_docs
                seen_content = set()
                unique_docs = []
                for doc in all_docs:
                    content_hash = hash(doc.page_content[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        unique_docs.append(doc)
                docs = unique_docs
                retrieval_methods.append('fallback')
            
            # Rerank for better relevance
            reranked_docs = self.rerank_documents(docs, query)
            
            # Log overall retrieval metrics
            total_elapsed = time.perf_counter() - total_start
            log_retrieval_metrics(logger, None, len(reranked_docs), total_elapsed, methods=retrieval_methods)
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Enhanced retrieval failed: {e}")
            fallback_start = time.perf_counter()
            result = self._semantic_retriever.invoke(query)
            fallback_elapsed = time.perf_counter() - fallback_start
            
            log_timing(logger, "COMPLETE_FALLBACK", {
                'duration_ms': round(fallback_elapsed * 1000, 2),
                'error': str(e)[:50]
            })
            
            return result

    # Note: Do not override 'invoke' manually. BaseRetriever handles it and calls _get_relevant_documents.

def load_retriever_from_disk():
    """Load the enhanced retriever with fallback to basic retriever"""
    try:
        # Pass path as a kwarg if needed, or rely on default
        enhanced_retriever = EnhancedRetriever(faiss_index_path=FAISS_INDEX_PATH)
        print("Enhanced retriever loaded successfully")
        return enhanced_retriever
    except Exception as e:
        print(f"Enhanced retriever failed, using basic retriever: {e}")
        # Fallback to basic retriever
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vector_store.as_retriever(search_kwargs={"k": 8})