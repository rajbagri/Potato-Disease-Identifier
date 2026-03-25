

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import List
import re
import time
from src.logging_utils import setup_logger, timer, log_timing

# Initialize logger
logger = setup_logger('query_processor')

class QueryProcessor:
    """Advanced query processing and enhancement for agricultural domain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Standalone question generation prompt
        self.standalone_prompt = PromptTemplate(
            template="""Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question that can be understood without the conversation history.
Focus on potato diseases, agricultural practices, and plant pathology.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""",
            input_variables=["chat_history", "question"]
        )
        
        # Query expansion prompt
        self.expansion_prompt = PromptTemplate(
            template="""You are an agricultural expert. Given a question about potato diseases or farming, 
generate 2-3 alternative phrasings or related questions that would help find comprehensive information.
Include technical terms, synonyms, and different perspectives.

Original Question: {question}

Alternative Questions:
1.""",
            input_variables=["question"]
        )
    
    def create_standalone_question(self, question: str, chat_history: List) -> str:
        """Convert follow-up question to standalone question"""
        start_time = time.perf_counter()
        
        try:
            if not chat_history or len(chat_history) == 0:
                elapsed = time.perf_counter() - start_time
                log_timing(logger, "STANDALONE_QUESTION", {
                    'duration_ms': round(elapsed * 1000, 2),
                    'status': 'skipped_no_history'
                })
                return question
            
            # Format chat history
            history_str = ""
            for i, (human, ai) in enumerate(chat_history[-3:]):  # Last 3 exchanges
                history_str += f"Human: {human}\nAssistant: {ai}\n"
            
            standalone_question = self.llm.invoke(
                self.standalone_prompt.format(
                    chat_history=history_str, 
                    question=question
                )
            ).content
            
            elapsed = time.perf_counter() - start_time
            log_timing(logger, "STANDALONE_QUESTION", {
                'duration_ms': round(elapsed * 1000, 2),
                'history_turns': len(chat_history),
                'status': 'generated'
            })
            
            return standalone_question.strip()
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Standalone question generation failed: {e}")
            log_timing(logger, "STANDALONE_QUESTION", {
                'duration_ms': round(elapsed * 1000, 2),
                'status': 'failed',
                'error': type(e).__name__
            })
            return question
    
    def expand_query(self, question: str) -> List[str]:
        """Generate alternative phrasings for better retrieval"""
        start_time = time.perf_counter()
        
        try:
            response = self.llm.invoke(self.expansion_prompt.format(question=question))
            
            queries = [question]  # Always include original
            lines = response.content.split('\n')
            
            expanded_count = 0
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    expanded_query = re.sub(r'^\d+\.\s*', '', line.strip())
                    if expanded_query and len(expanded_query) > 10:
                        queries.append(expanded_query)
                        expanded_count += 1
            
            result = queries[:3]  # Limit to 3 total queries
            
            elapsed = time.perf_counter() - start_time
            log_timing(logger, "QUERY_EXPANSION", {
                'duration_ms': round(elapsed * 1000, 2),
                'expanded_count': expanded_count,
                'total_queries': len(result)
            })
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Query expansion failed: {e}")
            log_timing(logger, "QUERY_EXPANSION", {
                'duration_ms': round(elapsed * 1000, 2),
                'status': 'failed'
            })
            return [question]
    
    def enhance_query_with_domain_knowledge(self, question: str) -> str:
        """Add domain-specific terms and context"""
        question_lower = question.lower()
        
        # Agricultural domain enhancements
        enhancements = {
            # Disease mappings
            'late blight': ['phytophthora infestans', 'oomycete', 'sporangia'],
            'early blight': ['alternaria solani', 'alternaria', 'fungal pathogen'],
            'blackleg': ['pectobacterium', 'soft rot bacteria', 'seed decay'],
            'ring rot': ['clavibacter sepedonicus', 'bacterial ring rot', 'vascular pathogen'],
            'scab': ['streptomyces scabies', 'actinomycete', 'tuber lesions'],
            'dry rot': ['fusarium', 'storage disease', 'wound pathogen'],
            'wilt': ['verticillium', 'vascular wilt', 'soil pathogen'],
            
            # Symptom mappings
            'symptoms': ['signs', 'disease symptoms', 'pathology'],
            'lesions': ['spots', 'necrosis', 'tissue damage'],
            'wilting': ['wilt', 'water stress', 'vascular blockage'],
            'rot': ['decay', 'decomposition', 'tissue breakdown'],
            
            # Management terms
            'control': ['management', 'prevention', 'treatment'],
            'fungicide': ['chemical control', 'pesticide', 'spray'],
            'resistance': ['tolerance', 'immunity', 'cultivar resistance'],
        }
        
        enhanced_terms = []
        for key_term, related_terms in enhancements.items():
            if key_term in question_lower:
                enhanced_terms.extend(related_terms)
        
        if enhanced_terms:
            return f"{question} {' '.join(enhanced_terms[:5])}"  # Add up to 5 related terms
        
        return question
    
    def preprocess_question(self, question: str, chat_history: List = None) -> dict:
        """Complete question preprocessing pipeline"""
        pipeline_start = time.perf_counter()
        
        # Step 1: Create standalone question if needed
        if chat_history:
            standalone_question = self.create_standalone_question(question, chat_history)
        else:
            standalone_question = question
        
        # Step 2: Enhance with domain knowledge
        enhanced_question = self.enhance_query_with_domain_knowledge(standalone_question)
        
        # Step 3: Generate alternative phrasings
        expanded_queries = self.expand_query(enhanced_question)
        
        pipeline_elapsed = time.perf_counter() - pipeline_start
        log_timing(logger, "FULL_PREPROCESSING", {
            'duration_ms': round(pipeline_elapsed * 1000, 2),
            'has_history': chat_history is not None,
            'expanded_queries': len(expanded_queries)
        })
        
        return {
            'original_question': question,
            'standalone_question': standalone_question,
            'enhanced_question': enhanced_question,
            'expanded_queries': expanded_queries,
            'primary_query': enhanced_question  # Use this for main retrieval
        }

class ContextFilter:
    """Filter and rank retrieved contexts for better relevance"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.relevance_prompt = PromptTemplate(
            template="""Rate the relevance of this document excerpt to the given question on a scale of 1-10.
Focus on potato diseases, agricultural practices, and plant pathology.

Question: {question}

Document Excerpt: {document}

Relevance Score (1-10): """,
            input_variables=["question", "document"]
        )
    
    def filter_contexts(self, documents: List, question: str, max_contexts: int = 6) -> List:
        """Filter and rank contexts by relevance"""
        if not documents:
            return documents
        
        scored_docs = []
        
        for doc in documents:
            try:
                # Get relevance score
                score_response = self.llm.invoke(
                    self.relevance_prompt.format(
                        question=question,
                        document=doc.page_content[:500]  # First 500 chars
                    )
                )
                
                # Extract numeric score
                score_text = score_response.content.strip()
                score_match = re.search(r'\d+', score_text)
                
                if score_match:
                    score = int(score_match.group())
                else:
                    score = 5  # Default score
                
                scored_docs.append((doc, score))
                
            except Exception as e:
                print(f"Scoring failed for document: {e}")
                scored_docs.append((doc, 5))  # Default score
        
        # Sort by score and return top contexts
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:max_contexts]]
    
    def combine_contexts_intelligently(self, documents: List, question: str) -> str:
        """Combine contexts in an intelligent way"""
        if not documents:
            return "No relevant information found."
        
        # Group by source for better organization
        source_groups = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)
        
        combined_context = ""
        for source, docs in source_groups.items():
            combined_context += f"\n--- From {source} ---\n"
            for i, doc in enumerate(docs):
                combined_context += f"{doc.page_content}\n"
                if i < len(docs) - 1:
                    combined_context += "\n"
        
        return combined_context

