import os
import concurrent.futures
from dotenv import load_dotenv
import time

# --- UPDATED IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# -----------------------

from typing import List, Tuple, Dict, Generator
from src.logging_utils import setup_logger, timer, log_timing, log_generation_metrics

load_dotenv()

# Initialize logger
logger = setup_logger('generation')

class ImprovedConversationalChain:
    """
    Custom conversational RAG chain built without ConversationalRetrievalChain.

    Key improvements over the previous design:
    - FIX 1: Eliminates the double CONDENSE_QUESTION_LLM call that occurred because
      ConversationalRetrievalChain reused the streaming LLM for both condensing and
      generation, causing two sequential LLM invocations per query.
    - FIX 2: Condense-question and retrieval-on-original-question run in parallel via
      ThreadPoolExecutor, so retrieval is never blocked waiting for the condense LLM.
    - FIX 4: Main LLM is capped at max_tokens=600 to prevent unbounded response growth.
    - FIX 5: Dedicated non-streaming LLMs for condensing and memory summarisation so
      the streaming LLM is never invoked in contexts that do not require streaming.
    """

    def __init__(self, retriever):
        self.retriever = retriever

        # FIX 4 & FIX 2: cap output tokens and add connection resilience
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1, 
            streaming=True, 
            max_tokens=600,
            timeout=60.0,      # Give OpenAI 60 seconds to respond
            max_retries=3      # Automatically retry on dropped connections
        )

        # FIX 1 & 5: Dedicated non-streaming LLM for condensing follow-up questions.
        # This completely prevents the double-call bug — the streaming LLM is only
        # ever used for final answer generation.
        self.condense_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

        # FIX 5: Dedicated non-streaming LLM for memory summarisation.
        # Previously ConversationSummaryBufferMemory shared self.llm (streaming=True),
        # causing unpredictable latency spikes when the buffer exceeded 1000 tokens.
        self.summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False)

        # Memory uses summary_llm exclusively
        self.memory = ConversationSummaryBufferMemory(
            llm=self.summary_llm,
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            max_token_limit=1000
        )

        # Condense prompt — used only when chat history exists
        self.condense_prompt = PromptTemplate(
            template="""Given the conversation history and a follow-up question, rephrase the follow-up as a concise standalone question.
Focus on potato diseases, agricultural practices, and plant pathology.
If the follow-up is already self-contained or is a simple greeting, return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""",
            input_variables=["chat_history", "question"]
        )

        # QA prompt — added "Be concise" instruction to limit token growth
        self.qa_prompt = PromptTemplate(
            template="""You are Aloo Sahayak, an expert agricultural AI assistant specializing in potato diseases, nutrition, and cultivation practices.

INSTRUCTIONS:
1. If the user sends a simple greeting (hi, hello, hey, thanks, bye), respond politely and briefly.
2. For questions about potatoes (diseases, nutrition, fertilizers, cultivation), use the provided context to give accurate, detailed answers.
3. If the context doesn't contain sufficient information to answer the question, say "I don't have enough information about this in my knowledge base."
4. NEVER return generic greetings as answers to factual questions - always attempt to use the context first.
5. Be professional, helpful, and cite sources when available.
6. Be concise — limit your answer to 3-4 focused paragraphs.

Conversation History:
{chat_history}

Context from Knowledge Base:
{context}

User Question: {question}

Provide a comprehensive but concise answer based on the context. For factual questions, always prioritize using the retrieved context over general knowledge.

Answer:""",
            input_variables=["context", "question", "chat_history"]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_history_str(self, chat_history: List[Tuple[str, str]]) -> str:
        """Format last 3 chat turns into a plain string for prompt injection."""
        history_str = ""
        for human_msg, ai_msg in chat_history[-3:]:
            history_str += f"Human: {human_msg}\nAssistant: {ai_msg}\n"
        return history_str.strip()

    def _condense_question(self, question: str, chat_history: List[Tuple[str, str]]) -> str:
        """
        Rephrase a follow-up question into a standalone question using the dedicated
        non-streaming condense_llm. Called in a background thread by
        _parallel_condense_and_retrieve so it runs concurrently with retrieval.
        """
        if not chat_history:
            return question

        condense_start = time.perf_counter()
        logger.info("TIMING | CONDENSE_QUESTION_LLM             | Started (rephrasing follow-up as standalone question)")

        try:
            history_str = self._format_history_str(chat_history)
            prompt_str = self.condense_prompt.format(
                chat_history=history_str,
                question=question
            )
            response = self.condense_llm.invoke(prompt_str)
            condensed = response.content.strip()

            log_timing(logger, "CONDENSE_QUESTION_LLM", {
                'duration_ms': round((time.perf_counter() - condense_start) * 1000, 2)
            })
            return condensed

        except Exception as e:
            log_timing(logger, "CONDENSE_QUESTION_LLM", {
                'duration_ms': round((time.perf_counter() - condense_start) * 1000, 2),
                'status': 'failed_fallback'
            })
            logger.warning(f"Condense question failed ({e}), using original question")
            return question

    def _retrieve(self, question: str) -> List[Document]:
        """Run retrieval for a given question string."""
        return self.retriever.invoke(question)

    def _parallel_condense_and_retrieve(
        self, question: str, chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Document]]:
        """
        FIX 2: Runs condense-question and retrieval concurrently using two threads.

        Strategy:
          Thread A — condense follow-up → standalone question  (LLM call)
          Thread B — retrieve using the original question immediately  (FAISS + BM25)

        Once both threads finish:
          - If the condensed question differs from the original, a second fast retrieval
            fires on the condensed question and results are merged/deduplicated.
          - This ensures retrieval is NEVER serialised behind the condense LLM call.
        """
        if not chat_history:
            # No history: skip condense, retrieve directly
            docs = self._retrieve(question)
            return question, docs

        parallel_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_condense = executor.submit(self._condense_question, question, chat_history)
            future_retrieve = executor.submit(self._retrieve, question)

            condensed_question = future_condense.result()
            original_docs = future_retrieve.result()

        log_timing(logger, "PARALLEL_CONDENSE_RETRIEVE", {
            'duration_ms': round((time.perf_counter() - parallel_start) * 1000, 2),
            'original_docs': len(original_docs)
        })

        # If condensed question is meaningfully different, retrieve again and merge
        merged_docs = original_docs
        if condensed_question.strip().lower() != question.strip().lower():
            try:
                merge_start = time.perf_counter()
                condensed_docs = self._retrieve(condensed_question)

                seen = {hash(doc.page_content[:100]) for doc in original_docs}
                extra_docs = [d for d in condensed_docs if hash(d.page_content[:100]) not in seen]
                merged_docs = original_docs + extra_docs

                log_timing(logger, "CONDENSED_RETRIEVAL_MERGE", {
                    'duration_ms': round((time.perf_counter() - merge_start) * 1000, 2),
                    'extra_docs': len(extra_docs),
                    'total_docs': len(merged_docs)
                })
            except Exception as e:
                logger.warning(f"Condensed retrieval merge failed: {e}")

        return condensed_question, merged_docs

    def _build_context(self, docs: List[Document]) -> str:
        """Build a context string from retrieved documents, grouped by source."""
        if not docs:
            return "No relevant information found."
        source_groups: Dict[str, List[Document]] = {}
        for doc in docs:
            src = doc.metadata.get('source', 'Unknown')
            source_groups.setdefault(src, []).append(doc)
        context = ""
        for src, group_docs in source_groups.items():
            context += f"\n--- From {src} ---\n"
            for doc in group_docs:
                context += doc.page_content + "\n"
        return context.strip()

    def _sync_memory_with_external_history(self, external_history: List[Tuple[str, str]]):
        """Sync external chat history with internal ConversationSummaryBufferMemory."""
        self.memory.clear()
        for human_msg, ai_msg in external_history[-5:]:
            self.memory.chat_memory.add_user_message(human_msg)
            self.memory.chat_memory.add_ai_message(ai_msg)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, inputs: dict) -> dict:
        """Non-streaming invocation — returns full answer dict."""
        invoke_start = time.perf_counter()
        question = inputs.get("question", "")
        external_chat_history = inputs.get("chat_history", [])

        try:
            if external_chat_history:
                sync_start = time.perf_counter()
                self._sync_memory_with_external_history(external_chat_history)
                log_timing(logger, "MEMORY_SYNC", {
                    'duration_ms': round((time.perf_counter() - sync_start) * 1000, 2),
                    'history_items': len(external_chat_history)
                })
            else:
                logger.info("TIMING | MEMORY_CLEAR                   | Cleared stale chain memory for new conversation")
                self.memory.clear()

            # FIX 2: parallel condense + retrieve
            condensed_question, docs = self._parallel_condense_and_retrieve(question, external_chat_history)

            log_timing(logger, "SOURCE_DOCS_CAPTURED", {'count': len(docs)})

            context = self._build_context(docs)
            history_str = self._format_history_str(external_chat_history)
            prompt_str = self.qa_prompt.format(
                context=context,
                question=condensed_question,
                chat_history=history_str
            )

            chain_start = time.perf_counter()
            response = self.llm.invoke(prompt_str)
            answer = response.content

            log_timing(logger, "CHAIN_INVOCATION", {
                'duration_ms': round((time.perf_counter() - chain_start) * 1000, 2),
                'answer_length': len(answer),
                'source_docs': len(docs)
            })

            # Update memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(answer)

            total_elapsed = time.perf_counter() - invoke_start
            log_generation_metrics(logger, None, total_elapsed)

            return {
                "answer": answer,
                "source_documents": docs,
                "generated_question": condensed_question
            }

        except Exception as e:
            elapsed = time.perf_counter() - invoke_start
            logger.error(f"Error in chain invocation: {e}")
            log_timing(logger, "INVOKE_ERROR", {
                'duration_ms': round(elapsed * 1000, 2),
                'error': type(e).__name__
            })
            return {
                "answer": "I apologize, but I encountered an error processing your question. Please try again.",
                "source_documents": [],
                "generated_question": question
            }

    def stream(self, inputs: dict) -> Generator:
        """
        Streaming invocation — yields chunk dicts followed by a final complete dict.

        Flow:
          1. Parallel condense + retrieve (FIX 2)
          2. Build prompt from merged docs + condensed question
          3. Stream tokens directly from self.llm.stream() — single LLM call (FIX 1)
          4. Yield {'type': 'complete', ...} with source docs at the end
        """
        stream_start = time.perf_counter()
        chunk_count = 0
        question = inputs.get("question", "")
        external_chat_history = inputs.get("chat_history", [])

        try:
            if external_chat_history:
                sync_start = time.perf_counter()
                self._sync_memory_with_external_history(external_chat_history)
                log_timing(logger, "MEMORY_SYNC_STREAM", {
                    'duration_ms': round((time.perf_counter() - sync_start) * 1000, 2),
                    'history_items': len(external_chat_history)
                })
            else:
                logger.info("TIMING | MEMORY_CLEAR                   | Cleared stale chain memory for new conversation")
                self.memory.clear()

            # FIX 2: parallel condense + retrieve
            condensed_question, docs = self._parallel_condense_and_retrieve(question, external_chat_history)

            log_timing(logger, "SOURCE_DOCS_CAPTURED", {'count': len(docs)})

            context = self._build_context(docs)
            history_str = self._format_history_str(external_chat_history)
            prompt_str = self.qa_prompt.format(
                context=context,
                question=condensed_question,
                chat_history=history_str
            )

            full_answer = ""
            chunk_start = time.perf_counter()

            # FIX 1: stream directly from self.llm — single LLM call, no double-condense
            for chunk in self.llm.stream(prompt_str):
                chunk_count += 1
                text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_answer += text
                yield {'type': 'chunk', 'content': text}

            chunk_elapsed = time.perf_counter() - chunk_start
            log_timing(logger, "STREAMING_CHUNKS", {
                'duration_ms': round(chunk_elapsed * 1000, 2),
                'chunk_count': chunk_count,
                'answer_length': len(full_answer)
            })

            total_stream_elapsed = time.perf_counter() - stream_start
            log_generation_metrics(logger, None, total_stream_elapsed,
                                   chunk_count=chunk_count, token_count=len(full_answer.split()))

            # Update memory after streaming completes
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(full_answer)

            log_timing(logger, "SOURCE_DOCS_FINAL", {
                'captured_count': len(docs),
                'result_count': len(docs),
                'final_count': len(docs)
            })

            yield {
                'type': 'complete',
                'answer': full_answer,
                'source_documents': docs
            }

        except Exception as e:
            elapsed = time.perf_counter() - stream_start
            logger.error(f"Error in chain streaming: {e}")
            log_timing(logger, "STREAM_ERROR", {
                'duration_ms': round(elapsed * 1000, 2),
                'chunks_sent': chunk_count,
                'error': type(e).__name__
            })
            yield {
                'type': 'error',
                'content': "I apologize, but I encountered an error processing your question. Please try again."
            }


def create_conversational_chain(retriever):
    """
    Creates an improved conversational RAG chain with better memory management.
    """
    print("Initializing improved conversational RAG chain...")

    chain = ImprovedConversationalChain(retriever)

    print("Improved conversational RAG chain created successfully.")
    return chain