# Sample Query Timing Log Output

This file shows what your query_timing.log will look like when the system is running.

```
2025-02-22 10:30:45,123 | INFO | main | ==================================================================================================================
2025-02-22 10:30:45,124 | INFO | main | QUERY_START | ID: a7c2e1b3 | Question: What are the symptoms of late blight in potatoes?...
2025-02-22 10:30:45,124 | INFO | main | ==================================================================================================================
2025-02-22 10:30:45,150 | INFO | query_processor | TIMING | STANDALONE_QUESTION | Duration: 26.45ms | history_turns=2 | status=generated
2025-02-22 10:30:45,200 | INFO | query_processor | TIMING | QUERY_EXPANSION | Duration: 55.32ms | expanded_count=2 | total_queries=3
2025-02-22 10:30:46,800 | INFO | retrieval | TIMING | QUERY_PREPROCESS | Duration: 1.23ms | enhancements=1 | applied_domains=late blight
2025-02-22 10:30:47,150 | INFO | retrieval | TIMING | SEMANTIC_RETRIEVAL | Duration: 348.56ms | docs_retrieved=9
2025-02-22 10:30:47,200 | INFO | retrieval | TIMING | DOCUMENT_RERANKING | Duration: 48.32ms | docs_processed=9 | docs_returned=8
2025-02-22 10:30:47,210 | INFO | retrieval | TIMING | RETRIEVAL | num_docs=8 | retrieval_ms=398.45 | methods=semantic
2025-02-22 10:30:47,220 | INFO | generation | TIMING | MEMORY_SYNC | Duration: 12.45ms | history_items=2
2025-02-22 10:30:47,230 | INFO | api | TIMING | AI_CHAIN_INVOCATION | Duration: 2345.67ms
2025-02-22 10:30:49,575 | INFO | generation | TIMING | STREAMING_CHUNKS | Duration: 2345.12ms | chunk_count=145 | answer_length=2847
2025-02-22 10:30:49,576 | INFO | generation | TIMING | GENERATION | duration_ms=2456.78 | chunks=145 | tokens=487
2025-02-22 10:30:49,590 | INFO | api | TIMING | DB_ADD_AI_WS | Duration: 14.23ms | response_length=2847
2025-02-22 10:30:49,590 | INFO | main | ==================================================================================================================
2025-02-22 10:30:49,591 | INFO | main | QUERY_COMPLETE | ID: a7c2e1b3 | Status: SUCCESS
2025-02-22 10:30:49,591 | INFO | main | TOTAL_TIME | 4466.78ms
2025-02-22 10:30:49,591 | INFO | main | TIMING_BREAKDOWN:
2025-02-22 10:30:49,591 | INFO | main |   AI_CHAIN_INVOCATION             | 2345.67ms
2025-02-22 10:30:49,591 | INFO | main |   STREAMING_CHUNKS               | 2345.12ms
2025-02-22 10:30:49,591 | INFO | main |   SEMANTIC_RETRIEVAL             | 348.56ms
2025-02-22 10:30:49,591 | INFO | main |   QUERY_EXPANSION                | 55.32ms
2025-02-22 10:30:49,591 | INFO | main |   DOCUMENT_RERANKING             | 48.32ms
2025-02-22 10:30:49,591 | INFO | main |   STANDALONE_QUESTION            | 26.45ms
2025-02-22 10:30:49,591 | INFO | main | ==================================================================================================================
```

## Performance Breakdown Example

From the above logs, we can see:

| Component | Time | Percentage | Notes |
|-----------|------|-----------|-------|
| AI Chain Invocation | 2345.67ms | 52.5% | LLM processing |
| Streaming Chunks | 2345.12ms | 52.5% | Output streaming |
| Semantic Retrieval | 348.56ms | 7.8% | Document search |
| Query Expansion | 55.32ms | 1.2% | Alternative queries |
| Document Reranking | 48.32ms | 1.1% | Relevance sorting |
| Standalone Question | 26.45ms | 0.6% | Chat context processing |
| **Total** | **4466.78ms** | **100%** | ~4.5 seconds |

## Interpretation

- **Total Latency**: 4.47 seconds - reasonable for a complex RAG query
- **Bottleneck**: LLM generation accounts for ~52% of time
- **Retrieval**: ~7.8% of time (efficient)
- **Preprocessing**: ~2% of time (minimal overhead)

## Multiple Query Example

When processing multiple queries, logs will show pattern like:

```
...First Query logs...
==================================================================================================================

...Second Query logs...
==================================================================================================================

...Third Query logs...
==================================================================================================================
```

Each query is clearly separated with its own ID and timing breakdown.

## Error Case Example

```
2025-02-22 10:35:12,234 | INFO | main | ==================================================================================================================
2025-02-22 10:35:12,235 | INFO | main | QUERY_START | ID: f4d9a2c6 | Question: Invalid question query...
2025-02-22 10:35:12,235 | INFO | main | ==================================================================================================================
2025-02-22 10:35:12,450 | ERROR | retrieval | Retrieval service unavailable
2025-02-22 10:35:12,451 | INFO | main | ==================================================================================================================
2025-02-22 10:35:12,451 | INFO | main | QUERY_COMPLETE | ID: f4d9a2c6 | Status: FAILED: ConnectionError
2025-02-22 10:35:12,451 | INFO | main | TOTAL_TIME | 216.78ms
2025-02-22 10:35:12,451 | INFO | main | ==================================================================================================================
```

## Streaming Performance Example

For WebSocket streaming requests:

```
2025-02-22 10:40:15,100 | INFO | api | TIMING | DB_ADD_USER_WS | Duration: 8.45ms
2025-02-22 10:40:15,120 | INFO | api | TIMING | HISTORY_RETRIEVAL_WS | Duration: 5.23ms | history_items=3
2025-02-22 10:40:15,300 | INFO | api | TIMING | STREAMING_COMPLETE | Duration: 2450.67ms | chunks=156 | response_length=3100 | first_chunk_ms=245.23
2025-02-22 10:40:15,320 | INFO | api | TIMING | DB_ADD_AI_WS | Duration: 19.87ms | response_length=3100
```

This shows:
- **First chunk time**: 245ms (time to first token)
- **Total streaming time**: 2.45 seconds
- **Chunks sent**: 156 pieces
- **DB operations**: Minimal overhead

---

Use this log structure to:
1. Identify bottlenecks in your pipeline
2. Optimize slow components
3. Set performance baselines
4. Monitor system health over time
