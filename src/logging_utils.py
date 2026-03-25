import logging
import os
import time
import json
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Any

LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'query_timing.log')

# Global timing stats for the current query
_current_query_timing: Dict[str, float] = {}
_current_query_details: Dict[str, Any] = {}


def setup_logger(name: str = 'aloo'):
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s')

    # Rotating file handler
    fh = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Also log to console for convenience
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


@contextmanager
def timer(logger, label: str, **extra):
    """Context manager to time code blocks and log results"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        extra['duration_ms'] = round(elapsed * 1000, 2)
        log_timing(logger, label, extra)


def log_timing(logger, component: str, metrics: Dict[str, Any], query_id: Optional[str] = None):
    """Log timing metrics with consistent formatting"""
    message = f"TIMING | {component:<30} | "
    
    # Extract duration if present
    duration = metrics.pop('duration_ms', None)
    
    # Format other metrics
    metric_strs = [f"{k}={v}" for k, v in metrics.items()]
    
    if duration is not None:
        message += f"Duration: {duration}ms"
        if metric_strs:
            message += " | " + " | ".join(metric_strs)
    else:
        message += " | ".join(metric_strs) if metric_strs else ""
    
    # Store for query summary
    if query_id:
        _current_query_timing[component] = duration
        _current_query_details[component] = metrics
    
    logger.info(message)


def log_query_event(logger, event: str, details: Dict[str, Any] = None):
    """Log query processing events"""
    message = f"EVENT | {event}"
    if details:
        detail_str = " | ".join([f"{k}={v}" for k, v in details.items()])
        message += f" | {detail_str}"
    logger.info(message)


def log_query_start(logger, query_id: str, question: str):
    """Log the start of query processing"""
    global _current_query_timing, _current_query_details
    _current_query_timing = {}
    _current_query_details = {}
    
    logger.info("=" * 100)
    logger.info(f"QUERY_START | ID: {query_id} | Question: {question[:80]}...")
    logger.info("=" * 100)


def log_query_complete(logger, query_id: str, total_time: float, status: str = "SUCCESS"):
    """Log query completion with timing summary"""
    logger.info("=" * 100)
    logger.info(f"QUERY_COMPLETE | ID: {query_id} | Status: {status}")
    logger.info(f"TOTAL_TIME | {round(total_time * 1000, 2)}ms")
    
    # Log component breakdown
    if _current_query_timing:
        logger.info("TIMING_BREAKDOWN:")
        sorted_timings = sorted(_current_query_timing.items(), key=lambda x: x[1] if x[1] else 0, reverse=True)
        for component, duration in sorted_timings:
            if duration is not None:
                logger.info(f"  {component:<30} | {duration}ms")
    
    logger.info("=" * 100)


def log_retrieval_metrics(logger, query_id: str, num_docs: int, retrieval_time: float, 
                         rank_time: Optional[float] = None, methods: Optional[List[str]] = None):
    """Log retrieval-specific metrics"""
    metrics = {
        'num_docs': num_docs,
        'retrieval_ms': round(retrieval_time * 1000, 2),
    }
    
    if rank_time is not None:
        metrics['rerank_ms'] = round(rank_time * 1000, 2)
    
    if methods:
        metrics['methods'] = ','.join(methods)
    
    log_timing(logger, "RETRIEVAL", metrics, query_id)


def log_generation_metrics(logger, query_id: str, generation_time: float, 
                          chunk_count: Optional[int] = None, token_count: Optional[int] = None):
    """Log generation-specific metrics"""
    metrics = {
        'generation_ms': round(generation_time * 1000, 2),
    }
    
    if chunk_count is not None:
        metrics['chunks'] = chunk_count
    
    if token_count is not None:
        metrics['tokens'] = token_count
    
    log_timing(logger, "GENERATION", metrics, query_id)


def get_timing_summary() -> Dict[str, float]:
    """Get the timing summary for the current query"""
    return _current_query_timing.copy()
