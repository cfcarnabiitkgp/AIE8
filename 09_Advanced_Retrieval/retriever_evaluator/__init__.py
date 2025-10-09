"""
Retriever Evaluation Framework

Simple function-based evaluation for LangChain retrievers using:
- RAGAS for quality metrics (recall, precision, entity recall)
- LangSmith for cost and latency tracking (automatic)

Usage:
    from retriever_evaluator import evaluate_retriever, compare_retrievers
    
    # Evaluate single retriever
    result = evaluate_retriever(
        retriever=my_retriever,
        testset=testset,
        metrics=[context_recall, context_precision],
        name="My Retriever"
    )
    
    # Compare multiple retrievers
    results_df = compare_retrievers([
        (retriever1, metrics1, "Name 1", "Description"),
        (retriever2, metrics2, "Name 2", "Description"),
    ], testset)
"""

from .simple import (
    evaluate_retriever,
    compare_retrievers,
    print_metrics_guide,
    get_suggested_metrics,
    SUGGESTED_METRICS
)

__version__ = "2.0.0"  # Simplified version
__all__ = [
    'evaluate_retriever',
    'compare_retrievers',
    'print_metrics_guide',
    'get_suggested_metrics',
    'SUGGESTED_METRICS'
]


