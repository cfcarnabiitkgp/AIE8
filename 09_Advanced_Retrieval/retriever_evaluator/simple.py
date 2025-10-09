"""
Simple function-based retriever evaluation

Usage:
    from retriever_evaluator.simple import evaluate_retriever, compare_retrievers
    
    result = evaluate_retriever(
        retriever=my_retriever,
        testset=testset,
        metrics=[context_recall, context_precision],
        name="My Retriever"
    )
"""

import os
import time
from typing import List, Any, Dict, Optional, Tuple
import pandas as pd
from datetime import datetime
from langsmith import Client
from datasets import Dataset
from ragas import evaluate


def evaluate_retriever(
    retriever,
    testset: Dataset,
    metrics: List[Any],
    name: str = "Retriever",
    description: str = ""
) -> Dict[str, Any]:
    """
    Evaluate a single retriever (simple and direct!)
    
    Args:
        retriever: Your pre-built retriever instance (LCEL chain, etc.)
        testset: RAGAS testset with questions and ground truth
        metrics: List of RAGAS metrics (e.g., [context_recall, context_precision])
        name: Display name for this retriever
        description: Optional description
        
    Returns:
        Dictionary with quality metrics (RAGAS) and performance metrics (LangSmith)
    """
    print(f"\n{'='*70}")
    print(f"🔍 Evaluating: {name}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*70}\n")
    
    # Step 1: Prepare evaluation dataset (retrieve for each question)
    print(f"📊 Retrieving for {len(testset)} test questions...")
    
    questions = []
    retrieved_contexts = []
    ground_truths = []
    
    testset_df = testset.to_pandas() if hasattr(testset, 'to_pandas') else testset
    
    for idx, row in testset_df.iterrows():
        question = row['user_input']
        ground_truth = row.get('reference', row.get('ground_truth', ''))
        
        # Retrieve with LangSmith tracing (automatic cost/latency tracking)
        docs = retriever.invoke(
            question,
            config={
                "run_name": f"{name}_retrieval",
                "tags": [name, "evaluation"],
                "metadata": {"question_idx": idx, "retriever_name": name}
            }
        )
        
        # Handle both list of Documents and direct list of strings
        if docs and hasattr(docs[0], 'page_content'):
            contexts = [doc.page_content for doc in docs]
        else:
            contexts = docs if isinstance(docs, list) else [str(docs)]
        
        questions.append(question)
        retrieved_contexts.append(contexts)
        ground_truths.append(ground_truth)
    
    print(f"  ✓ Retrieval complete for {len(questions)} questions\n")
    
    # Step 2: Run RAGAS evaluation
    print(f"📈 Running RAGAS evaluation with {len(metrics)} metrics...")
    eval_dataset = Dataset.from_dict({
        'question': questions,
        'contexts': retrieved_contexts,
        'ground_truth': ground_truths
    })
    
    ragas_start = time.time()
    ragas_results = evaluate(dataset=eval_dataset, metrics=metrics)
    ragas_duration = time.time() - ragas_start
    
    print(f"✅ RAGAS evaluation completed in {ragas_duration:.2f}s\n")
    
    # Step 3: Get cost/latency from LangSmith (optional)
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        print("💰 Fetching cost and latency from LangSmith...")
        time.sleep(3)  # Brief pause for LangSmith to sync
        langsmith_metrics = _get_langsmith_metrics(name)
    else:
        print("⚠️  LangSmith tracing disabled - skipping cost/latency metrics")
        langsmith_metrics = _empty_metrics()
    
    # Step 4: Extract quality metrics from RAGAS results
    # RAGAS returns an EvaluationResult object, extract scores as dict
    quality_metrics = {}
    if hasattr(ragas_results, 'to_pandas'):
        # Convert to pandas and get mean scores for numeric columns only
        df = ragas_results.to_pandas()
        for col in df.columns:
            # Only process numeric columns (metric scores)
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                quality_metrics[col] = float(df[col].mean())
    elif hasattr(ragas_results, '_scores_dict'):
        # Direct access to scores dict
        for key, values in ragas_results._scores_dict.items():
            if isinstance(values, list):
                quality_metrics[key] = sum(values) / len(values) if values else 0
            else:
                quality_metrics[key] = values
    else:
        # Fallback: try to iterate
        for metric in metrics:
            metric_name = metric.name if hasattr(metric, 'name') else str(metric)
            if hasattr(ragas_results, metric_name):
                quality_metrics[metric_name] = getattr(ragas_results, metric_name)
    
    # Step 5: Compile results
    result = {
        'name': name,
        'description': description,
        'quality_metrics': quality_metrics,
        'performance_metrics': langsmith_metrics,
        'ragas_duration': ragas_duration,
        'timestamp': datetime.now().isoformat()
    }
    
    _print_results(result)
    return result


def compare_retrievers(
    retrievers: List[Tuple[Any, List[Any], str, str]],
    testset: Dataset
) -> pd.DataFrame:
    """
    Compare multiple retrievers
    
    Args:
        retrievers: List of (retriever, metrics, name, description) tuples
        testset: RAGAS testset
        
    Returns:
        DataFrame with comparison of all retrievers
        
    Example:
        results_df = compare_retrievers([
            (naive_retriever, [context_recall, context_precision], "Naive", "Baseline"),
            (bm25_retriever, [context_recall, context_entity_recall], "BM25", "Keyword search"),
        ], testset)
    """
    results = []
    
    print("\n" + "="*80)
    print("STARTING RETRIEVER COMPARISON")
    print("="*80)
    
    for retriever, metrics, name, description in retrievers:
        result = evaluate_retriever(retriever, testset, metrics, name, description)
        results.append(result)
    
    print("\n" + "="*80)
    print("✅ ALL EVALUATIONS COMPLETE")
    print("="*80 + "\n")
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results:
        row = {
            'Retriever': result['name'],
            'Description': result['description']
        }
        
        # Quality metrics
        for metric, score in result['quality_metrics'].items():
            row[f'Q_{metric}'] = score
        
        # Performance metrics
        perf = result['performance_metrics']
        row['Cost_USD'] = perf['total_cost_usd']
        row['Cost_per_Query'] = perf['cost_per_query_usd']
        row['Avg_Latency_s'] = perf['avg_latency_seconds']
        row['Total_Tokens'] = perf['total_tokens']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    print("📊 COMPARISON RESULTS:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def _get_langsmith_metrics(retriever_name: str) -> Dict[str, Any]:
    """Extract cost and latency from LangSmith traces"""
    try:
        client = Client()
        project_name = os.environ.get("LANGCHAIN_PROJECT")
        
        runs = list(client.list_runs(
            project_name=project_name,
            filter=f'name == "{retriever_name}_retrieval"',
            limit=100
        ))
        
        if not runs:
            print(f"⚠️  No LangSmith runs found - metrics will be zero")
            return _empty_metrics()
        
        total_cost = sum(getattr(run, 'total_cost', 0) or 0 for run in runs)
        
        latencies = []
        for run in runs:
            if hasattr(run, 'end_time') and hasattr(run, 'start_time') and run.end_time and run.start_time:
                latency = (run.end_time - run.start_time).total_seconds()
                latencies.append(latency)
        
        total_tokens = sum(getattr(run, 'total_tokens', 0) or 0 for run in runs)
        
        return {
            'total_cost_usd': total_cost,
            'avg_latency_seconds': sum(latencies) / len(latencies) if latencies else 0,
            'min_latency_seconds': min(latencies) if latencies else 0,
            'max_latency_seconds': max(latencies) if latencies else 0,
            'p50_latency_seconds': sorted(latencies)[len(latencies)//2] if latencies else 0,
            'p95_latency_seconds': sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            'total_tokens': total_tokens,
            'num_queries': len(runs),
            'cost_per_query_usd': total_cost / len(runs) if runs else 0
        }
    except Exception as e:
        # Silently fail and return empty metrics if LangSmith unavailable
        print(f"⚠️  LangSmith unavailable (API key issue) - cost/latency metrics will be zero")
        return _empty_metrics()


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure"""
    return {
        'total_cost_usd': 0,
        'avg_latency_seconds': 0,
        'min_latency_seconds': 0,
        'max_latency_seconds': 0,
        'p50_latency_seconds': 0,
        'p95_latency_seconds': 0,
        'total_tokens': 0,
        'num_queries': 0,
        'cost_per_query_usd': 0
    }


def _print_results(result: Dict[str, Any]):
    """Pretty print evaluation results"""
    print(f"\n{'='*70}")
    print(f"📊 RESULTS: {result['name']}")
    print(f"{'='*70}")
    
    # Quality Metrics
    print("\n🎯 QUALITY METRICS (RAGAS):")
    print("-" * 70)
    for metric, score in result['quality_metrics'].items():
        print(f"  {metric:.<50} {score:.4f}")
    
    # Performance Metrics
    print(f"\n⚡ PERFORMANCE METRICS (LangSmith):")
    print("-" * 70)
    perf = result['performance_metrics']
    print(f"  Total Cost:............................ ${perf['total_cost_usd']:.6f}")
    print(f"  Cost per Query:........................ ${perf['cost_per_query_usd']:.6f}")
    print(f"  Avg Latency:........................... {perf['avg_latency_seconds']:.3f}s")
    print(f"  Min/Max Latency:....................... {perf['min_latency_seconds']:.3f}s / {perf['max_latency_seconds']:.3f}s")
    print(f"  Total Tokens:.......................... {perf['total_tokens']:,}")
    print(f"  Number of Queries:..................... {perf['num_queries']}")
    
    print(f"\n{'='*70}\n")


# Convenience: Suggested metrics for each retriever type
SUGGESTED_METRICS = {
    'naive': {
        'metrics': ['context_recall', 'context_precision'],
        'rationale': 'Baseline semantic performance - test both recall and precision'
    },
    'bm25': {
        'metrics': ['context_recall', 'context_entity_recall'],
        'rationale': 'BM25 excels at keyword/entity matching'
    },
    'compression': {
        'metrics': ['context_precision'],
        'rationale': 'Reranking optimizes for precision'
    },
    'multi_query': {
        'metrics': ['context_recall', 'context_entity_recall'],
        'rationale': 'Query expansion improves recall'
    },
    'parent_document': {
        'metrics': ['context_recall', 'context_precision'],
        'rationale': 'Test completeness vs noise trade-off'
    },
    'ensemble': {
        'metrics': ['context_recall', 'context_precision', 'context_entity_recall'],
        'rationale': 'Comprehensive evaluation of combined strategies'
    }
}


def get_suggested_metrics(retriever_type: str) -> List[str]:
    """
    Get suggested metrics for a retriever type
    
    Args:
        retriever_type: One of 'naive', 'bm25', 'compression', etc.
        
    Returns:
        List of suggested metric names
    """
    if retriever_type in SUGGESTED_METRICS:
        return SUGGESTED_METRICS[retriever_type]['metrics']
    return ['context_recall', 'context_precision']


def print_metrics_guide():
    """Print guide of suggested metrics for each retriever type"""
    print("\n" + "="*80)
    print("SUGGESTED METRICS GUIDE")
    print("="*80 + "\n")
    
    for retriever_type, info in SUGGESTED_METRICS.items():
        print(f"📍 {retriever_type.upper()}")
        print("-" * 80)
        print(f"Metrics: {', '.join(info['metrics'])}")
        print(f"Rationale: {info['rationale']}")
        print()

