import os
import numpy as np
import time
from enum import StrEnum
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.llms.base import BaseRagasLLM
from ragas.metrics import Metric, context_recall, context_precision, context_entity_recall
from langsmith import Client
from langchain_core.runnables.base import RunnableSequence
from langchain_core.tracers.context import tracing_v2_enabled


class RetrieverType(StrEnum):
    NAIVE = "naive"
    BM25 = "bm25"
    COMPRESSION = "compression"
    MULTI_QUERY = "multi_query"
    PARENT_DOCUMENT = "parent_document"
    ENSEMBLE = "ensemble"
   

def evaluate_retriever(
    retriever_chain: RunnableSequence,
    name: str | RetrieverType,
    testset_df: pd.DataFrame,
    evaluator_llm: BaseRagasLLM,
    metrics: List[Metric],
) -> Dict[str, Any]:
    """Evaluate a single retriever using evaluation metrics,
    
    Args:
        retriever_chain: Your pre-built retriever instance (LCEL chain, etc.)
        name: Display name for this retriever
        testset_df: Pandas DataFrame with 'user_input', 'reference', and 'reference_contexts' columns (can have additional columns too)
        evaluator_llm: RAGAS LLM instance for evaluation
        metrics: List of RAGAS metrics (e.g., [context_recall, context_precision])
        
    Returns:
        Dictionary with quality metrics (RAGAS) and performance metrics (LangSmith)
    """

    print(f"\n{'='*70}")
    print(f"🔍 Evaluating: {name} retriever")
    print(f"{'='*70}\n")

    # Step 1: Setup separate LangSmith project for this retriever
    retriever_project = None
    
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        # Create a unique project name for this retriever
        retriever_project = f"{name}_retriever_eval"
        
        # Delete existing project if it exists
        try:
            client = Client()
            try:
                client.read_project(project_name=retriever_project)
                # Project exists, delete it
                client.delete_project(project_name=retriever_project)
                print(f"🗑️  Deleted existing project: {retriever_project}")
                time.sleep(2)  # Brief pause for deletion to complete
            except Exception:
                # Project doesn't exist, no need to delete
                pass
        except Exception as e:
            print(f"⚠️ Warning: Could not check/delete existing project: {e}")
        
        os.environ["LANGCHAIN_PROJECT"] = retriever_project
        print(f"📁 LangSmith Project: {retriever_project}\n")
    
    # Step 2: Prepare evaluation dataset (retrieve for each question)
    print(f"📊 Retrieving for {len(testset_df)} test questions...")

    # make a copy of testset_df to avoid modifying the original dataframe
    testset_df_copy = testset_df.copy(deep=True)
    responses, retrieved_contexts = [], []

    for idx, row in testset_df_copy.iterrows():
        with tracing_v2_enabled(project_name=retriever_project):
            retriever_output = retriever_chain.invoke(
                {"question": str(row['user_input'])},
                config = {
                    "run_name": f"{name}_retrieval",
                    "tags": [name, "evaluation"],
                    "metadata": {"question_idx": idx, "retriever_name": name}
                }
            )
        responses.append(retriever_output['response'].content)
        retrieved_contexts.append([doc.page_content for doc in retriever_output['context']])

    testset_df_copy['response'] = responses
    testset_df_copy['retrieved_contexts'] = retrieved_contexts

    print(f" Retrieval for {name} completed for {len(testset_df_copy)} eval samples.\n")
    
    # Step 3: Run RAGAS evaluation
    print(f"Running RAGAS evaluation ...")
    eval_dataset = EvaluationDataset.from_pandas(testset_df_copy)
    
    ragas_start = time.time()
    ragas_results = evaluate(dataset=eval_dataset, metrics=metrics, llm=evaluator_llm)
    print(ragas_results)
    ragas_duration = time.time() - ragas_start
    
    print(f"✅ RAGAS evaluation completed in {ragas_duration:.2f}s\n")
    
    # Step 4: Get cost/latency from LangSmith (optional)
    if retriever_project:
        print("💰 Fetching cost and latency from LangSmith...")
        time.sleep(3)  # Brief pause for LangSmith to sync
        cost_latency_metrics = _get_cost_latency_metrics(retriever_project)
    else:
        print("LangSmith tracing disabled - skipping cost/latency metrics")
        cost_latency_metrics = _empty_metrics()
    
    # Step 5: Extract quality metrics from RAGAS results
    # Convert to mean values since ragas_results contains per-sample scores
    quality_metrics = {
        metric.name: np.mean(ragas_results[metric.name]) for metric in metrics
    }
    
    # Step 6: retrun results
    return {
        'name': name,
        'quality_metrics': quality_metrics,
        'performance_metrics': cost_latency_metrics,
        'project_duration': ragas_duration,
    }


def _get_cost_latency_metrics(project_name: str) -> Dict[str, Any]:
    """Extract cost and latency from LangSmith project statistics"""
    try:
        client = Client()
        
        # Get project-level statistics with pre-calculated metrics
        project = client.read_project(project_name=project_name, include_stats=True)
        
        # Also list runs to get count and calculate total cost/tokens
        runs = list(client.list_runs(
            project_name=project_name,
            limit=100
        ))
        
        if not runs:
            print("No LangSmith runs found - sending empty metrics")
            return _empty_metrics()
        
        # Calculate total costs
        total_cost = sum(getattr(run, 'total_cost', 0) or 0 for run in runs) 
        prompt_cost = sum(getattr(run, 'prompt_cost', 0) or 0 for run in runs) 
        completion_cost = sum(getattr(run, 'completion_cost', 0) or 0 for run in runs) 
        
        # calculate total tokens used 
        total_tokens = sum(getattr(run, 'total_tokens', 0) or 0 for run in runs)
        
        # Extract latency statistics from project (returned as timedelta objects)
        latency_p50 = getattr(project, 'latency_p50', None)
        latency_p99 = getattr(project, 'latency_p99', None)
        
        return {
            'p50_latency_seconds': latency_p50.total_seconds() if latency_p50 else 0.0,
            'p99_latency_seconds': latency_p99.total_seconds() if latency_p99 else 0.0,
            'cost_per_query_usd': total_cost / len(runs) if runs else 0.0,
            'prompt_cost_per_query_usd': prompt_cost / len(runs) if runs else 0.0,
            'completion_cost_per_query_usd': completion_cost / len(runs) if runs else 0.0,
            'total_tokens_per_query': total_tokens / len(runs) if runs else 0.0,
        }
    except Exception:
        # Silently fail and return empty metrics if LangSmith unavailable
        print("No LangSmith runs found - sending empty metrics")
        return _empty_metrics()


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics structure"""
    return {
        'p50_latency_seconds': 0.0,
        'p99_latency_seconds': 0.0,
        'prompt_cost_per_query_usd': 0.0,
        'completion_cost_per_query_usd': 0.0,
        'cost_per_query_usd': 0.0,
        'total_tokens_per_query': 0.0,
    }


def print_results(result: Dict[str, Any]):
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
    print(f"\n ⚡ COST and LATENCY METRICS:")
    print("-" * 70)
    perf = result['performance_metrics']
    print(f"  Cost per query (USD):............................ ${perf['cost_per_query_usd']:.6f}")
    print(f"  Prompt cost per query (USD):............................ ${perf['prompt_cost_per_query_usd']:.6f}")
    print(f"  Completion cost per query (USD):............................ ${perf['completion_cost_per_query_usd']:.6f}")
    print(f"  Latency (P50):............................ {perf['p50_latency_seconds']:.3f}s")
    print(f"  Latency (P99):............................ {perf['p99_latency_seconds']:.3f}s")
    print(f"  Tokens per query:............................ {perf['total_tokens_per_query']:.3f}")
    print(f"\n{'='*70}\n")


# Convenience: Suggested metrics for each retriever type
SUGGESTED_METRICS = {
    RetrieverType.NAIVE: {
        'metrics': [context_recall, context_precision, context_entity_recall],
        'rationale': 'Baseline semantic performance'
    },
    RetrieverType.BM25: {
        'metrics': [context_recall, context_precision, context_entity_recall],
        'rationale': 'BM25 excels at keyword/entity matching so context entity recall is important.'
    },
    RetrieverType.COMPRESSION: {
        'metrics': [context_recall, context_precision, context_entity_recall],
        'rationale': 'Reranking optimizes for precision'
    },
    RetrieverType.MULTI_QUERY: {
        'metrics': [context_recall, context_precision, context_entity_recall],
        'rationale': 'Query expansion improves recall'
    },
    RetrieverType.PARENT_DOCUMENT: {
        'metrics': [context_recall, context_precision, context_entity_recall],
        'rationale': 'Test completeness vs noise trade-off'
    },
    RetrieverType.ENSEMBLE: {
        'metrics': [context_recall, context_precision, context_entity_recall],
        'rationale': 'Comprehensive evaluation of combined strategies'
    }
}

def get_suggested_metrics(retriever_type: RetrieverType) -> List[Metric]:
    """Get suggested metrics for a retriever type."""

    if retriever_type in SUGGESTED_METRICS:
        return SUGGESTED_METRICS[retriever_type]['metrics']
    return [context_recall, context_precision]


def print_metrics_guide():
    """Print guide of suggested metrics for each retriever type"""
    print("\n" + "="*80)
    print("SUGGESTED METRICS GUIDE")
    print("="*80 + "\n")
    
    for retriever_type, info in SUGGESTED_METRICS.items():
        print(f"📍 {retriever_type.value}")
        print("-" * 70)
        print(f"Metrics: {info['metrics']}")
        print(f"Rationale: {info['rationale']}")
        print("-" * 70 + "\n")