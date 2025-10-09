# Retriever Evaluator - Simple API

A simple function-based framework for evaluating LangChain retrievers using:
- **RAGAS** for quality metrics (recall, precision, entity recall)
- **LangSmith** for automatic cost and latency tracking

## Features

- ✅ **Simple**: Just functions, no classes
- ✅ **Flexible**: Pass any pre-built retriever
- ✅ **Complete**: Quality + Cost + Latency in one call
- ✅ **Direct**: Results printed automatically

## Structure

```
retriever_evaluator/
├── __init__.py           # Package exports
├── simple.py            # Main evaluation functions
└── README.md            # This file
```

## Quick Start

### 1. Setup LangSmith Tracing (for cost/latency tracking)

```python
import os
import getpass

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "retriever-evaluation"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")
```

### 2. Evaluate Single Retriever

```python
from retriever_evaluator import evaluate_retriever
from ragas.metrics import context_recall, context_precision

# Use your pre-built retriever directly!
result = evaluate_retriever(
    retriever=naive_retriever,  # Your LCEL chain or retriever
    testset=testset,            # RAGAS testset
    metrics=[context_recall, context_precision],
    name="Naive Retriever",
    description="Baseline vector similarity"
)

# Access results
print(f"Recall: {result['quality_metrics']['context_recall']:.4f}")
print(f"Cost: ${result['performance_metrics']['cost_per_query_usd']:.6f}")
```

### 3. Compare Multiple Retrievers

```python
from retriever_evaluator import compare_retrievers
from ragas.metrics import context_recall, context_precision, context_entity_recall

# Compare all your retrievers at once
results_df = compare_retrievers([
    # (retriever, metrics, name, description)
    (naive_retriever, [context_recall, context_precision], "Naive", "Baseline"),
    (bm25_retriever, [context_recall, context_entity_recall], "BM25", "Keyword"),
    (compression_retriever, [context_precision], "Rerank", "Cohere rerank"),
    (multi_query_retriever, [context_recall, context_entity_recall], "Multi-Query", "Expansion"),
    (ensemble_retriever, [context_recall, context_precision, context_entity_recall], "Ensemble", "Combined"),
], testset)

# Results DataFrame automatically displayed and returned
results_df.to_csv('comparison.csv', index=False)
```

## Suggested Metrics by Retriever Type

Use `print_metrics_guide()` to see suggestions:

```python
from retriever_evaluator import print_metrics_guide

print_metrics_guide()
```

**Quick Reference:**

| Retriever Type | Suggested Metrics | Why? |
|----------------|-------------------|------|
| Naive | `context_recall`, `context_precision` | Baseline semantic performance |
| BM25 | `context_recall`, `context_entity_recall` | Excels at keyword/entity matching |
| Compression | `context_precision` | Reranking optimizes precision |
| Multi-Query | `context_recall`, `context_entity_recall` | Query expansion improves recall |
| Parent Document | `context_recall`, `context_precision` | Completeness vs noise trade-off |
| Ensemble | All three | Comprehensive evaluation |

## What You Get

### Quality Metrics (RAGAS)
- `context_recall`: Are all relevant docs retrieved?
- `context_precision`: Are retrieved docs well-ranked?
- `context_entity_recall`: Are specific entities found?

### Performance Metrics (LangSmith - Automatic!)
- Total cost and cost per query
- Average, min, max latency
- P50 and P95 latency
- Total tokens used

## Example Output

```
======================================================================
🔍 Evaluating: BM25 Keyword Search
======================================================================

📊 Retrieving for 15 test questions...
  ✓ Retrieval complete for 15 questions

📈 Running RAGAS evaluation with 2 metrics...
✅ RAGAS evaluation completed in 12.34s

💰 Fetching cost and latency from LangSmith...

======================================================================
📊 RESULTS: BM25 Keyword Search
======================================================================

🎯 QUALITY METRICS (RAGAS):
----------------------------------------------------------------------
  context_recall........................................ 0.8542
  context_entity_recall................................. 0.9123

⚡ PERFORMANCE METRICS (LangSmith):
----------------------------------------------------------------------
  Total Cost:............................ $0.002340
  Cost per Query:........................ $0.000156
  Avg Latency:........................... 0.234s
  Min/Max Latency:....................... 0.221s / 0.312s
  Total Tokens:.......................... 1,234
  Number of Queries:..................... 15

======================================================================
```

## Requirements

```
langchain
ragas
langsmith
datasets
pandas
```

