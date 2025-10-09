# 🎯 Simple Retriever Evaluation - Quick Start

## ✨ Super Simple API

No OOP, no configuration, just pass your retrievers and get results!

## 📖 Quick Start

### 1. Import
```python
from retriever_evaluator import evaluate_retriever, compare_retrievers
from ragas.metrics import context_recall, context_precision, context_entity_recall
```

### 2. Evaluate Single Retriever

```python
# Your pre-built retriever (LCEL chain, LangChain retriever, whatever!)
result = evaluate_retriever(
    retriever=naive_retriever,          # Your retriever instance
    testset=testset,                    # RAGAS testset
    metrics=[context_recall, context_precision],  # Metrics to evaluate
    name="Naive Retriever",             # Display name
    description="Baseline vector similarity"  # Optional
)

# Access results
print(f"Recall: {result['quality_metrics']['context_recall']:.4f}")
print(f"Cost: ${result['performance_metrics']['cost_per_query_usd']:.6f}")
```

### 3. Compare Multiple Retrievers

```python
# Compare all your retrievers at once
results_df = compare_retrievers([
    # (retriever, metrics, name, description)
    (naive_retriever, [context_recall, context_precision], "Naive", "Baseline"),
    (bm25_retriever, [context_recall, context_entity_recall], "BM25", "Keyword"),
    (compression_retriever, [context_precision], "Rerank", "Cohere rerank"),
    (multi_query_retriever, [context_recall, context_entity_recall], "Multi-Query", "Query expansion"),
    (parent_document_retriever, [context_recall, context_precision], "Parent Doc", "Small-to-big"),
    (ensemble_retriever, [context_recall, context_precision, context_entity_recall], "Ensemble", "All combined"),
], testset)

# Results automatically displayed!
# Returns DataFrame for further analysis
```

## 🎨 Complete Example

```python
# After building your retrievers...

from retriever_evaluator import compare_retrievers
from ragas.metrics import context_recall, context_precision, context_entity_recall

# Define what to evaluate
retrievers_to_compare = [
    # Naive with semantic chunking
    (naive_semantic_retriever, 
     [context_recall, context_precision], 
     "Naive (Semantic Chunking)", 
     "Vector similarity with semantic chunking"),
    
    # Naive with recursive chunking
    (naive_recursive_retriever, 
     [context_recall, context_precision], 
     "Naive (Recursive Chunking)", 
     "Vector similarity with recursive chunking"),
    
    # BM25 (no chunking needed)
    (bm25_retriever, 
     [context_recall, context_entity_recall], 
     "BM25", 
     "Keyword-based retrieval"),
    
    # Compression/Rerank
    (compression_retriever, 
     [context_precision], 
     "Rerank", 
     "Cohere reranking on top of naive"),
    
    # Multi-Query
    (multi_query_retriever, 
     [context_recall, context_entity_recall], 
     "Multi-Query", 
     "Query expansion for better recall"),
    
    # Ensemble
    (ensemble_retriever, 
     [context_recall, context_precision, context_entity_recall], 
     "Ensemble", 
     "All strategies combined with RRF"),
]

# Run comparison
comparison_df = compare_retrievers(retrievers_to_compare, testset)

# Save results
comparison_df.to_csv('retriever_comparison.csv', index=False)
```

## 💡 What Metrics to Use?

### Option 1: Use Suggested Metrics

```python
from retriever_evaluator import print_metrics_guide

# Print guide
print_metrics_guide()

# Output:
# ================================================================================
# SUGGESTED METRICS GUIDE
# ================================================================================
#
# 📍 NAIVE
# --------------------------------------------------------------------------------
# Metrics: context_recall, context_precision
# Rationale: Baseline semantic performance - test both recall and precision
#
# 📍 BM25
# --------------------------------------------------------------------------------
# Metrics: context_recall, context_entity_recall
# Rationale: BM25 excels at keyword/entity matching
# ...
```

### Option 2: Choose Your Own

Just pass any RAGAS metrics you want:

```python
from ragas.metrics import (
    context_recall,
    context_precision,
    context_entity_recall,
    # ... any RAGAS metric
)

result = evaluate_retriever(
    retriever=my_retriever,
    testset=testset,
    metrics=[context_recall, context_precision],  # Your choice!
    name="My Retriever"
)
```

## 📊 Output Example

```
======================================================================
🔍 Evaluating: Naive Retriever
======================================================================

📊 Retrieving for 15 test questions...
  ✓ Retrieval complete for 15 questions

📈 Running RAGAS evaluation with 2 metrics...
✅ RAGAS evaluation completed in 12.34s

💰 Fetching cost and latency from LangSmith...

======================================================================
📊 RESULTS: Naive Retriever
======================================================================

🎯 QUALITY METRICS (RAGAS):
----------------------------------------------------------------------
  context_recall........................................ 0.8542
  context_precision..................................... 0.7834

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

## 🔑 Key Advantages

### ✅ Simple
- No classes, no configuration objects
- Just functions
- Pass your retrievers directly

### ✅ Flexible
- Use any retriever (LCEL chain, LangChain retriever, custom)
- Choose any metrics
- Works with what you've already built

### ✅ Complete
- RAGAS quality metrics
- LangSmith cost tracking
- LangSmith latency tracking
- All in one function call

### ✅ Direct
- Get results immediately
- No need to call multiple methods
- Results printed automatically

## 🆚 When to Use Simple vs OOP API

### Use Simple API When:
- ✅ You've already built your retrievers
- ✅ You want quick evaluation
- ✅ You want minimal code
- ✅ You know which metrics to use

### Use OOP API When:
- You want automatic metric selection based on type
- You need extensive customization
- You're building a retriever evaluation pipeline
- You want chunking strategy management

## 📝 Simple vs OOP Comparison

### Simple API (New)
```python
from retriever_evaluator import evaluate_retriever
from ragas.metrics import context_recall, context_precision

result = evaluate_retriever(
    retriever=naive_retriever,  # Your pre-built retriever
    testset=testset,
    metrics=[context_recall, context_precision],
    name="Naive"
)
```

### OOP API (Advanced)
```python
from retriever_evaluator import RetrieverBenchmark

benchmark = RetrieverBenchmark(testset)
result = benchmark.add_retriever(
    'naive',  # Type (auto-selects metrics)
    naive_retriever,
    chunking_strategy='recursive'
)
```

Both work! Use what fits your workflow. 🎯

## 🚀 Getting Started

1. **Build your retrievers** (you've already done this!)
2. **Generate testset** (RAGAS)
3. **Evaluate**:
   ```python
   from retriever_evaluator import compare_retrievers
   from ragas.metrics import context_recall, context_precision
   
   results = compare_retrievers([
       (retriever1, [context_recall, context_precision], "Retriever 1", ""),
       (retriever2, [context_recall, context_precision], "Retriever 2", ""),
   ], testset)
   ```

That's it! 🎉

