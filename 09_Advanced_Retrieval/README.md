# Advanced Retrieval with LangChain - Assignment

## 📁 Project Structure

```
09_Advanced_Retrieval/
├── Advanced_Retrieval_with_LangChain_Assignment.ipynb  # Main notebook
├── retriever_evaluator/                                # Evaluation framework
│   ├── __init__.py                                     # Package exports
│   ├── simple.py                                       # Simple evaluation functions
│   └── README.md                                       # Module documentation
├── USAGE.md                                            # Quick start guide
├── data/                                               # Your data files
└── pyproject.toml                                      # Dependencies
```

## 🎯 Quick Start - Evaluating Retrievers

### 1. Setup LangSmith (for cost/latency tracking)

```python
import os
import getpass

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "retriever-evaluation"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")
```

### 2. Evaluate a Single Retriever

```python
from retriever_evaluator import evaluate_retriever
from ragas.metrics import context_recall, context_precision

result = evaluate_retriever(
    retriever=naive_retriever,  # Your pre-built retriever
    testset=testset,            # RAGAS testset
    metrics=[context_recall, context_precision],
    name="Naive Retriever"
)
```

### 3. Compare Multiple Retrievers

```python
from retriever_evaluator import compare_retrievers
from ragas.metrics import context_recall, context_precision, context_entity_recall

results_df = compare_retrievers([
    (naive_retriever, [context_recall, context_precision], "Naive", "Baseline"),
    (bm25_retriever, [context_recall, context_entity_recall], "BM25", "Keyword"),
    (compression_retriever, [context_precision], "Rerank", "Cohere"),
    (multi_query_retriever, [context_recall, context_entity_recall], "Multi-Query", "Expansion"),
    (ensemble_retriever, [context_recall, context_precision, context_entity_recall], "Ensemble", "Combined"),
], testset)
```

## 📊 What You Get

### Quality Metrics (RAGAS)
- `context_recall`: Are all relevant docs retrieved?
- `context_precision`: Are retrieved docs well-ranked?
- `context_entity_recall`: Are specific entities found?

### Performance Metrics (LangSmith - Automatic!)
- Total cost and cost per query
- Average, min, max, P50, P95 latency
- Total tokens used

## 📖 Documentation

- **USAGE.md** - Complete usage guide with examples
- **retriever_evaluator/README.md** - Module documentation
- **Advanced_Retrieval_with_LangChain_Assignment.ipynb** - Main notebook

## 🔑 Key Features

- ✅ **Simple**: Just 2 functions - `evaluate_retriever()` and `compare_retrievers()`
- ✅ **Flexible**: Pass any pre-built retriever (LCEL chains, LangChain retrievers, etc.)
- ✅ **Complete**: Quality (RAGAS) + Cost + Latency in one call
- ✅ **Direct**: Results printed automatically

## 💡 Suggested Metrics by Retriever Type

| Retriever | Metrics | Why? |
|-----------|---------|------|
| Naive | `context_recall`, `context_precision` | Baseline semantic performance |
| BM25 | `context_recall`, `context_entity_recall` | Excels at keyword/entity matching |
| Compression | `context_precision` | Reranking optimizes precision |
| Multi-Query | `context_recall`, `context_entity_recall` | Query expansion improves recall |
| Parent Document | `context_recall`, `context_precision` | Completeness vs noise trade-off |
| Ensemble | All three | Comprehensive evaluation |

## 🚀 Assignment Goals

1. ✅ Create a "golden dataset" using Synthetic Data Generation (RAGAS)
2. ✅ Evaluate each retriever with appropriate RAGAS metrics
3. ✅ Track cost and latency using LangSmith
4. ✅ Compare retrievers and analyze results
5. ✅ Write analysis considering cost, latency, and performance

## 📝 Example Usage

See `USAGE.md` for complete examples and best practices.

---

**Version**: 2.0.0 (Simplified)
**Framework**: Simple function-based API for maximum flexibility
