"""LangGraph Agent Library

A library for LangGraph agents with caching, monitoring, and agent integration.
"""

from .agents import create_langgraph_agent, create_langgraph_helpful_agent
from .caching import CacheBackedEmbeddings, setup_llm_cache
from .rag import ProductionRAGChain
from .models import get_openai_model

# Guardrails integration (optional - requires guardrails-ai)
try:
    from .agents_with_guardrails import create_guarded_langgraph_agent
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    create_guarded_langgraph_agent = None

__version__ = "0.1.0"
__all__ = [
    "create_langgraph_agent",
    "create_langgraph_helpful_agent",
    "create_guarded_langgraph_agent",
    "CacheBackedEmbeddings",
    "setup_llm_cache",
    "ProductionRAGChain",
    "get_openai_model",
]

