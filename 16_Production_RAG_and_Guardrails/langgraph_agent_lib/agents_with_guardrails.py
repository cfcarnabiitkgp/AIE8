"""LangGraph agent with integrated Guardrails for production safety.

This module provides a production-safe LangGraph agent that integrates
Guardrails AI for input/output validation.
"""

from typing import Dict, Any, List, Optional
import time

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain
from .agents import get_default_tools

# Guardrails imports
try:
    from guardrails.hub import (
        RestrictToTopic,
        DetectJailbreak,
        ProfanityFree,
        GuardrailsPII
    )
    from guardrails import Guard
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    print("⚠️ Guardrails not available. Install with: uv sync")


class GuardedAgentState(TypedDict):
    """Enhanced state schema with guardrail tracking."""
    messages: Annotated[List[BaseMessage], add_messages]
    guard_logs: List[Dict[str, Any]]  # Track guard activations
    validation_failures: int  # Count refinement loops


def create_input_guards(valid_topics: List[str], invalid_topics: List[str]) -> Guard:
    """Create input validation guard combining multiple checks.

    Args:
        valid_topics: List of allowed conversation topics
        invalid_topics: List of prohibited topics

    Returns:
        Configured Guard instance for input validation
    """
    if not GUARDRAILS_AVAILABLE:
        raise ImportError("Guardrails not available. Please install guardrails-ai")

    # Create combined input guard
    guard = Guard()

    # 1. Jailbreak Detection (runs first - critical security)
    guard = guard.use(DetectJailbreak(on_fail="exception"))

    # 2. Topic Restriction (ensure on-topic conversations)
    guard = guard.use(
        RestrictToTopic(
            valid_topics=valid_topics,
            invalid_topics=invalid_topics,
            disable_classifier=True,
            disable_llm=False,
            on_fail="exception"
        )
    )

    # 3. PII Detection in input (sanitize sensitive data)
    guard = guard.use(
        GuardrailsPII(
            entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
            on_fail="fix"  # Redact PII instead of blocking
        )
    )

    return guard


def create_output_guards() -> Guard:
    """Create output validation guard for agent responses.

    Returns:
        Configured Guard instance for output validation
    """
    if not GUARDRAILS_AVAILABLE:
        raise ImportError("Guardrails not available. Please install guardrails-ai")

    # Create combined output guard
    guard = Guard()

    # 1. Profanity Check (keep responses professional)
    guard = guard.use(
        ProfanityFree(
            threshold=0.8,
            validation_method="sentence",
            on_fail="exception"
        )
    )

    # 2. PII Leakage Detection (prevent data exposure)
    guard = guard.use(
        GuardrailsPII(
            entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
            on_fail="fix"  # Redact leaked PII
        )
    )

    return guard


def create_guarded_langgraph_agent(
    model_name: str = "gpt-4.1-mini",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    valid_topics: Optional[List[str]] = None,
    invalid_topics: Optional[List[str]] = None,
    enable_input_guards: bool = True,
    enable_output_guards: bool = True,
    max_refinement_loops: int = 3
):
    """Create a LangGraph agent with integrated Guardrails.

    This agent includes:
    - Input validation (jailbreak, topic, PII detection)
    - Output validation (profanity, PII leakage)
    - Graceful error handling
    - Refinement loops for failed validations

    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        valid_topics: List of allowed topics (for topic restriction)
        invalid_topics: List of prohibited topics
        enable_input_guards: Whether to enable input validation
        enable_output_guards: Whether to enable output validation
        max_refinement_loops: Maximum refinement attempts for failed output validation

    Returns:
        Compiled LangGraph agent with guardrails
    """
    if not GUARDRAILS_AVAILABLE:
        raise ImportError(
            "Guardrails not available. Please install with: "
            "uv sync && uv run guardrails configure"
        )

    # Default topics for student loan domain
    if valid_topics is None:
        valid_topics = [
            "student loans", "financial aid", "education financing",
            "loan repayment", "loan forgiveness", "student debt"
        ]

    if invalid_topics is None:
        invalid_topics = [
            "investment advice", "cryptocurrency", "gambling",
            "politics", "medical advice"
        ]

    # Get tools
    if tools is None:
        tools = get_default_tools(rag_chain)

    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    # Create guards
    input_guard = create_input_guards(valid_topics, invalid_topics) if enable_input_guards else None
    output_guard = create_output_guards() if enable_output_guards else None

    # Node functions
    def input_guard_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate user input before agent processing."""
        if not enable_input_guards or not input_guard:
            return {"guard_logs": [{"node": "input_guard", "status": "skipped"}]}

        user_message = state["messages"][-1].content
        guard_log = {
            "node": "input_guard",
            "timestamp": time.time(),
            "status": "checking"
        }

        try:
            # Run input validation
            result = input_guard.validate(user_message)

            # Check if any validation failed
            if not result.validation_passed:
                guard_log["status"] = "failed"
                guard_log["failures"] = str(result.error)
                # Add blocking message
                return {
                    "messages": [AIMessage(content=f"❌ Input validation failed: {result.error}")],
                    "guard_logs": state.get("guard_logs", []) + [guard_log],
                    "blocked": True
                }

            guard_log["status"] = "passed"

            # If PII was redacted, update the message
            if result.validated_output != user_message:
                guard_log["pii_redacted"] = True
                # Replace user message with sanitized version
                # Ensure validated_output is a string
                sanitized_content = (
                    result.validated_output
                    if isinstance(result.validated_output, str)
                    else str(result.validated_output)
                )
                sanitized_messages = state["messages"][:-1] + [
                    HumanMessage(content=sanitized_content)
                ]
                return {
                    "messages": sanitized_messages,
                    "guard_logs": state.get("guard_logs", []) + [guard_log]
                }

            return {"guard_logs": state.get("guard_logs", []) + [guard_log]}

        except Exception as e:
            # Guard raised exception (jailbreak or off-topic)
            guard_log["status"] = "blocked"
            guard_log["reason"] = str(e)

            return {
                "messages": [AIMessage(content=f"🛡️ Request blocked: {str(e)}")],
                "guard_logs": state.get("guard_logs", []) + [guard_log],
                "blocked": True
            }

    def call_model(state: GuardedAgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def output_guard_node(state: GuardedAgentState) -> Dict[str, Any]:
        """Validate agent output before returning to user."""
        if not enable_output_guards or not output_guard:
            return {"guard_logs": state.get("guard_logs", []) + [{"node": "output_guard", "status": "skipped"}]}

        agent_response = state["messages"][-1].content
        guard_log = {
            "node": "output_guard",
            "timestamp": time.time(),
            "status": "checking"
        }

        try:
            # Run output validation
            result = output_guard.validate(agent_response)

            if not result.validation_passed:
                guard_log["status"] = "failed"
                guard_log["failures"] = str(result.error)

                # Check if we've exceeded max refinement loops
                current_failures = state.get("validation_failures", 0)
                if current_failures >= max_refinement_loops:
                    guard_log["max_retries_exceeded"] = True
                    return {
                        "messages": [AIMessage(content="⚠️ Unable to generate safe response. Please rephrase your question.")],
                        "guard_logs": state.get("guard_logs", []) + [guard_log],
                        "needs_refinement": False
                    }

                # Trigger refinement
                refinement_prompt = (
                    "The previous response failed validation. "
                    "Please provide a more appropriate response that is professional, "
                    "factual, and does not contain sensitive information."
                )

                return {
                    "messages": state["messages"] + [HumanMessage(content=refinement_prompt)],
                    "guard_logs": state.get("guard_logs", []) + [guard_log],
                    "validation_failures": current_failures + 1,
                    "needs_refinement": True
                }

            guard_log["status"] = "passed"

            # If PII was redacted in output, update the message
            if result.validated_output != agent_response:
                guard_log["pii_redacted"] = True
                # Ensure validated_output is a string
                sanitized_content = (
                    result.validated_output
                    if isinstance(result.validated_output, str)
                    else str(result.validated_output)
                )
                sanitized_messages = state["messages"][:-1] + [
                    AIMessage(content=sanitized_content)
                ]
                return {
                    "messages": sanitized_messages,
                    "guard_logs": state.get("guard_logs", []) + [guard_log],
                    "needs_refinement": False
                }

            return {
                "guard_logs": state.get("guard_logs", []) + [guard_log],
                "needs_refinement": False
            }

        except Exception as e:
            # Guard raised exception (profanity detected)
            guard_log["status"] = "blocked"
            guard_log["reason"] = str(e)

            # Trigger refinement
            current_failures = state.get("validation_failures", 0)
            if current_failures >= max_refinement_loops:
                return {
                    "messages": [AIMessage(content="⚠️ Unable to generate appropriate response.")],
                    "guard_logs": state.get("guard_logs", []) + [guard_log],
                    "needs_refinement": False
                }

            refinement_prompt = "Please rephrase your response to be more professional and appropriate."
            return {
                "messages": state["messages"] + [HumanMessage(content=refinement_prompt)],
                "guard_logs": state.get("guard_logs", []) + [guard_log],
                "validation_failures": current_failures + 1,
                "needs_refinement": True
            }

    # Routing functions
    def route_after_input_guard(state: GuardedAgentState):
        """Route based on input validation results."""
        if state.get("blocked", False):
            return END  # Block malicious/off-topic input
        return "agent"

    def should_continue_to_tools_or_output(state: GuardedAgentState):
        """Route to tools if needed, otherwise to output validation."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "output_guard"

    def route_after_output_guard(state: GuardedAgentState):
        """Route based on output validation results."""
        if state.get("needs_refinement", False):
            return "agent"  # Refine response
        return END  # Safe response, return to user

    # Build graph
    graph = StateGraph(GuardedAgentState)
    tool_node = ToolNode(tools)

    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("output_guard", output_guard_node)

    # Set entry point
    graph.set_entry_point("input_guard")

    # Add conditional edges
    graph.add_conditional_edges(
        "input_guard",
        route_after_input_guard,
        {"agent": "agent", END: END}
    )

    graph.add_conditional_edges(
        "agent",
        should_continue_to_tools_or_output,
        {"tools": "tools", "output_guard": "output_guard"}
    )

    graph.add_edge("tools", "agent")

    graph.add_conditional_edges(
        "output_guard",
        route_after_output_guard,
        {"agent": "agent", END: END}
    )

    return graph.compile()
