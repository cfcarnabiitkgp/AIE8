"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")

    return graph.compile()


def create_langgraph_helpful_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    helpfulness_model: str = "gpt-4.1-mini",
    max_iterations: int = 10
):
    """Create a LangGraph agent with helpfulness evaluation.

    This agent includes a self-reflection loop that evaluates whether responses
    are helpful and iteratively refines them until they meet quality standards.

    Args:
        model_name: OpenAI model name for the main agent
        temperature: Model temperature for the main agent
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        helpfulness_model: Model to use for helpfulness evaluation
        max_iterations: Maximum number of refinement iterations

    Returns:
        Compiled LangGraph agent with helpfulness checking
    """
    if tools is None:
        tools = get_default_tools(rag_chain)

    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_to_action_or_helpfulness(state: AgentState):
        """Route to tools if the last message has tool calls, otherwise to helpfulness check."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"

    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Evaluate helpfulness of the latest response relative to the initial query."""
        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > max_iterations:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        prompt_template = """Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

Initial Query:
{initial_query}

Final Response:
{final_response}"""

        helpfulness_prompt = PromptTemplate.from_template(prompt_template)
        helpfulness_model_instance = get_openai_model(model_name=helpfulness_model, temperature=0.1)
        helpfulness_chain = helpfulness_prompt | helpfulness_model_instance | StrOutputParser()

        helpfulness_response = helpfulness_chain.invoke({
            "initial_query": initial_query.content,
            "final_response": final_response.content,
        })

        decision = "Y" if "Y" in helpfulness_response else "N"
        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
        # Check loop-limit marker
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
            return END

        last = state["messages"][-1]
        text = getattr(last, "content", "")
        if "HELPFULNESS:Y" in text:
            return "end"
        return "continue"

    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.set_entry_point("agent")

    # From agent: go to action if tool calls, otherwise to helpfulness check
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"}
    )

    # From helpfulness: either end (helpful) or continue (not helpful - loop back to agent)
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END}
    )

    # From action (tools): always go back to agent
    graph.add_edge("action", "agent")

    return graph.compile()

