"""
LangGraph recommendation copilot.

A ReAct agent that:
  1. Looks up customer profile
  2. Runs recommendation + uplift scoring
  3. Generates personalised RM pitch via RAG
  4. Human-in-the-loop: RM approves or rejects

The agent decides which tools to call based on the RM's query.
"""

import logging
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.agents.tools import customer_lookup, recommend_and_score, generate_pitch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOOLS = [customer_lookup, recommend_and_score, generate_pitch]

SYSTEM_PROMPT = """You are a recommendation copilot for bank relationship managers (RMs).

Your job is to help RMs decide which products to recommend to customers,
using data-driven uplift modelling to target the right customers.

You have access to these tools:
1. customer_lookup: Get a customer's profile, demographics, and product holdings
2. recommend_and_score: Get top product recommendations with uplift scores and segment
3. generate_pitch: Generate a personalised talking-point pitch for a specific product

IMPORTANT RULES:
- Always check the customer's uplift segment before recommending
- If the customer is a "Sleeping Dog", warn the RM that outreach may hurt
- If the customer is a "Sure Thing", suggest the RM save time for persuadables
- Only recommend products the customer doesn't already hold
- Be concise and actionable — RMs are busy
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_copilot(llm):
    """
    Create the LangGraph recommendation copilot.

    Args:
        llm: A ChatModel instance (e.g., ChatGroq, ChatOpenAI)
    """
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def agent_node(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_copilot(copilot, query: str) -> str:
    """
    Run the copilot with an RM query and return the response.
    """
    result = copilot.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # Get the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "No response generated."
