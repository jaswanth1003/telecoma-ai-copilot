from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from langchain_core.messages import messages_from_dict, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from tools import get_site_kpi_extreme, get_peak_kpi_day_for_site, compare_kpi_impact, describe_kpi_dataset, kpi_anomalies
import os
from dotenv import load_dotenv
load_dotenv()

nvidia_key = os.getenv("NVIDIA_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")


# llm = ChatNVIDIA(model="nv-mistralai/mistral-nemo-12b-instruct", temperature=0.3, streaming=True)

# llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct", streaming=True)

llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", streaming=True, api_key=nvidia_key)

search_tool = TavilySearch(max_results=3, tavily_api_key=tavily_key)

all_tools = [search_tool, get_site_kpi_extreme, get_peak_kpi_day_for_site, compare_kpi_impact, describe_kpi_dataset, kpi_anomalies]
llm_with_tools = llm.bind_tools(all_tools)

tool_node = ToolNode(all_tools)

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "__end__"

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()


agent_executor: Runnable = RunnableLambda(lambda x: app.invoke({
    "messages": [
        SystemMessage(content=(
            "You are a helpful AI assistant specialized in telecom network analytics.\n"
            "You have access to KPI data like SINR, throughput, drop rate, CPU utilization, etc.\n"
            "You may use search tool or internal analytics tools to assist the user.\n"
            "Explain clearly and professionally, and offer insights and summarize findings with detailed explanations."
            "Always explain your reasoning clearly.\n"
            "If uncertain, say 'I’m not sure' or suggest further data analysis.\n"
            "Avoid making speculative conclusions."
        ))
    ] + messages_from_dict(x.get("chat_history", [])) + [HumanMessage(content=x["input"])]
}))

if __name__ == "__main__":
    while True:
        user_input = input("Ask the Telecom KPI Analysis Agent: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": []
        })

        for msg in result["messages"]:
            if hasattr(msg, 'content'):
                print(msg.content)