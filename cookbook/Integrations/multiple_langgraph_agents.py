import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-xxx"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-xxx"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

# Your openai key
os.environ["OPENAI_API_KEY"] = "xxx"
# 使用的是智谱api，可替换
os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4"


# Create Multiple LangGraph Agents

from langfuse import get_client, Langfuse
from langfuse.langchain import CallbackHandler

langfuse = get_client()

# Generate deterministic trace ID from external system
predefined_trace_id = Langfuse.create_trace_id()

# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

# 每一个step 按顺序可独立运行
# Step 1 sub-agent

from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOpenAI(model="GLM-4.5-Flash", temperature=0.2)


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
sub_agent = graph_builder.compile()


# Step 2 Creat Tool
# Set the tool that uses the research-sub-agent to answer questions.

from langchain_core.tools import tool


@tool
def langgraph_research(question):
    """Conducts research for various topics."""

    with langfuse.start_as_current_span(
        name="🤖-sub-research-agent", trace_context={"trace_id": predefined_trace_id}
    ) as span:
        span.update_trace(input=question)

        response = sub_agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "callbacks": [langfuse_handler],
            },
        )

        span.update_trace(output=response["messages"][1].content)
    return response["messages"][1].content


# Set up a second simple LangGraph agent that uses the new langgraph_research.
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="GLM-4.5-Flash", temperature=0.2)

main_agent = create_react_agent(
    model=llm, tools=[langgraph_research], name="Multiple LangGraph Agents"
)


# Step 3 Start

user_question = "What are LLM Observations Traces and Scores?"

# Use the predefined trace ID with trace_context
with langfuse.start_as_current_span(
    name="🤖-main-agent", trace_context={"trace_id": predefined_trace_id}
) as span:
    span.update_trace(input=user_question)

    # LangChain execution will be part of this trace
    response = main_agent.invoke(
        {"messages": [{"role": "user", "content": user_question}]},
        config={"callbacks": [langfuse_handler]},
    )

    span.update_trace(output=response["messages"][1].content)

print(f"Trace ID: {predefined_trace_id}")  # Use this for scoring later
