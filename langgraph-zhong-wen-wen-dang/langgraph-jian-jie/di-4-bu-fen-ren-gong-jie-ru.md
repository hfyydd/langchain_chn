# 第4部分：人工介入

代理可能不可靠，可能需要人工输入才能成功完成任务。同样，对于某些操作，您可能希望在运行之前要求人工批准，以确保一切按预期运行。

LangGraph以多种方式支持人机循环human-in-the-loop工作流。在本节中，我们将使用LangGraph的interrupt\_before功能来始终中断工具节点。

首先，从我们现有的代码开始。以下是从第3部分复制的。

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

memory = SqliteSaver.from_conn_string(":memory:")


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
```

现在，编译图形，指定interrupt\_before操作节点。

```python
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ actions, if desired.
    # interrupt_after=["tools"]
)
```

```python
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

\================================ Human Message =================================

I'm learning LangGraph. Could you do some research on it for me? ================================== Ai Message ==================================

\[{'text': "Okay, let's do some research on LangGraph:", 'type': 'text'}, {'id': 'toolu\_01Be7aRgMEv9cg6ezaFjiCry', 'input': {'query': 'LangGraph'}, 'name': 'tavily\_search\_results\_json', 'type': 'tool\_use'}] Tool Calls: tavily\_search\_results\_json (toolu\_01Be7aRgMEv9cg6ezaFjiCry) Call ID: toolu\_01Be7aRgMEv9cg6ezaFjiCry Args: query: LangGraph

让我们检查图形状态以确认它有效。

```python
snapshot = graph.get_state(config)
snapshot.next
```

```
('action',)
```

请注意，与上次不同，“next”节点设置为“action”。我们在这里中断了！让我们检查一下工具调用。

```python
existing_message = snapshot.values["messages"][-1]
existing_message.tool_calls
```

```
[{'name': 'tavily_search_results_json',
  'args': {'query': 'LangGraph'},
  'id': 'toolu_01Be7aRgMEv9cg6ezaFjiCry'}]
```

这个查询似乎很合理。这里没有什么要过滤的。人类能做的最简单的事情就是让图形继续执行。让我们在下面这样做。

接下来，继续图形！传入None只会让图形继续它离开的地方，而不会向状态添加任何新内容。

```python
# `None` will append nothing new to the current state, letting it resume as if it had never been interrupted
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://github.com/langchain-ai/langgraph", "content": "LangGraph is a Python package that extends LangChain Expression Language with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam and can be used for agent-like behaviors, such as chatbots, with LLMs."}, {"url": "https://langchain-ai.github.io/langgraph//", "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain . It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam ."}]
================================== Ai Message ==================================

Based on the search results, LangGraph seems to be a Python library that extends the LangChain library to enable more complex, multi-step interactions with large language models (LLMs). Some key points:

- LangGraph allows coordinating multiple "chains" (or actors) over multiple steps of computation, in a cyclic manner. This enables more advanced agent-like behaviors like chatbots.
- It is inspired by distributed graph processing frameworks like Pregel and Apache Beam.
- LangGraph is built on top of the LangChain library, which provides a framework for building applications with LLMs.

So in summary, LangGraph appears to be a powerful tool for building more sophisticated applications and agents using large language models, by allowing you to coordinate multiple steps and actors in a flexible, graph-like manner. It extends the capabilities of the base LangChain library.

Let me know if you need any clarification or have additional questions!
```

查看此调用的LangSmith跟踪以查看上述调用中完成的确切工作。请注意，状态是在第一步加载的，以便您的聊天机器人可以从中断的地方继续。

恭喜！您使用了中断interrupt来为您的聊天机器人添加人工在循环执行，允许在需要时进行人工监督和干预。这打开了您可以使用AI系统创建的潜在UI。由于我们已经添加了检查指针，因此图表可以无限期暂停并随时恢复，就像什么都没发生一样。

接下来，我们将探索如何使用自定义状态更新进一步自定义机器人的行为。

下面是您在本节中使用的代码的副本。这与前面部分的唯一区别是添加了interrupt\_before参数。

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(
    checkpointer=memory,
    # This is new!
    interrupt_before=["tools"],
    # Note: can also interrupt __after__ actions, if desired.
    # interrupt_after=["tools"]
)
```
