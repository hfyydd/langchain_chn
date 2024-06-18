# 第5部分：手动更新状态

在上一节中，我们展示了如何中断图形，以便人类可以检查它的操作。这让人类可以读取状态，但是如果他们想改变代理的路线，他们需要有写访问权限。

值得庆幸的是，LangGraph允许您手动更新状态！更新状态可以让您通过修改代理的操作（甚至修改过去！）来控制代理的轨迹。当您想要纠正代理的错误、探索替代路径或引导代理实现特定目标时，此功能特别有用。我们将在下面展示如何更新检查点状态。和以前一样，首先，定义您的图。我们将重用与以前完全相同的图。

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
    # Note: can also interrupt **after** actions, if desired.
    # interrupt_after=["tools"]
)

user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream({"messages": [("user", user_input)]}, config)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```python
snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
existing_message.pretty_print()
```

```
================================== Ai Message ==================================

[{'id': 'toolu_01DTyDpJ1kKdNps5yxv3AGJd', 'input': {'query': 'LangGraph'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01DTyDpJ1kKdNps5yxv3AGJd)
 Call ID: toolu_01DTyDpJ1kKdNps5yxv3AGJd
  Args:
    query: LangGraph
```

到目前为止，所有这些都是上一节的精确重复。LLM刚刚要求使用搜索引擎工具，我们的图表被中断了。如果我们像以前一样继续，该工具将被调用来搜索网络。

但是如果用户想调解呢？如果我们认为聊天机器人不需要使用该工具怎么办？

让我们直接提供正确的回应！

```python
from langchain_core.messages import AIMessage

answer = (
    "LangGraph is a library for building stateful, multi-actor applications with LLMs."
)
new_messages = [
    # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
    ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
    # And then directly "put words in the LLM's mouth" by populating its response.
    AIMessage(content=answer),
]

new_messages[-1].pretty_print()
graph.update_state(
    # Which state to update
    config,
    # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
    # to the existing state. We will review how to update existing messages in the next section!
    {"messages": new_messages},
)

print("\n\nLast 2 messages;")
print(graph.get_state(config).values["messages"][-2:])
```

```
================================== Ai Message ==================================

LangGraph is a library for building stateful, multi-actor applications with LLMs.


Last 2 messages;
[ToolMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', id='14589ef1-15db-4a75-82a6-d57c40a216d0', tool_call_id='toolu_01DTyDpJ1kKdNps5yxv3AGJd'), AIMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', id='1c657bfb-7690-44c7-a26d-d0d22453013d')]
```

现在图表完成了，因为我们已经提供了最终的响应消息！由于状态更新模拟了图形步骤，它们甚至生成了相应的跟踪。Inspec上面update\_state调用的LangSmith跟踪以查看发生了什么。

请注意，我们的新消息附加到已处于状态的消息中。还记得我们如何定义State类型吗？

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

我们使用预先构建的add\_messages函数注释消息。这指示图形始终将值附加到现有列表，而不是直接覆盖列表。这里应用了相同的逻辑，因此我们传递给update\_state的消息以相同的方式附加！

update\_state函数运行时就好像它是你的图中的节点之一！默认情况下，更新操作使用上次执行的节点，但你可以在下面手动指定它。让我们添加一个更新，并告诉图将其视为来自“chatbot”。

```python
graph.update_state(
    config,
    {"messages": [AIMessage(content="I'm an AI expert!")]},
    # Which node for this function to act as. It will automatically continue
    # processing as if this node just ran.
    as_node="chatbot",
)
```

```
{'configurable': {'thread_id': '1',
  'thread_ts': '2024-05-06T22:27:57.350721+00:00'}}
```

在提供的链接上查看此更新调用的LangSmith跟踪。请注意，跟踪图继续进入tools\_condition边缘。我们刚刚告诉图处理更新as\_node="聊天机器人"。如果我们按照下图从聊天机器人节点开始，我们自然会在tools\_condition边缘结束，然后\_\_end\_\_，因为我们更新的消息缺少工具调用。

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

<figure><img src="../../.gitbook/assets/下载 (3).jpeg" alt=""><figcaption></figcaption></figure>

像以前一样检查当前状态以确认检查点反映了我们的手动更新。

```python
snapshot = graph.get_state(config)
print(snapshot.values["messages"][-3:])
print(snapshot.next)
```

```
[ToolMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', id='14589ef1-15db-4a75-82a6-d57c40a216d0', tool_call_id='toolu_01DTyDpJ1kKdNps5yxv3AGJd'), AIMessage(content='LangGraph is a library for building stateful, multi-actor applications with LLMs.', id='1c657bfb-7690-44c7-a26d-d0d22453013d'), AIMessage(content="I'm an AI expert!", id='acd668e3-ba31-42c0-843c-00d0994d5885')]
()
```

注意：我们继续向状态添加AI消息。由于我们充当聊天机器人并使用不包含tool\_calls的AIMessage进行响应，因此图知道它已进入完成状态（next为空）。

如果要覆盖现有消息怎么办？

上面用于注释图形状态的add\_messages函数控制如何更新消息键。此函数查看新消息列表中的任何消息ID。如果ID与现有状态下的消息匹配，add\_messages用新内容覆盖现有消息。

例如，让我们更新工具调用以确保我们从搜索引擎中获得良好的结果！首先，启动一个新线程：

```python
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "2"}}  # we'll use thread_id = 2 here
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

I'm learning LangGraph. Could you do some research on it for me?
================================== Ai Message ==================================

[{'id': 'toolu_013MvjoDHnv476ZGzyPFZhrR', 'input': {'query': 'LangGraph'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_013MvjoDHnv476ZGzyPFZhrR)
 Call ID: toolu_013MvjoDHnv476ZGzyPFZhrR
  Args:
    query: LangGraph
```

接下来，让我们更新代理的工具调用。也许我们特别想搜索人工循环工作流。

```python
from langchain_core.messages import AIMessage

snapshot = graph.get_state(config)
existing_message = snapshot.values["messages"][-1]
print("Original")
print("Message ID", existing_message.id)
print(existing_message.tool_calls[0])
new_tool_call = existing_message.tool_calls[0].copy()
new_tool_call["args"]["query"] = "LangGraph human-in-the-loop workflow"
new_message = AIMessage(
    content=existing_message.content,
    tool_calls=[new_tool_call],
    # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
    id=existing_message.id,
)

print("Updated")
print(new_message.tool_calls[0])
print("Message ID", new_message.id)
graph.update_state(config, {"messages": [new_message]})

print("\n\nTool calls")
graph.get_state(config).values["messages"][-1].tool_calls
```

```
Original
Message ID run-59283969-1076-45fe-bee8-ebfccab163c3-0
{'name': 'tavily_search_results_json', 'args': {'query': 'LangGraph'}, 'id': 'toolu_013MvjoDHnv476ZGzyPFZhrR'}
Updated
{'name': 'tavily_search_results_json', 'args': {'query': 'LangGraph human-in-the-loop workflow'}, 'id': 'toolu_013MvjoDHnv476ZGzyPFZhrR'}
Message ID run-59283969-1076-45fe-bee8-ebfccab163c3-0


Tool calls
```

```
[{'name': 'tavily_search_results_json',
  'args': {'query': 'LangGraph human-in-the-loop workflow'},
  'id': 'toolu_013MvjoDHnv476ZGzyPFZhrR'}]
```

请注意，我们已经修改了AI的工具调用以搜索“LangGraph人机交互工作流”，而不是简单的“LangGraph”。

查看LangSmith跟踪以查看状态更新调用-您可以看到我们的新消息已成功更新先前的AI消息。

通过输入None和现有配置的流来恢复图形。

```python
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

\================================= Tool Message ================================= Name: tavily\_search\_results\_json

\[{"url": "https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/", "content": "Human-in-the-loop\u00b6 When creating LangGraph agents, it is often nice to add a human in the loop component. This can be helpful when giving them access to tools. ... from langgraph.graph import MessageGraph, END # Define a new graph workflow = MessageGraph # Define the two nodes we will cycle between workflow. add\_node ("agent", call\_model) ..."}, {"url": "https://langchain-ai.github.io/langgraph/how-tos/chat\_agent\_executor\_with\_function\_calling/human-in-the-loop/", "content": "Human-in-the-loop. In this example we will build a ReAct Agent that has a human in the loop. We will use the human to approve specific actions. This examples builds off the base chat executor. It is highly recommended you learn about that executor before going through this notebook. You can find documentation for that example here."}] ================================== Ai Message ==================================

Based on the search results, LangGraph appears to be a framework for building AI agents that can interact with humans in a conversational way. The key points I gathered are:

* LangGraph allows for "human-in-the-loop" workflows, where a human can be involved in approving or reviewing actions taken by the AI agent.
* This can be useful for giving the AI agent access to various tools and capabilities, with the human able to provide oversight and guidance.
* The framework includes components like "MessageGraph" for defining the conversational flow between the agent and human.

Overall, LangGraph seems to be a way to create conversational AI agents that can leverage human input and guidance, rather than operating in a fully autonomous way. Let me know if you need any clarification or have additional questions!

查看跟踪以查看工具调用和稍后的LLM响应。请注意，现在图表使用我们更新的查询词查询搜索引擎-我们能够在此处手动覆盖LLM的搜索！

所有这些都反映在图的检查点内存中，这意味着如果我们继续对话，它会调用所有修改后的状态。

```python
events = graph.stream(
    {
        "messages": (
            "user",
            "Remember what I'm learning about?",
        )
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember what I'm learning about?
================================== Ai Message ==================================

Ah yes, now I remember - you mentioned earlier that you are learning about LangGraph.

LangGraph is the framework I researched in my previous response, which is for building conversational AI agents that can incorporate human input and oversight.

So based on our earlier discussion, it seems you are currently learning about and exploring the LangGraph system for creating human-in-the-loop AI agents. Please let me know if I have the right understanding now.
```

恭喜！作为人机交互工作流的一部分，您已经使用interrupt\_before和update\_state手动修改状态。中断和状态修改允许您控制代理的行为方式。结合持久检查点，这意味着您可以暂停操作并在任何时候恢复。当图表中断时，您的用户不必可用！

本节的图形代码与前面的代码相同。要记住的关键片段是添加. compile（…，interrupt\_before=\[…]）（或interrupt\_after），如果您想在图形到达节点时显式暂停图形。然后您可以使用update\_state来修改检查点并控制图形应如何进行。
