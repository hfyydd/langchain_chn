# 第3部分：向聊天机器人添加内存

我们的聊天机器人现在可以使用工具来回答用户问题，但它不记得以前交互的上下文。这限制了它进行连贯、多轮对话的能力。

LangGraph通过持久检查点**checkpointing**解决了这个问题。如果您在编译图时提供检查指针`checkpointer` ，在调用图时提供thread\_id，LangGraph会在每一步后自动保存状态。当您使用相同的thread\_id再次调用图时，图会加载其保存的状态，允许聊天机器人从中断的地方继续。

稍后我们将看到检查点比简单的聊天记忆强大得多——它可以让你随时保存和恢复复杂的状态，用于错误恢复、人在循环工作流程、时间旅行交互等等。但是在我们太超前之前，让我们添加检查点来实现多轮对话。

首先，创建一个SqliteSaver检查指针。

```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
```

请注意，我们已经将：内存指定为Sqlite DB路径。这对我们的教程很方便（它将其全部保存在内存中）。在生产应用程序中，您可能会更改它以连接到您自己的数据库和/或使用其他检查指针类之一。

接下来定义图。现在您已经构建了自己的BasicToolNode，我们将用LangGraph的预构建ToolNode和tools\_condition替换它，因为它们可以做一些很好的事情，比如并行API执行。除此之外，以下内容都复制自第2部分。

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

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
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
```

```
/Users/wfh/code/lc/langchain/libs/core/langchain_core/_api/beta_decorator.py:87: LangChainBetaWarning: The method `ChatAnthropic.bind_tools` is in beta. It is actively being worked on, so the API may change.
  warn_beta(
```

最后，使用提供的检查指针编译图形。

```python
graph = graph_builder.compile(checkpointer=memory)
```

请注意，自第2部分以来，图的连通性没有改变。当图通过每个节点工作时，我们所做的只是检查点状态。

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

<figure><img src="../../.gitbook/assets/下载 (2).jpeg" alt=""><figcaption></figcaption></figure>

现在您可以与您的机器人交互了！首先，选择一个线程作为此对话的key。

```python
config = {"configurable": {"thread_id": "1"}}
```

接下来，调用您的聊天机器人。

```python
user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Hi there! My name is Will.
================================== Ai Message ==================================

It's nice to meet you, Will! I'm an AI assistant created by Anthropic. I'm here to help you with any questions or tasks you may have. Please let me know how I can assist you today.
```

注意：调用我们的图形时，配置是作为第二个位置参数提供的。重要的是，它没有嵌套在图形输入中（{'messages': \[]}). 让我们问一个后续问题：看看它是否记得你的名字。

让我们问一个后续问题：看看它是否记得你的名字。

```python
user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

Of course, your name is Will. It's nice to meet you again!
```

请注意，我们不是使用外部列表的内存：它都由检查指针处理！您可以在此LangSmith跟踪中检查完整执行以查看发生了什么。

不相信我？尝试使用不同的配置。

```python
# The only difference is we change the `thread_id` here to "2" instead of "1"
events = graph.stream(
    {"messages": [("user", user_input)]},
    {"configurable": {"thread_id": "2"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

```
================================ Human Message =================================

Remember my name?
================================== Ai Message ==================================

I'm afraid I don't actually have the capability to remember your name. As an AI assistant, I don't have a persistent memory of our previous conversations or interactions. I respond based on the current context provided to me. Could you please restate your name or provide more information so I can try to assist you?
```

请注意，我们所做的唯一更改是修改配置中的thread\_id。现在请参阅此调用的LangSmith跟踪比较.

到现在为止，我们在两个不同的线程中创建了一些检查点。但是检查点有什么？要随时检查给定配置的图形状态，请调用get\_state（config）。

```python
snapshot = graph.get_state(config)
snapshot
```

```
StateSnapshot(values={'messages': [HumanMessage(content='Hi there! My name is Will.', id='aad97d7f-8845-4f9e-b723-2af3b7c97590'), AIMessage(content="It's nice to meet you, Will! I'm an AI assistant created by Anthropic. I'm here to help you with any questions or tasks you may have. Please let me know how I can assist you today.", response_metadata={'id': 'msg_01VCz7Y5jVmMZXibBtnECyvJ', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 375, 'output_tokens': 49}}, id='run-66cf1695-5ba8-4fd8-a79d-ded9ee3c3b33-0'), HumanMessage(content='Remember my name?', id='ac1e9971-dbee-4622-9e63-5015dee05c20'), AIMessage(content="Of course, your name is Will. It's nice to meet you again!", response_metadata={'id': 'msg_01RsJ6GaQth7r9soxbF7TSpQ', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 431, 'output_tokens': 19}}, id='run-890149d3-214f-44e8-9717-57ec4ef68224-0')]}, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '2024-05-06T22:23:20.430350+00:00'}}, parent_config=None)
```

```python
snapshot.next  # (since the graph ended this turn, `next` is empty. If you fetch a state from within a graph invocation, next tells which node will execute next)
```

```
()
```

上面的快照包含当前状态值、相应的配置和下一个要处理的节点。在我们的例子中，图已经达到\_\_end\_\_状态，所以next是空的。

恭喜！由于LangGraph的检查点系统，您的聊天机器人现在可以跨会话保持对话状态。这为更自然的上下文交互开辟了令人兴奋的可能性。LangGraph的检查点甚至可以处理任意复杂的图形状态，这比简单的聊天记忆更具表现力和强大。

在下一部分中，我们将向我们的机器人介绍人工监督，以处理在进行之前可能需要指导或验证的情况。

查看下面的代码片段，查看本节中的图表。

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
graph = graph_builder.compile(checkpointer=memory)
```
