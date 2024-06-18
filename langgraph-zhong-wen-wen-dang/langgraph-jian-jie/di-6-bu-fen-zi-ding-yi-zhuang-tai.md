# 第6部分：自定义状态

到目前为止，我们依赖于一个简单的状态（它只是一个消息列表！）。你可以用这个简单的状态走得更远，但是如果你想在不依赖消息列表的情况下定义复杂的行为，你可以在状态中添加额外的字段。在本节中，我们将使用一个新节点来扩展我们的聊天机器人来说明这一点。

在上面的示例中，我们确定性地涉及了一个人：每当调用工具时，图表总是中断。假设我们希望我们的聊天机器人可以选择依赖人类。

一种方法是创建一个直通的“人类”节点，在此之前，图形将始终停止。只有当LLM调用“人类”工具时，我们才会执行这个节点。为了方便起见，我们将在图形状态中包含一个“ask\_human”标志，如果LLM调用这个工具，我们将翻转该标志。

下面，用更新的State定义这个新图表

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
    # This flag is new
    ask_human: bool
```

接下来，定义一个模式来显示模型，让它决定请求帮助。

```python
from langchain_core.pydantic_v1 import BaseModel


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str
```

接下来，定义聊天机器人节点。这里的主要修改是如果我们看到聊天机器人已经调用了Request辅助标志，则翻转ask\_human标志。

```python
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
# We can bind the llm to a tool definition, a pydantic model, or a json schema
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}
```

接下来，创建图形构建器并将聊天机器人和工具节点添加到图形中，与之前相同。

```python
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))
```

接下来，创建“human”节点。这个节点函数主要是我们图表中的占位符，它将触发中断。如果人类在中断interrupt期间没有手动更新状态，它会插入一条工具消息，这样LLM就知道用户被请求了，但没有响应。这个节点还取消了ask\_human标志，这样图表就知道除非提出进一步的请求，否则不会重新访问该节点。

```python
from langchain_core.messages import AIMessage, ToolMessage


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)
```

接下来，定义条件逻辑。如果设置了标志，select\_next\_node将路由到人工节点。否则，它让预构建的tools\_condition函数选择下一个节点。

回想一下，tools\_condition函数只是检查聊天机器人是否在其响应消息中tool\_calls。如果是，它将路由到操作节点。否则，它将结束图表。

```python
def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)
```

最后，添加简单的有向边并编译图。每当a完成执行时，这些边指示图始终从节点a->b流出。

```python
# The rest is the same
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(
    checkpointer=memory,
    # We interrupt before 'human' here instead.
    interrupt_before=["human"],
)
```

如果您安装了可视化依赖项，您可以看到下面的图形结构：

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

<figure><img src="../../.gitbook/assets/下载 (1) (1).jpeg" alt=""><figcaption></figcaption></figure>

聊天机器人可以向人类请求帮助（chatbot->select->human），调用搜索引擎工具（chatbot->select->action），或者直接响应（chatbot->select->**end**）。一旦做出动作或请求，图表将转换回聊天机器人节点以继续操作。

让我们看看这个图表。我们将请求专家协助来说明我们的图表。

```python
user_input = "I need some expert guidance for building this AI agent. Could you request assistance for me?"
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

I need some expert guidance for building this AI agent. Could you request assistance for me? ================================== Ai Message ==================================

\[{'id': 'toolu\_017XaQuVsoAyfXeTfDyv55Pc', 'input': {'request': 'I need some expert guidance for building this AI agent.'}, 'name': 'RequestAssistance', 'type': 'tool\_use'}] Tool Calls: RequestAssistance (toolu\_017XaQuVsoAyfXeTfDyv55Pc) Call ID: toolu\_017XaQuVsoAyfXeTfDyv55Pc Args: request: I need some expert guidance for building this AI agent.

注意：LLM已经调用了我们提供给它的“RequestAssistant”工具，并且中断已经设置。让我们检查图形状态以确认。

```python
snapshot = graph.get_state(config)
snapshot.next
```

```
('human',)
```

图形状态确实在“human”节点之前被中断。在这种情况下，我们可以充当“专家”，并通过使用我们的输入添加新的ToolMessage来手动更新状态。

接下来，通过以下方式响应聊天机器人的请求：

1. 使用我们的响应创建ToolMessage。这将被传递回聊天机器人。
2. 调用update\_state手动更新图形状态。

```python
ai_message = snapshot.values["messages"][-1]
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)
tool_message = create_response(human_response, ai_message)
graph.update_state(config, {"messages": [tool_message]})
```

```
{'configurable': {'thread_id': '1',
  'thread_ts': '2024-05-06T22:31:39.973392+00:00'}}
```

您可以检查状态以确认我们的响应已添加。

```python
graph.get_state(config).values["messages"]
```

```
[HumanMessage(content='I need some expert guidance for building this AI agent. Could you request assistance for me?', id='ab75eb9d-cce7-4e44-8de7-b0b375a86972'),
 AIMessage(content=[{'id': 'toolu_017XaQuVsoAyfXeTfDyv55Pc', 'input': {'request': 'I need some expert guidance for building this AI agent.'}, 'name': 'RequestAssistance', 'type': 'tool_use'}], response_metadata={'id': 'msg_0199PiK6kmVAbeo1qmephKDq', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'tool_use', 'stop_sequence': None, 'usage': {'input_tokens': 486, 'output_tokens': 63}}, id='run-ff07f108-5055-4343-8910-2fa40ead3fb9-0', tool_calls=[{'name': 'RequestAssistance', 'args': {'request': 'I need some expert guidance for building this AI agent.'}, 'id': 'toolu_017XaQuVsoAyfXeTfDyv55Pc'}]),
 ToolMessage(content="We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.", id='19f2eb9f-a742-46aa-9047-60909c30e64a', tool_call_id='toolu_017XaQuVsoAyfXeTfDyv55Pc')]
```

接下来，通过调用它来恢复图形，并将None作为输入。

```python
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================= Tool Message =================================

We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.
================================== Ai Message ==================================

It looks like the experts have provided some guidance on how to build your AI agent. They suggested checking out LangGraph, which they say is more reliable and extensible than simple autonomous agents. Please let me know if you need any other assistance - I'm happy to help coordinate with the expert team further.
```

请注意，聊天机器人已将更新的状态合并到其最终响应中。由于所有内容都经过检查点，因此循环中的“专家”可以随时执行更新，而不会影响图形的执行。

恭喜你！你现在已经在你的辅助图中添加了一个额外的节点，让聊天机器人自己决定是否需要中断执行。您是通过用一个新的ask\_human字段更新图状态并在编译图时修改中断逻辑来做到这一点的。这使您可以在每次执行图时动态地将一个人包括在循环中，同时保持完整的内存。

我们几乎完成了本教程，但在完成之前，我们还想回顾一下连接检查点和状态更新的另一个概念。本节的代码转载如下供您参考。

```python
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-haiku-20240307")
# We can bind the llm to a tool definition, a pydantic model, or a json schema
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = SqliteSaver.from_conn_string(":memory:")
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
)
```
