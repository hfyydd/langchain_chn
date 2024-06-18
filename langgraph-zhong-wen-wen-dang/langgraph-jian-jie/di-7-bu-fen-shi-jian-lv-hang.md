# 第7部分：时间旅行

在典型的聊天机器人工作流程中，用户与机器人交互1次或多次以完成任务。在前面的部分中，我们看到了如何添加内存和人工循环，以便能够检查我们的图形状态并手动覆盖状态以控制未来的响应。

但是，如果你想让你的用户从之前的响应开始，然后“分支”探索一个单独的结果呢？或者，如果你想让用户能够“倒带”你助手的工作来修复一些错误或尝试不同的策略（在自主软件工程师等应用程序中很常见），该怎么办？

您可以使用LangGraph的内置“时间旅行”功能创建这两种体验以及更多体验。

在本节中，您将通过使用图形的get\_state\_history方法获取检查点来“倒带”图形。然后，您可以在前一个时间点恢复执行。

首先，回想一下我们的聊天机器人图。我们不需要对以前进行任何更改：

```python
from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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


def select_next_node(state: State) -> Literal["human", "tools", "__end__"]:
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

```python
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

<figure><img src="../../.gitbook/assets/下载 (2) (1).jpeg" alt=""><figcaption></figcaption></figure>

让我们的图表采取几个步骤。每一步都将在其状态历史记录中被检查点：

```python
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            ("user", "I'm learning LangGraph. Could you do some research on it for me?")
        ]
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

I'm learning LangGraph. Could you do some research on it for me?
================================== Ai Message ==================================

[{'text': "Okay, let me look into LangGraph for you. Here's what I found:", 'type': 'text'}, {'id': 'toolu_011AQ2FT4RupVka2LVMV3Gci', 'input': {'query': 'LangGraph'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_011AQ2FT4RupVka2LVMV3Gci)
 Call ID: toolu_011AQ2FT4RupVka2LVMV3Gci
  Args:
    query: LangGraph
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://langchain-ai.github.io/langgraph/", "content": "LangGraph is framework agnostic (each node is a regular python function). It extends the core Runnable API (shared interface for streaming, async, and batch calls) to make it easy to: Seamless state management across multiple turns of conversation or tool usage. The ability to flexibly route between nodes based on dynamic criteria."}, {"url": "https://blog.langchain.dev/langgraph-multi-agent-workflows/", "content": "As a part of the launch, we highlighted two simple runtimes: one that is the equivalent of the AgentExecutor in langchain, and a second that was a version of that aimed at message passing and chat models.\n It's important to note that these three examples are only a few of the possible examples we could highlight - there are almost assuredly other examples out there and we look forward to seeing what the community comes up with!\n LangGraph: Multi-Agent Workflows\nLinks\nLast week we highlighted LangGraph - a new package (available in both Python and JS) to better enable creation of LLM workflows containing cycles, which are a critical component of most agent runtimes. \"\nAnother key difference between Autogen and LangGraph is that LangGraph is fully integrated into the LangChain ecosystem, meaning you take fully advantage of all the LangChain integrations and LangSmith observability.\n As part of this launch, we're also excited to highlight a few applications built on top of LangGraph that utilize the concept of multiple agents.\n"}]
================================== Ai Message ==================================

Based on the search results, here's what I've learned about LangGraph:

- LangGraph is a framework-agnostic tool that extends the Runnable API to make it easier to manage state and routing between different nodes or agents in a conversational workflow. 

- It's part of the LangChain ecosystem, so it integrates with other LangChain tools and observability features.

- LangGraph enables the creation of multi-agent workflows, where you can have different "nodes" or agents that can communicate and pass information to each other.

- This allows for more complex conversational flows and the ability to chain together different capabilities, tools, or models.

- The key benefits seem to be around state management, flexible routing between agents, and the ability to create more sophisticated and dynamic conversational workflows.

Let me know if you need any clarification or have additional questions! I'm happy to do more research on LangGraph if you need further details.
```

```python
events = graph.stream(
    {
        "messages": [
            ("user", "Ya that's helpful. Maybe I'll build an autonomous agent with it!")
        ]
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

Ya that's helpful. Maybe I'll build an autonomous agent with it!
================================== Ai Message ==================================

[{'text': "That's great that you're interested in building an autonomous agent using LangGraph! Here are a few additional thoughts on how you could approach that:", 'type': 'text'}, {'id': 'toolu_01L3V9FhZG5Qx9jqRGfWGtS2', 'input': {'query': 'building autonomous agents with langgraph'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_01L3V9FhZG5Qx9jqRGfWGtS2)
 Call ID: toolu_01L3V9FhZG5Qx9jqRGfWGtS2
  Args:
    query: building autonomous agents with langgraph
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://github.com/langchain-ai/langgraphjs", "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain.js.It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam.The current interface exposed is one inspired by ..."}, {"url": "https://github.com/langchain-ai/langgraph", "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam.The current interface exposed is one inspired by NetworkX.. The main use is for adding cycles to your LLM ..."}]
================================== Ai Message ==================================

The key things to keep in mind:

1. LangGraph is designed to help coordinate multiple "agents" or "actors" that can pass information back and forth. This allows you to build more complex, multi-step workflows.

2. You'll likely want to define different nodes or agents that handle specific tasks or capabilities. LangGraph makes it easy to route between these agents based on the state of the conversation.

3. Make sure to leverage the LangChain ecosystem - things like prompts, memory, agents, tools etc. LangGraph integrates with these to give you a powerful set of building blocks.

4. Pay close attention to state management - LangGraph helps you manage state across multiple interactions, which is crucial for an autonomous agent.

5. Consider how you'll handle things like user intent, context, and goal-driven behavior. LangGraph gives you the flexibility to implement these kinds of complex behaviors.

Let me know if you have any other specific questions as you start prototyping your autonomous agent! I'm happy to provide more guidance.
```

现在我们已经让代理采取了几个步骤，我们可以重播完整的状态历史以查看发生的一切。

```python
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
```

```
um Messages:  8 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  7 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  6 Next:  ('action',)
--------------------------------------------------------------------------------
Num Messages:  5 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  4 Next:  ()
--------------------------------------------------------------------------------
Num Messages:  3 Next:  ('chatbot',)
--------------------------------------------------------------------------------
Num Messages:  2 Next:  ('action',)
--------------------------------------------------------------------------------
Num Messages:  1 Next:  ('chatbot',)
--------------------------------------------------------------------------------
```

请注意，图表的每一步都保存了检查点。这\_spansinvocations\_\_，因此您可以倒带完整线程的历史记录。我们已经选择了to\_replay作为要恢复的状态。这是上面第二个图表调用中聊天机器人节点之后的状态。

从这一点恢复应该接下来调用操作节点。

```python
print(to_replay.next)
print(to_replay.config)
```

```
('action',)
{'configurable': {'thread_id': '1', 'thread_ts': '2024-05-06T22:33:10.211424+00:00'}}
```

请注意，检查点的配置（to\_replay. config）包含thread\_ts时间戳。提供此thread\_ts值会告诉LangGraph的检查指针从该时刻开始加载状态。让我们在下面尝试一下：

```python
# The `thread_ts` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

```
================================= Tool Message =================================
Name: tavily_search_results_json

[{"url": "https://valentinaalto.medium.com/getting-started-with-langgraph-66388e023754", "content": "Sign up\nSign in\nSign up\nSign in\nMember-only story\nGetting Started with LangGraph\nBuilding multi-agents application with graph frameworks\nValentina Alto\nFollow\n--\nShare\nOver the last year, LangChain has established itself as one of the most popular AI framework available in the market. This new library, introduced in January\u2026\n--\n--\nWritten by Valentina Alto\nData&AI Specialist at @Microsoft | MSc in Data Science | AI, Machine Learning and Running enthusiast\nHelp\nStatus\nAbout\nCareers\nBlog\nPrivacy\nTerms\nText to speech\nTeams Since the concept of multi-agent applications \u2014 the ones exhibiting different agents, each having a specific personality and tools to access \u2014 is getting real and mainstream (see the rise of libraries projects like AutoGen), LangChain\u2019s developers introduced a new library to make it easier to manage these kind of agentic applications. Nevertheless, those chains were lacking the capability of introducing cycles into their runtime, meaning that there is no out-of-the-box framework to enable the LLM to reason over the next best action in a kind of for-loop scenario. The main feature of LangChain \u2014 as the name suggests \u2014 is its ability to easily create the so-called chains."}, {"url": "https://blog.langchain.dev/langgraph-multi-agent-workflows/", "content": "As a part of the launch, we highlighted two simple runtimes: one that is the equivalent of the AgentExecutor in langchain, and a second that was a version of that aimed at message passing and chat models.\n It's important to note that these three examples are only a few of the possible examples we could highlight - there are almost assuredly other examples out there and we look forward to seeing what the community comes up with!\n LangGraph: Multi-Agent Workflows\nLinks\nLast week we highlighted LangGraph - a new package (available in both Python and JS) to better enable creation of LLM workflows containing cycles, which are a critical component of most agent runtimes. \"\nAnother key difference between Autogen and LangGraph is that LangGraph is fully integrated into the LangChain ecosystem, meaning you take fully advantage of all the LangChain integrations and LangSmith observability.\n As part of this launch, we're also excited to highlight a few applications built on top of LangGraph that utilize the concept of multiple agents.\n"}]
================================== Ai Message ==================================

The key things I gathered are:

- LangGraph is well-suited for building multi-agent applications, where you have different agents with their own capabilities, tools, and personality.

- It allows you to create more complex workflows with cycles and feedback loops, which is critical for building autonomous agents that can reason about their next best actions.

- The integration with LangChain means you can leverage other useful features like state management, observability, and integrations with various language models and data sources.

Some tips for building an autonomous agent with LangGraph:

1. Define the different agents/nodes in your workflow and their specific responsibilities/capabilities.
2. Set up the connections and routing between the agents so they can pass information and decisions back and forth.
3. Implement logic within each agent to assess the current state and determine the optimal next action.
4. Use LangChain features like memory and toolkits to give your agents access to relevant information and abilities.
5. Monitor the overall system behavior and iteratively improve the agent interactions and decision-making.

Let me know if you have any other questions! I'm happy to provide more guidance as you start building your autonomous agent with LangGraph.
```

请注意，该图从**action**节点恢复执行。您可以看出是这种情况，因为上面打印的第一个值是我们搜索引擎工具的响应。

恭喜！您现在已经在LangGraph中使用了时间旅行检查点遍历。能够倒带和探索替代路径为调试、实验和交互式应用程序打开了一个充满可能性的世界。
