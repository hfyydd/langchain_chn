# 信息收集

## 提示生成器

在这个例子中，我们将创建一个聊天机器人，帮助用户生成提示。它将首先收集用户的需求，然后生成提示（并根据用户输入细化它）。这些被分成两个独立的状态，LLM决定何时在它们之间转换。

该系统的图形表示可以在下面找到。

### 收集信息

首先，让我们定义图中收集用户需求的部分。这将是一个带有特定系统消息的LLM调用。它将有权访问一个工具，当它准备好生成提示时可以调用该工具。

```python
from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
```

```python
template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]


llm = ChatOpenAI(temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])

chain = get_messages_info | llm_with_tool
```

### 生成提示

我们现在设置将生成提示的状态。这将需要一个单独的系统消息，以及一个过滤掉工具调用之前的所有消息的函数（因为这是前一个状态决定是时候生成提示了

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# New system prompt
prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""


# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


prompt_gen_chain = get_prompt_messages | llm
```

### 定义状态逻辑

这是聊天机器人处于什么状态的逻辑。如果最后一条消息是工具调用，那么我们处于“提示创建者”（提示）应该响应的状态。否则，如果最后一条消息不是人类消息，那么我们知道人类接下来应该响应，所以我们处于结束状态。如果最后一条消息是人类消息，那么如果之前有工具调用，我们处于提示状态。否则，我们处于“信息收集”（info）状态。

```python
from typing import Literal

from langgraph.graph import END


def get_state(messages) -> Literal["add_tool_message", "info", "__end__"]:
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"
```

### 创建图表

我们现在可以创建图形了。我们将使用SqliteSaver来持久化对话历史。

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessageGraph

memory = SqliteSaver.from_conn_string(":memory:")
workflow = MessageGraph()
workflow.add_node("info", chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state: list):
    return ToolMessage(
        content="Prompt generated!", tool_call_id=state[-1].tool_calls[0]["id"]
    )


workflow.add_conditional_edges("info", get_state)
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)
```

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

<figure><img src="../../.gitbook/assets/下载 (4).jpeg" alt=""><figcaption></figcaption></figure>

### 用图表

我们现在可以使用创建的聊天机器人。

```python
import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
while True:
    user = input("User (q/Q to quit): ")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break
    output = None
    for output in graph.stream(
        [HumanMessage(content=user)], config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))
        last_message.pretty_print()

    if output and "prompt" in output:
        print("Done!")
```

```
================================== Ai Message ==================================

Hello! How can I assist you today?
================================== Ai Message ==================================

Sure! I can help you with that. To create an extraction prompt, I need some information from you. Could you please provide the following details:

1. What is the objective of the prompt?
2. What variables will be passed into the prompt template?
3. Any constraints for what the output should NOT do?
4. Any requirements that the output MUST adhere to?

Once I have this information, I can create the extraction prompt for you.
================================== Ai Message ==================================

Great! To create an extraction prompt for filling out a CSAT (Customer Satisfaction) survey, I will need the following information:

1. Objective: To gather feedback on customer satisfaction.
2. Variables: Customer name, Date of interaction, Service provided, Rating (scale of 1-5), Comments.
3. Constraints: The output should not include any personally identifiable information (PII) of the customer.
4. Requirements: The output must include a structured format with fields for each variable mentioned above.

With this information, I will proceed to create the extraction prompt template for filling out a CSAT survey. Let's get started!
Tool Calls:
  PromptInstructions (call_aU48Bjo7X29tXfRtCcrXkrqq)
 Call ID: call_aU48Bjo7X29tXfRtCcrXkrqq
  Args:
    objective: To gather feedback on customer satisfaction.
    variables: ['Customer name', 'Date of interaction', 'Service provided', 'Rating (scale of 1-5)', 'Comments']
    constraints: ['The output should not include any personally identifiable information (PII) of the customer.']
    requirements: ['The output must include a structured format with fields for each variable mentioned above.']
================================= Tool Message =================================

Prompt generated!
================================== Ai Message ==================================

Please provide feedback on your recent interaction with our service. Your input is valuable to us in improving our services.

Customer name: 
Date of interaction: 
Service provided: 
Rating (scale of 1-5): 
Comments: 

Please note that the output should not include any personally identifiable information (PII) of the customer. Your feedback will be kept confidential and used for internal evaluation purposes only. Thank you for taking the time to share your thoughts with us.
Done!
================================== Ai Message ==================================

I'm glad you found it helpful! If you need any more assistance or have any other requests, feel free to let me know. Have a great day!
AI: Byebye
```

