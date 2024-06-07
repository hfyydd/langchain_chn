# 构建聊天机器人

### 概述

我们将讨论如何设计和实现一个基于LLM（大语言模型）的聊天机器人。这个聊天机器人能够进行对话并记住先前的交互。

请注意，我们构建的这个聊天机器人只会使用语言模型进行对话。您可能还在寻找以下几个相关概念：

#### 对话式RAG：实现基于外部数据源的聊天机器人体验&#x20;

#### 代理：构建能够执行操作的聊天机器人&#x20;

本教程将涵盖基本知识，这些知识对于上述两个更高级的主题很有帮助，但如果您愿意，也可以直接跳到这些高级主题。

### 概念&#x20;

以下是我们将要处理的一些高级组件：

#### 聊天模型。

聊天机器人界面是围绕消息而不是纯文本构建的，因此更适合于聊天模型而不是文本大语言模型（LLM）。

#### &#x20;提示模板。

简化了组合默认消息、用户输入、聊天历史和（可选的）附加检索上下文的提示组装过程。&#x20;

#### 聊天历史。

允许聊天机器人“记住”过去的交互，并在回答后续问题时将其考虑在内。&#x20;

使用LangSmith调试和追踪您的应用程序 我们将介绍如何将上述组件结合起来，以创建一个强大的对话聊天机器人。

我们将介绍如何将上述组件组合在一起以创建强大的会话聊天机器人。

### 快速入门

首先让我们直接使用模型。聊天模型是LangChain "Runnables" 的实例，这意味着它们提供了一个用于与之交互的标准接口。为了简单地调用模型，我们可以将一系列消息传递给.invoke方法。

```python
from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hi! I'm Bob")])
```

```
AIMessage(content='Hello Bob! How can I assist you today?', response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 12, 'total_tokens': 22}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-be38de4a-ccef-4a48-bf82-4292510a8cbf-0')
```

模型本身没有任何状态的概念。例如，如果你问一个后续问题：

```python
model.invoke([HumanMessage(content="What's my name?")])
```

```
AIMessage(content="I'm sorry, as an AI assistant, I do not have the capability to know your name unless you provide it to me.", response_metadata={'token_usage': {'completion_tokens': 26, 'prompt_tokens': 12, 'total_tokens': 38}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_caf95bb1ae', 'finish_reason': 'stop', 'logprobs': None}, id='run-8d8a9d8b-dddb-48f1-b0ed-ce80ce5397d8-0')
```

为了解决这个问题，我们需要将整个对话历史传递到模型中。让我们看看这样做会发生什么：

```python
from langchain_core.messages import AIMessage

model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
```

```
AIMessage(content='Your name is Bob.', response_metadata={'token_usage': {'completion_tokens': 5, 'prompt_tokens': 35, 'total_tokens': 40}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-5692718a-5d29-4f84-bad1-a9819a6118f1-0')
```

现在我们可以看到我们得到了一个很好的回复！

这是支持聊天机器人进行对话能力的基本理念。那么我们该如何最好地实现这一点呢？

### Message History&#x20;

我们可以使用消息历史类来包装我们的模型并使其具有状态。这将跟踪模型的输入和输出，并将它们存储在某个数据存储中。未来的交互将加载这些消息，并将它们作为输入的一部分传递到链中。让我们看看如何使用它！

首先，让我们确保安装langchain-community，因为我们将使用其中的一个集成来存储消息历史。

```
# ! pip install langchain_community
```

之后，我们可以导入相关的类并设置我们的链，该链包装了模型并添加了消息历史记录。这里的一个关键部分是我们传递给get\_session\_history的函数。这个函数预计会接收一个session\_id并返回一个消息历史记录对象。这个session\_id用于区分不同的对话，并应该作为调用新链时的配置的一部分传递（我们将展示如何做到这一点）。

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(model, get_session_history)
```

现在我们需要创建一个配置，每次都传递给可运行对象。这个配置包含的信息不是直接作为输入的一部分，但仍然很有用。在这种情况下，我们想要包含一个session\_id。配置应该如下所示：

```python
config = {"configurable": {"session_id": "abc2"}}
```

```python
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)

response.content
```

```python
'Hello Bob! How can I assist you today?'
```

```python
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content
```

```
'Your name is Bob.'
```

太棒了！我们的聊天机器人现在能记住关于我们的事情。如果我们将配置更改为引用不同的session\_id，我们可以看到它会重新开始对话。

```python
config = {"configurable": {"session_id": "abc3"}}

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content
```

```
"I'm sorry, I do not have the ability to know your name unless you tell me."
```

然而，我们始终可以回到原始的对话（因为我们将其持久化在数据库中）。

```python
config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content
```

```
'Your name is Bob.'
```

这就是我们如何支持聊天机器人与许多用户进行对话！

现在，我们所做的一切只是在模型周围添加了一个简单的持久化层。我们可以通过添加提示模板来使其更加复杂和个性化。

### Prompt templates <a href="#prompt-templates" id="prompt-templates"></a>

提示模板有助于将原始用户信息转换为LLM可以处理的格式。在这种情况下，原始用户输入只是一条消息，我们将其传递给LLM。现在让我们让它变得更复杂一些。首先，让我们添加一个带有一些自定义指令的系统消息（但仍然以消息作为输入）。接下来，我们将添加除了消息之外的更多输入。

首先，让我们添加一个系统消息。为此，我们将创建一个ChatPromptTemplate。我们将利用MessagesPlaceholder来传递所有的消息。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model
```

请注意，这略微更改了输入类型 - 现在我们不再传入消息列表，而是传入一个字典，其中包含一个名为messages的键，其中包含一个消息列表。

```python
response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})

response.content
```

```
'Hello, Bob! How can I assist you today?'
```

现在我们可以像之前一样将其包装在相同的消息历史对象中。

```python
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
```

```
config = {"configurable": {"session_id": "abc5"}}
```

```
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Jim")],
    config=config,
)

response.content
```

```
'Hello, Jim! How can I assist you today?'
```

```
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content
```

```
'Your name is Jim. How can I assist you further, Jim?'
```

太棒了！现在让我们把我们的提示模板变得更加复杂一些。假设现在提示模板看起来像这样：

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model
```

请注意，我们在提示中添加了一个新的语言输入。现在，我们可以调用链并传入我们选择的语言。

```
response = chain.invoke(
    {"messages": [HumanMessage(content="hi! I'm bob")], "language": "Spanish"}
)

response.content
```

```
'¡Hola Bob! ¿En qué puedo ayudarte hoy?'
```

现在让我们将这个更复杂的链包装在一个消息历史类中。这次，因为输入中有多个键，我们需要指定正确的键来保存聊天历史记录。

```python
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
```

```
config = {"configurable": {"session_id": "abc11"}}
```

```python
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="hi! I'm todd")], "language": "Spanish"},
    config=config,
)

response.content
```

```
'¡Hola Todd! ¿En qué puedo ayudarte hoy?'
```

```python
response = with_message_history.invoke(
    {"messages": [HumanMessage(content="whats my name?")], "language": "Spanish"},
    config=config,
)

response.content
```

```
'Tu nombre es Todd. ¿Hay algo más en lo que pueda ayudarte?'
```

### Managing Conversation History <a href="#managing-conversation-history" id="managing-conversation-history"></a>

在构建聊天机器人时理解的一个重要概念是如何管理对话历史记录。如果不加管理地进行，消息列表将不受限制地增长，并有可能溢出LLM的上下文窗口。因此，重要的是在传递消息之前添加一个限制消息大小的步骤。

重要的是，在加载来自消息历史记录的先前消息之后，您将希望在提示模板之前执行此操作。

我们可以通过在提示之前添加一个简单的步骤来完成这个任务，该步骤适当地修改传入的消息键，然后将该新链包装在消息历史类中。首先，让我们定义一个函数，该函数将修改传入的消息。我们将其设置为选择最近的k条消息。然后，我们可以通过在开头添加该函数来创建一个新的链。

```python
from langchain_core.runnables import RunnablePassthrough


def filter_messages(messages, k=10):
    return messages[-k:]


chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | model
)
```

让我们现在尝试一下！如果我们创建一个超过10条消息的消息列表，我们可以看到它不再记得早期消息中的信息。

```python
messages = [
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
```

```python
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
response.content
```

```
"I'm sorry, I don’t have access to your name. Can I help you with anything else?"
```

但如果我们询问的信息在最近的十条消息中，它仍然会记得。

```python
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my fav ice cream")],
        "language": "English",
    }
)
response.content
```

```
'You mentioned that you like vanilla ice cream.'
```

现在让我们将其包装在消息历史中。

```python
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc20"}}
```

```python
response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)

response.content
```

```
"I'm sorry, I don't know your name."
```

现在聊天历史中有两条新消息。这意味着我们对话历史中原本可访问的更多信息现在不再可用！

```python
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="whats my favorite ice cream?")],
        "language": "English",
    },
    config=config,
)

response.content
```

```
"I'm sorry, I don't know your favorite ice cream flavor."
```

### Streaming <a href="#streaming" id="streaming"></a>

现在我们有了一个聊天机器人的函数。然而，对于聊天机器人应用程序来说，一个非常重要的用户体验考虑因素是流式传输。LLM有时需要一段时间才能做出响应，因此为了改善用户体验，大多数应用程序都会将每个生成的标记流式返回。这样用户就可以看到进度了。

实际上，做到这一点非常容易！

所有链都公开了一个 .stream 方法，使用消息历史的链也不例外。我们只需使用该方法即可获得一个流式响应。

```
|Sure|,| Todd|!| Here|'s| a| joke| for| you|:

|Why| don|'t| scientists| trust| atoms|?

|Because| they| make| up| everything|!||
```

### Next Steps

现在您已经了解了如何在LangChain中创建聊天机器人的基础知识，您可能会对一些更高级的教程感兴趣：

* 对话RAG：在外部数据源上启用聊天机器人体验
* 代理：构建可以执行操作的聊天机器人

如果您想深入了解特定内容，一些值得查看的内容包括：

* 流式传输：流式传输对于聊天应用程序至关重要
* 如何添加消息历史：深入探讨与消息历史相关的所有内容
