# 使用LCEL构建一个简单的LLM应用程序

在本快速入门中，我们将向您展示如何使用LangChain构建一个简单的LLM应用程序。此应用程序将文本从英语翻译成另一种语言。这是一个相对简单的LLM应用程序——它只是一个LLM调用加上一些提示。尽管如此，这仍然是开始使用LangChain的好方法——只需一些提示和一个LLM调用，就可以构建许多功能！

阅读本教程后，您将对以下内容有一个高层次的概述：

* 使用语言模型
* 使用 PromptTemplates 和 OutputParsers
* 使用 LangChain Expression Language (LCEL) 将组件连接在一起
* 使用 LangSmith 调试和跟踪您的应用程序
* 使用 LangServe 部署您的应用程序

让我们开始吧！

### Setup <a href="#outputparsers" id="outputparsers"></a>

Jupyter Notebook&#x20;

本指南（以及文档中的大多数其他指南）使用 Jupyter notebook，并假设读者也是如此。Jupyter notebook 非常适合学习如何使用 LLM 系统，因为通常会出现问题（意外输出、API 异常等），在交互式环境中浏览指南是更好地理解它们的好方法。

本教程和其他教程最方便的运行环境可能是 Jupyter notebook。请参阅此处了解如何安装。

安装&#x20;

要安装 LangChain，请运行：

Pip Conda

```bash
pip install langchain
```

LangSmith&#x20;

您使用 LangChain 构建的许多应用程序将包含多个步骤和多次 LLM 调用。随着这些应用程序变得越来越复杂，能够检查链或代理内部发生的事情变得至关重要。最好的方法是使用 LangSmith。

在上述链接注册后，请确保设置环境变量以开始记录跟踪信息：

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

或者，如果在 notebook 中，可以使用以下方法设置：

```python
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

使用语言模型 首先，让我们学习如何单独使用语言模型。LangChain 支持许多不同的语言模型，您可以互换使用——选择您要使用的模型！

```bash
pip install -qU langchain-openai
```

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")
```

让我们首先直接使用模型。ChatModels是LangChain“Runnables”的实例，这意味着它们公开了一个与它们交互的标准接口。要简单地调用模型，我们可以将消息列表传递给. invoke方法。

```python

# 直接使用模型
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="将以下内容从英文翻译成意大利语"),
    HumanMessage(content="hi!"),
]

model.invoke(messages)
```

```python
AIMessage(content='ciao!', response_metadata={'token_usage': {'completion_tokens': 3, 'prompt_tokens': 20, 'total_tokens': 23}, 'model_name': 'gpt-4', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-fc5d7c88-9615-48ab-a3c7-425232b562c5-0')

```

如果我们启用了 LangSmith，可以看到此次运行被记录在 LangSmith 中，并查看 LangSmith 跟踪信息。

### OutputParsers <a href="#outputparsers" id="outputparsers"></a>

注意，模型的响应是一个 AIMessage。它包含一个字符串响应以及有关响应的其他元数据。通常我们可能只想处理字符串响应。我们可以使用一个简单的输出解析器来解析出这个响应。

首先导入简单输出解析器。

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
```

使用它的一种方法是单独使用它。例如，我们可以保存语言模型调用的结果，然后将其传递给解析器。

```
# 可以单独使用它
result = model.invoke(messages)

parser.invoke(result)
# 'Ciao!'
```

更常见的是，我们可以将模型与这个输出解析器“链”在一起。这意味着每次在这个链中调用时都会调用这个输出解析器。这个链采用语言模型的输入类型（字符串或消息列表）并返回输出解析器的输出类型（字符串）。

我们可以轻松使用 | 操作符创建链。 | 操作符用于在 LangChain 中组合两个元素。

```python
chain = model | parser

chain.invoke(messages)
# 'Ciao!'
```

如果我们现在查看 LangSmith，可以看到该链有两个步骤：首先调用语言模型，然后将结果传递给输出解析器。

### Prompt Templates <a href="#prompt-templates" id="prompt-templates"></a>

目前我们直接将消息列表传递给语言模型。这个消息列表从何而来？通常，它是从用户输入和应用程序逻辑的组合中构建的。此应用程序逻辑通常接受原始用户输入并将其转换为准备传递给语言模型的消息列表。常见的转换包括添加系统消息或使用用户输入格式化模板。

PromptTemplates 是 LangChain 中设计用于帮助此转换的概念。它们接收原始用户输入并返回准备传递给语言模型的数据（提示）。

让我们在此处创建一个 PromptTemplate。它将接受两个用户变量：

* 语言：要将文本翻译成的语言
* 文本：要翻译的文本

```
from langchain_core.prompts import ChatPromptTemplate
```

首先，让我们创建一个字符串，我们将其格式化为系统消息：

```
system_template = "将以下内容翻译成{language}："
```

接下来，我们可以创建PromptTemplate。这将是system\_template的组合以及放置文本的更简单模板

```
# 创建 PromptTemplate
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
```

这个提示模板的输入是字典。我们可以自己玩这个提示模板，看看它自己做了什么

```python
# 此提示模板的输入是一个字典。我们可以单独试验此提示模板以查看其效果。
result = prompt_template.invoke({"language": "italian", "text": "hi"})

result
```

<pre><code><strong>ChatPromptValue(messages=[SystemMessage(content='将以下内容翻译成italian:'), HumanMessage(content='hi')])
</strong></code></pre>

我们可以看到它返回一个由两条消息组成的ChatPromptValue。如果我们想直接访问消息，我们会这样做：

```
result.to_messages()
```

```
[SystemMessage(content='Translate the following into italian:'),
 HumanMessage(content='hi')]
```

### Chaining together components with LCEL <a href="#chaining-together-components-with-lcel" id="chaining-together-components-with-lcel"></a>

我们现在可以使用管道操作符 | 将其与上述模型和输出解析器结合起来：

```python
chain = prompt_template | model | parser

chain.invoke({"language": "italian", "text": "hi"})
# 'ciao'
```

这是使用 LangChain Expression Language (LCEL) 链接 LangChain 模块的一个简单示例。此方法的几个优点包括优化流式传输和跟踪支持。

如果查看 LangSmith 跟踪，可以看到所有三个组件都显示在 LangSmith 跟踪中。

### Serving with LangServe <a href="#serving-with-langserve" id="serving-with-langserve"></a>

&#x20;现在我们已经构建了一个应用程序，需要提供服务。这就是 LangServe 的用武之地。LangServe 帮助开发人员将 LangChain 链部署为 REST API。您无需使用 LangServe 即可使用 LangChain，但在本指南中我们将展示如何使用 LangServe 部署应用程序。

虽然本指南的第一部分旨在在 Jupyter Notebook 或脚本中运行，但现在我们将离开该环境。我们将创建一个 Python 文件，然后从命令行与之交互。

安装：

```bash
pip install "langserve[all]"
```

#### Server <a href="#server" id="server"></a>

要为我们的应用程序创建服务器，我们将创建一个 serve.py 文件。这将包含我们的应用程序服务逻辑。它包含三个部分：

1. 我们刚刚构建的链的定义
2. 我们的 FastAPI 应用程序
3. 使用 langserve.add\_routes 定义的服务链的路由

```python
#!/usr/bin/env python
from typing import List
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. 创建提示模板
system_template = "将以下内容翻译成{language}："
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. 创建模型
model = ChatOpenAI()

# 3. 创建解析器
parser = StrOutputParser()

# 4. 创建链
chain = prompt_template | model | parser

# 4. 应用定义
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="使用 LangChain Runnable 接口的简单 API 服务器",
)

# 5. 添加链路由
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
```

这就是全部！如果我们执行此文件：

```bash
python serve.py
```

我们应该看到我们的链在 http://localhost:8000 提供服务。

#### Playground <a href="#playground" id="playground"></a>

&#x20;每个 LangServe 服务都带有一个简单的内置 UI，用于配置和调用应用程序，支持流式输出和查看中间步骤。前往 http://localhost:8000/chain/playground/ 试用！输入与之前相同的参数 - {"language": "italian", "text": "hi"} - 应该会有相同的响应。

#### Client <a href="#client" id="client"></a>

&#x20;现在让我们设置一个客户端以编程方式与我们的服务交互。我们可以轻松使用 langserve.RemoteRunnable。使用此方法，我们可以与提供的链交互，就像它在客户端运行一样。

```python
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chain/")
remote_chain.invoke({"language": "italian", "text": "hi"})
# 'Ciao'
```



