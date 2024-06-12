---
description: >-
  本教程将让您熟悉 LangChain 的向量存储和检索器抽象。这些抽象旨在支持从（向量）数据库和其他来源检索数据，以便与 LLM
  工作流集成。它们对于获取数据作为模型推理的一部分进行推理的应用程序非常重要，例如检索增强生成或 RAG（请参阅此处的 RAG 教程）。
---

# 向量储存和检索

## 概念

本指南重点介绍文本数据的检索。我们将介绍以下概念：

文档&#x20;

向量存储&#x20;

检索

## 设置

### Jupyter 笔记本

本教程和其他教程可能在 Jupyter 笔记本中运行最方便。有关如何安装的说明，请参阅此处。

### 安装

本教程需要 langchain、langchain-chroma 和 langchain-openai 包：

{% code title="Pip" %}
```python
pip install langchain langchain-chroma langchain-openai
```
{% endcode %}

{% code title="Conda" %}
```python
conda install langchain langchain-chroma langchain-openai -c conda-forge
```
{% endcode %}

有关更多详细信息，请参阅我们的安装指南（[Installation guide](https://python.langchain.com/v0.2/docs/how\_to/installation/)）。

### LangSmith

您使用 LangChain 构建的许多应用程序将包含多个步骤，其中包含多次调用 LLM 调用。随着这些应用程序变得越来越复杂，能够检查您的链或代理内部到底发生了什么变得至关重要。最好的方法是使用 LangSmith。

在上面的链接上注册后，请确保设置环境变量以开始记录跟踪：&#x20;

```python
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
```

或者，如果在笔记本中，则可以使用以下方法设置它们：

```python
import getpass
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## 文档

LangChain实现了一个文档抽象，旨在表示文本单元和关联的元数据。它有两个属性： page\_content:表示内容的字符串 metadata: 包含任意元数据的字典。 metadata 属性可以捕获有关文档源、文档与其他文档的关系以及其他信息的信息。请注意，单个 Document 对象通常表示较大文档的一部分。

让我们生成一些示例文档：

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```

API 参考: [Document](https://api.python.langchain.com/en/latest/documents/langchain\_core.documents.base.Document.html)

在这里，我们生成了五个文档，其中包含指示三个不同“来源”的元数据。

## 向量存储

向量搜索是存储和搜索非结构化数据（如非结构化文本）的常用方法。这个想法是存储与文本关联的数字向量。给定一个查询，我们可以将其嵌入为相同维度的向量，并使用向量相似度量来识别存储中的相关数据。 LangChain VectorStore 对象包含用于将文本和 Document 对象添加到存储中的方法，以及使用各种相似度指标查询它们的方法。它们通常使用嵌入模型进行初始化，这些模型决定了如何将文本数据转换为数字向量。 LangChain 包括一套与不同向量存储技术的集成。一些矢量存储由提供商（例如，各种云提供商）托管，需要特定的凭据才能使用;有些（如Postgres）运行在单独的基础设施中，可以在本地或通过第三方运行;其他人可以在内存中运行轻量级工作负载。在这里，我们将演示使用 Chroma 使用 LangChain VectorStores，其中包括内存实现。 为了实例化向量存储，我们通常需要提供一个嵌入模型来指定如何将文本转换为数字向量。在这里，我们将使用 OpenAI 嵌入。

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)
```

&#x20;API 参考：[OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain\_openai.embeddings.base.OpenAIEmbeddings.html)

在此处调用 .from\_documents 会将文档添加到矢量存储中。VectorStore 实现用于添加文档的方法，这些文档也可以在实例化对象后调用。大多数实现都允许您连接到现有的向量存储，例如，通过提供客户端、索引名称或其他信息。有关更多详细信息，请参阅特定集成的文档。

一旦我们实例化了包含文档的 VectorStore，我们就可以查询它。VectorStore 包括用于查询的方法：

同步和异步;&#x20;

通过字符串查询和向量;&#x20;

有和没有返回相似性分数;&#x20;

通过相似性和最大边际相关性（以平衡相似性与查询到检索结果的多样性）。&#x20;

这些方法通常会在其输出中包含 Document 对象的列表。

### 示例

根据与字符串查询的相似性返回文档：

```python
vectorstore.similarity_search("cat")
```

```
[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
```

异步查询：

```python
await vectorstore.asimilarity_search("cat")
```

```
[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
```

返回分数：

```python
# Note that providers implement different scores; Chroma here
# returns a distance metric that should vary inversely with
# similarity.

vectorstore.similarity_search_with_score("cat")
```



```
[(Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
  0.3751849830150604),
 (Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
  0.48316916823387146),
 (Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
  0.49601367115974426),
 (Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'}),
  0.4972994923591614)]
```

根据与嵌入查询的相似性返回文档：

```python
embedding = OpenAIEmbeddings().embed_query("cat")

vectorstore.similarity_search_by_vector(embedding)
```

```
[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Dogs are great companions, known for their loyalty and friendliness.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Rabbits are social animals that need plenty of space to hop around.', metadata={'source': 'mammal-pets-doc'}),
 Document(page_content='Parrots are intelligent birds capable of mimicking human speech.', metadata={'source': 'bird-pets-doc'})]
```

了解更多：

* [API reference](https://api.python.langchain.com/en/latest/vectorstores/langchain\_core.vectorstores.VectorStore.html)
* [How-to guide](https://python.langchain.com/v0.2/docs/how\_to/vectorstores/)
* [Integration-specific docs](https://python.langchain.com/v0.2/docs/integrations/vectorstores/)

## 检索

LangChain VectorStore 对象不对 Runnable 进行子类化，因此不能立即集成到 LangChain 表达式语言链中。

LangChain Retriever 是可运行的，因此它们实现了一组标准方法（例如，同步和异步调用和批处理操作），并被设计为合并到 LCEL 链中。

我们可以自己创建一个简单的版本，而无需对 Retriever 进行子类化。如果我们选择我们希望使用哪种方法来检索文档，我们可以轻松创建一个可运行的。下面我们将围绕 similarity\_search 方法构建一个：

```python
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result

retriever.batch(["cat", "shark"])
```

API 参考：[Document](https://api.python.langchain.com/en/latest/documents/langchain\_core.documents.base.Document.html) | [RunnableLambda](https://api.python.langchain.com/en/latest/runnables/langchain\_core.runnables.base.RunnableLambda.html)

```
[[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'})],
 [Document(page_content='Goldfish are popular pets for beginners, requiring relatively simple care.', metadata={'source': 'fish-pets-doc'})]]
```

Vectorstores 实现一个 as\_retriever 方法，该方法将生成一个 Retriever，特别是一个 VectorStoreRetriever。这些检索器包括特定的 search\_type 和 search\_kwargs 属性，用于标识要调用的基础向量存储的哪些方法，以及如何参数化它们。例如，我们可以使用以下内容复制上述内容：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(["cat", "shark"])
```

```
[[Document(page_content='Cats are independent pets that often enjoy their own space.', metadata={'source': 'mammal-pets-doc'})],
 [Document(page_content='Goldfish are popular pets for beginners, requiring relatively simple care.', metadata={'source': 'fish-pets-doc'})]]
```

VectorStoreRetriever 支持“相似性”（默认）、“mmr”（最大边际相关性，如上所述）和“similarity\_score\_threshold”的搜索类型。我们可以使用后者通过相似性分数对检索器输出的文档进行阈值。

检索器可以很容易地合并到更复杂的应用程序中，例如检索增强生成 （RAG） 应用程序，它将给定的问题与检索到的上下文组合成 LLM 的提示。下面我们展示一个小例子。

{% code title="OpenAI   " fullWidth="true" %}
```
pip install -qU langchain-openai
```
{% endcode %}

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

API 参考: [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain\_core.prompts.chat.ChatPromptTemplate.html) | [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain\_core.runnables.passthrough.RunnablePassthrough.html)

```python
response = rag_chain.invoke("tell me about cats")

print(response.content)
```

```
猫是独立的宠物，经常享受自己的空间。
```

## 了解更多：

检索策略可以是丰富而复杂的。例如：

我们可以从查询中推断出硬规则和过滤器（例如，“使用 2020 年之后发布的文档”）;&#x20;

我们可以以某种方式（例如，通过某些文档分类法）返回链接到检索到的上下文的文档;&#x20;

我们可以为每个上下文单元生成多个嵌入;&#x20;

我们可以汇总来自多个检索器的结果;&#x20;

我们可以为文档分配权重，例如，将最近的文档权重更高。&#x20;

操作指南的检索器部分介绍了这些策略和其他内置检索策略。

扩展 BaseRetriever 类以实现自定义检索器也很简单。请在此处查看我们的操作指南。
