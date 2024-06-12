# 构建查询分析系统

本页将展示如何在一个基本的端到端示例中使用查询分析。这将涵盖创建一个简单的搜索引擎，显示将原始用户问题传递给该搜索时发生的故障模式，然后是查询分析如何帮助解决该问题的示例。有许多不同的查询分析技术，这个端到端示例不会展示所有技术。

为了本示例的目的，我们将在LangChainYouTube视频上进行检索。

### Setup

安装依赖项

```python
# %pip install -qU langchain langchain-community langchain-openai youtube-transcript-api pytube langchain-chroma
```

设置环境变量#

我们将在此示例中使用OpenAI：

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Optional, uncomment to trace runs with LangSmith. Sign up here: https://smith.langchain.com.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

### 加载文档

我们可以使用YouTubeLoader加载一些LangChain视频的转录本：

```python
from langchain_community.document_loaders import YoutubeLoader

urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]
docs = []
for url in urls:
    docs.extend(YoutubeLoader.from_youtube_url(url, add_video_info=True).load())
```

API Reference:[YoutubeLoader](https://api.python.langchain.com/en/latest/document\_loaders/langchain\_community.document\_loaders.youtube.YoutubeLoader.html)

```python
import datetime

# Add some additional metadata: what year the video was published
for doc in docs:
    doc.metadata["publish_year"] = int(
        datetime.datetime.strptime(
            doc.metadata["publish_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y")
    )
```

以下是我们加载的视频的标题：

```python
[doc.metadata["title"] for doc in docs]
```

```python
['OpenGPTs',
 'Building a web RAG chatbot: using LangChain, Exa (prev. Metaphor), LangSmith, and Hosted Langserve',
 'Streaming Events: Introducing a new `stream_events` method',
 'LangGraph: Multi-Agent Workflows',
 'Build and Deploy a RAG app with Pinecone Serverless',
 'Auto-Prompt Builder (with Hosted LangServe)',
 'Build a Full Stack RAG App With TypeScript',
 'Getting Started with Multi-Modal LLMs',
 'SQL Research Assistant',
 'Skeleton-of-Thought: Building a New Template from Scratch',
 'Benchmarking RAG over LangChain Docs',
 'Building a Research Assistant from Scratch',
 'LangServe and LangChain Templates Webinar']
```

这是与每个视频相关的元数据。我们可以看到每个文档也有标题、观看次数、发布日期和长度：

```python
docs[0].metadata
```

```python
{'source': 'HAn9vnJy6S4',
 'title': 'OpenGPTs',
 'description': 'Unknown',
 'view_count': 7210,
 'thumbnail_url': 'https://i.ytimg.com/vi/HAn9vnJy6S4/hq720.jpg',
 'publish_date': '2024-01-31 00:00:00',
 'length': 1530,
 'author': 'LangChain',
 'publish_year': 2024}
```

以下是文档内容的示例：

```python
docs[0].page_content[:500]
```

```python
"hello today I want to talk about open gpts open gpts is a project that we built here at linkchain uh that replicates the GPT store in a few ways so it creates uh end user-facing friendly interface to create different Bots and these Bots can have access to different tools and they can uh be given files to retrieve things over and basically it's a way to create a variety of bots and expose the configuration of these Bots to end users it's all open source um it can be used with open AI it can be us"
```

### 索引文档

每当我们执行检索时，我们都需要创建一个可以查询的文档索引。我们将使用向量存储来索引我们的文档，我们将首先对它们进行分块以使我们的检索更加简洁和精确：

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
chunked_docs = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunked_docs,
    embeddings,
)
```

API Reference:[OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain\_openai.embeddings.base.OpenAIEmbeddings.html) | [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain\_text\_splitters.character.RecursiveCharacterTextSplitter.html)

### 无需查询分析的检索

我们可以直接对用户问题执行相似性搜索以查找与问题相关的块：

```python
search_results = vectorstore.similarity_search("how do I build a RAG agent")
print(search_results[0].metadata["title"])
print(search_results[0].page_content[:500])
```

```python
Build and Deploy a RAG app with Pinecone Serverless
hi this is Lance from the Lang chain team and today we're going to be building and deploying a rag app using pine con serval list from scratch so we're going to kind of walk through all the code required to do this and I'll use these slides as kind of a guide to kind of lay the the ground work um so first what is rag so under capoy has this pretty nice visualization that shows LMS as a kernel of a new kind of operating system and of course one of the core components of our operating system is th
```

这工作得很好！我们的第一个结果与问题非常相关。如果我们想搜索特定时间段的结果怎么办？

```python
search_results = vectorstore.similarity_search("videos on RAG published in 2023")
print(search_results[0].metadata["title"])
print(search_results[0].metadata["publish_date"])
print(search_results[0].page_content[:500])
```

```python
OpenGPTs
2024-01-31
hardcoded that it will always do a retrieval step here the assistant decides whether to do a retrieval step or not sometimes this is good sometimes this is bad sometimes it you don't need to do a retrieval step when I said hi it didn't need to call it tool um but other times you know the the llm might mess up and not realize that it needs to do a retrieval step and so the rag bot will always do a retrieval step so it's more focused there because this is also a simpler architecture so it's always
```

我们的第一个结果来自2024年（尽管我们要求从2023年开始播放视频），并且与输入不太相关。由于我们只是搜索文档内容，因此无法根据任何文档属性过滤结果。这只是可能出现的一种故障模式。现在让我们看看查询分析的基本形式如何修复它！

### 查询分析

我们可以使用查询分析来改进检索结果。这将涉及定义一个包含一些日期过滤器的查询模式，并使用函数调用模型将用户问题转换为结构化查询。

#### 查询模式\#

在这种情况下，我们将为发布日期提供显式的min和max属性，以便对其进行过滤。

```python
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field


class Search(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    query: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    publish_year: Optional[int] = Field(None, description="Year video was published")
```

#### 查询生成

为了将用户问题转换为结构化查询，我们将使用OpenAI的工具调用API。具体来说，我们将使用新的[ChatModel.with\_structured\_output()](https://python.langchain.com/v0.2/docs/how\_to/structured\_output/) 构造函数来处理将模式传递给模型并解析输出。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a list of database queries optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm
```

API Reference:[ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain\_core.prompts.chat.ChatPromptTemplate.html) | [RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain\_core.runnables.passthrough.RunnablePassthrough.html) | [ChatOpenAI](https://api.python.langchain.com/en/latest/chat\_models/langchain\_openai.chat\_models.base.ChatOpenAI.html)

```python
/Users/bagatur/langchain/libs/core/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: The function `with_structured_output` is in beta. It is actively being worked on, so the API may change.
  warn_beta(
```

让我们看看我们的分析器为我们之前搜索的问题生成了哪些查询：

```python
query_analyzer.invoke("how do I build a RAG agent")
```

```python
Search(query='build RAG agent', publish_year=None)
```

```python
query_analyzer.invoke("videos on RAG published in 2023")
```

```python
Search(query='RAG', publish_year=2023)
```

### 使用查询分析进行检索

我们的查询分析看起来非常好；现在让我们尝试使用生成的查询来实际执行检索。

**注意**：在我们的示例中，我们指定了tool\_choice="Search"。这将强制LLM调用一个——而且只有一个——工具，这意味着我们将始终有一个优化的查询要查找。请注意，情况并非总是如此——请参阅其他指南，了解如何处理不返回或多个优化查询的情况。

```python
from typing import List

from langchain_core.documents import Document
```

API Reference:[Document](https://api.python.langchain.com/en/latest/documents/langchain\_core.documents.base.Document.html)

```python
def retrieval(search: Search) -> List[Document]:
    if search.publish_year is not None:
        # This is syntax specific to Chroma,
        # the vector database we are using.
        _filter = {"publish_year": {"$eq": search.publish_year}}
    else:
        _filter = None
    return vectorstore.similarity_search(search.query, filter=_filter)
```

```python
retrieval_chain = query_analyzer | retrieval
```

我们现在可以在之前有问题的输入上运行这个链，并看到它只产生当年的结果！

```python
results = retrieval_chain.invoke("RAG tutorial published in 2023")
```

```python
[(doc.metadata["title"], doc.metadata["publish_date"]) for doc in results]
```

```python
[('Getting Started with Multi-Modal LLMs', '2023-12-20 00:00:00'),
 ('LangServe and LangChain Templates Webinar', '2023-11-02 00:00:00'),
 ('Getting Started with Multi-Modal LLMs', '2023-12-20 00:00:00'),
 ('Building a Research Assistant from Scratch', '2023-11-16 00:00:00')]
```
