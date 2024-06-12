# 构建PDF摄取和问答系统

PDF 文件通常包含其他来源无法获得的重要非结构化数据。它们可能相当冗长，并且与纯文本文件不同，通常不能直接输入到语言模型的提示中。在本教程中，你将创建一个能够回答有关 PDF 文件问题的系统。更具体地说，你将使用文档加载器（Document Loader）以 LLM 可用的格式加载文本，然后构建一个检索增强生成（RAG）管道来回答问题，并包含源材料的引用。

本教程将略过一些在我们的 RAG 教程中更详细介绍的概念，因此如果你还没有阅读那些内容，建议先阅读那些教程。

让我们开始吧！

### 加载文件

首先，你需要选择一个要加载的 PDF。我们将使用耐克年度公开的 SEC 报告文件。该文件超过 100 页，包含一些与较长解释性文本混合的重要数据。不过，你可以自由选择你想要使用的 PDF。

一旦你选择了 PDF，下一步就是将其加载到一个 LLM 更容易处理的格式中，因为 LLM 通常需要文本输入。LangChain 有几个内置的文档加载器用于此目的，你可以尝试使用。

下面，我们将使用一个由 pypdf 包提供支持的加载器，它从文件路径读取：

```python
%pip install -qU pypdf langchain_community
```

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "../example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
```

API Reference:[PyPDFLoader](https://api.python.langchain.com/en/latest/document\_loaders/langchain\_community.document\_loaders.pdf.PyPDFLoader.html)

```python
107
```

```python
print(docs[0].page_content[0:100])
print(docs[0].metadata)
```

```python
Table of Contents
UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K

{'source': '../example_data/nke-10k-2023.pdf', 'page': 0}
```

刚刚发生了什么？

1. 加载器读取指定路径的 PDF 文件到内存中。
2. 它使用 pypdf 包提取文本数据。
3. 最后，它为 PDF 的每一页创建一个 LangChain 文档，包含页面内容及一些关于文本来源的元数据。

LangChain 还有许多用于其他数据源的文档加载器，或者你可以创建一个自定义文档加载器。

### 用RAG回答问题

接下来，您将准备加载的文档以供以后检索。使用文本拆分器，您将加载的文档拆分为更容易放入LLM上下文窗口的较小文档，然后将它们加载到矢量存储中。然后，您可以从矢量存储中创建一个检索器，用于我们的RAG链：

```python
pip install -qU langchain-openai
```

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
```

```python
%pip install langchain_chroma langchain_openai
```

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
```

API Reference:[OpenAIEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain\_openai.embeddings.base.OpenAIEmbeddings.html) | [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain\_text\_splitters.character.RecursiveCharacterTextSplitter.html)

最后，您将使用一些内置的助手来构建最终的rag\_chain：

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "What was Nike's revenue in 2023?"})

results
```

API Reference:[create\_retrieval\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create\_retrieval\_chain.html) | [create\_stuff\_documents\_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine\_documents.stuff.create\_stuff\_documents\_chain.html) | [ChatPromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain\_core.prompts.chat.ChatPromptTemplate.html)

```python
{'input': "What was Nike's revenue in 2023?",
 'context': [Document(page_content='Table of Contents\nFISCAL 2023 NIKE BRAND REVENUE HIGHLIGHTS\nThe following tables present NIKE Brand revenues disaggregated by reportable operating segment, distribution channel and major product line:\nFISCAL 2023 COMPARED TO FISCAL 2022\n•NIKE, Inc. Revenues were $51.2 billion in fiscal 2023, which increased 10% and 16% compared to fiscal 2022 on a reported and currency-neutral basis, respectively.\nThe increase was due to higher revenues in North America, Europe, Middle East & Africa ("EMEA"), APLA and Greater China, which contributed approximately 7, 6,\n2 and 1 percentage points to NIKE, Inc. Revenues, respectively.\n•NIKE Brand revenues, which represented over 90% of NIKE, Inc. Revenues, increased 10% and 16% on a reported and currency-neutral basis, respectively. This\nincrease was primarily due to higher revenues in Men\'s, the Jordan Brand, Women\'s and Kids\' which grew 17%, 35%,11% and 10%, respectively, on a wholesale\nequivalent basis.', metadata={'page': 35, 'source': '../example_data/nke-10k-2023.pdf'}),
  Document(page_content='Enterprise Resource Planning Platform, data and analytics, demand sensing, insight gathering, and other areas to create an end-to-end technology foundation, which we\nbelieve will further accelerate our digital transformation. We believe this unified approach will accelerate growth and unlock more efficiency for our business, while driving\nspeed and responsiveness as we serve consumers globally.\nFINANCIAL HIGHLIGHTS\n•In fiscal 2023, NIKE, Inc. achieved record Revenues of $51.2 billion, which increased 10% and 16% on a reported and currency-neutral basis, respectively\n•NIKE Direct revenues grew 14% from $18.7 billion in fiscal 2022 to $21.3 billion in fiscal 2023, and represented approximately 44% of total NIKE Brand revenues for\nfiscal 2023\n•Gross margin for the fiscal year decreased 250 basis points to 43.5% primarily driven by higher product costs, higher markdowns and unfavorable changes in foreign\ncurrency exchange rates, partially offset by strategic pricing actions', metadata={'page': 30, 'source': '../example_data/nke-10k-2023.pdf'}),
  Document(page_content="Table of Contents\nNORTH AMERICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 14,897 $ 12,228 22 % 22 %$ 11,644 5 % 5 %\nApparel 5,947 5,492 8 % 9 % 5,028 9 % 9 %\nEquipment 764 633 21 % 21 % 507 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nRevenues by:    \nSales to Wholesale Customers $ 11,273 $ 9,621 17 % 18 %$ 10,186 -6 % -6 %\nSales through NIKE Direct 10,335 8,732 18 % 18 % 6,993 25 % 25 %\nTOTAL REVENUES $ 21,608 $ 18,353 18 % 18 %$ 17,179 7 % 7 %\nEARNINGS BEFORE INTEREST AND TAXES $ 5,454 $ 5,114 7 % $ 5,089 0 %\nFISCAL 2023 COMPARED TO FISCAL 2022\n•North America revenues increased 18% on a currency-neutral basis, primarily due to higher revenues in Men's and the Jordan Brand. NIKE Direct revenues\nincreased 18%, driven by strong digital sales growth of 23%, comparable store sales growth of 9% and the addition of new stores.", metadata={'page': 39, 'source': '../example_data/nke-10k-2023.pdf'}),
  Document(page_content="Table of Contents\nEUROPE, MIDDLE EAST & AFRICA\n(Dollars in millions) FISCAL 2023FISCAL 2022 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGESFISCAL 2021 % CHANGE% CHANGE\nEXCLUDING\nCURRENCY\nCHANGES\nRevenues by:\nFootwear $ 8,260 $ 7,388 12 % 25 %$ 6,970 6 % 9 %\nApparel 4,566 4,527 1 % 14 % 3,996 13 % 16 %\nEquipment 592 564 5 % 18 % 490 15 % 17 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nRevenues by:    \nSales to Wholesale Customers $ 8,522 $ 8,377 2 % 15 %$ 7,812 7 % 10 %\nSales through NIKE Direct 4,896 4,102 19 % 33 % 3,644 13 % 15 %\nTOTAL REVENUES $ 13,418 $ 12,479 8 % 21 %$ 11,456 9 % 12 %\nEARNINGS BEFORE INTEREST AND TAXES $ 3,531 $ 3,293 7 % $ 2,435 35 % \nFISCAL 2023 COMPARED TO FISCAL 2022\n•EMEA revenues increased 21% on a currency-neutral basis, due to higher revenues in Men's, the Jordan Brand, Women's and Kids'. NIKE Direct revenues\nincreased 33%, driven primarily by strong digital sales growth of 43% and comparable store sales growth of 22%.", metadata={'page': 40, 'source': '../example_data/nke-10k-2023.pdf'})],
 'answer': 'According to the financial highlights, Nike, Inc. achieved record revenues of $51.2 billion in fiscal 2023, which increased 10% on a reported basis and 16% on a currency-neutral basis compared to fiscal 2022.'}
```

您可以看到，您在结果的答案键和LLM用于生成答案的上下文中都得到了最终答案。进一步检查上下文下的值，您可以看到它们是文档，每个文档都包含一大块摄取的页面内容。有用的是，这些文档还保留了您第一次加载它们时的原始元数据：

```python
print(results["context"][0].page_content)
```

```python
Table of Contents
FISCAL 2023 NIKE BRAND REVENUE HIGHLIGHTS
The following tables present NIKE Brand revenues disaggregated by reportable operating segment, distribution channel and major product line:
FISCAL 2023 COMPARED TO FISCAL 2022
•NIKE, Inc. Revenues were $51.2 billion in fiscal 2023, which increased 10% and 16% compared to fiscal 2022 on a reported and currency-neutral basis, respectively.
The increase was due to higher revenues in North America, Europe, Middle East & Africa ("EMEA"), APLA and Greater China, which contributed approximately 7, 6,
2 and 1 percentage points to NIKE, Inc. Revenues, respectively.
•NIKE Brand revenues, which represented over 90% of NIKE, Inc. Revenues, increased 10% and 16% on a reported and currency-neutral basis, respectively. This
increase was primarily due to higher revenues in Men's, the Jordan Brand, Women's and Kids' which grew 17%, 35%,11% and 10%, respectively, on a wholesale
equivalent basis.
```

```python
print(results["context"][0].metadata)
```

```python
{'page': 35, 'source': '../example_data/nke-10k-2023.pdf'}
```

此特定块来自原始PDF的第35页。您可以使用此数据显示答案来自PDF中的哪个页面，从而使用户能够快速验证答案是否基于源材料。
