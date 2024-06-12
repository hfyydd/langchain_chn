# 构建本地RAG应用程序

私有GPT、llama. cpp、GPT4All和llamafile等项目的流行强调了在本地运行LLM的重要性。

LangChain与许多可以在本地运行的开源LLM集成。

请参阅此处 [here](https://python.langchain.com/v0.2/docs/how\_to/local\_llms/) 了解这些LLM的设置说明。

例如，在这里，我们展示了如何使用本地嵌入和本地LLM在本地（例如，在您的笔记本电脑上）运行GPT4All或LLaMA2。

### 文件加载

首先，安装本地嵌入和矢量存储所需的包。

```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub gpt4all langchain-chroma 
```

加载并拆分示例文档。

我们将使用一篇关于代理的博客文章作为示例。

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
```

API Reference:[WebBaseLoader](https://api.python.langchain.com/en/latest/document\_loaders/langchain\_community.document\_loaders.web\_base.WebBaseLoader.html) | [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/character/langchain\_text\_splitters.character.RecursiveCharacterTextSplitter.html)

接下来，以下步骤将在本地下载GPT4All嵌入（如果您还没有它们）。

```python
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
```

API Reference:[GPT4AllEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain\_community.embeddings.gpt4all.GPT4AllEmbeddings.html)

\
测试相似性搜索正在使用我们的本地嵌入。

```python
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
len(docs)
```

```python
4
```

```python
docs[0]
```

```python
Document(page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'description': 'Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.\nAgent System Overview In a LLM-powered autonomous agent system, LLM functions as the agent’s brain, complemented by several key components:', 'language': 'en', 'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/', 'title': "LLM Powered Autonomous Agents | Lil'Log"})
```

### 模型

#### LLaMA2 <a href="#llama2" id="llama2"></a>

注意：新版本的lama-cpp-python使用GGUF模型文件（见此处[here](https://github.com/abetlen/llama-cpp-python/pull/633)）。

如果您有现有的GGML模型，请参阅此处 [here](https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/) 了解GGUF的转换说明。

和/或，您可以下载GGUF转换模型（例如，此处[here](https://huggingface.co/TheBloke)）。

最后，如这里 [here](https://python.langchain.com/v0.2/docs/how\_to/local\_llms/)详细说明的，安装lama-cpp-python

```python
%pip install --upgrade --quiet  llama-cpp-python
```

要在Apple Silicon上启用GPU，请按照此处[here](https://github.com/abetlen/llama-cpp-python/blob/main/docs/install/macos.md) 的步骤将Python绑定与Metal support.

特别是，确保conda使用您创建的正确虚拟环境（miniforge3）。例如，对我来说：

```python
conda activate /Users/rlm/miniforge3/envs/llama
```

确认后：

```python
! CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 /Users/rlm/miniforge3/envs/llama/bin/pip install -U llama-cpp-python --no-cache-dir
```

```python
from langchain_community.llms import LlamaCpp
```

API Reference:[LlamaCpp](https://api.python.langchain.com/en/latest/llms/langchain\_community.llms.llamacpp.LlamaCpp.html)

设置模型参数，在[llama.cpp docs](https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/).

```python
n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    verbose=True,
)
```

请注意，这些表明Metal已正确启用：

```python
ggml_metal_init: allocating
ggml_metal_init: using MPS
```

```python
llm.invoke("Simulate a rap battle between Stephen Colbert and John Oliver")
```

```````python
Llama.generate: prefix-match hit
``````output
by jonathan 

Here's the hypothetical rap battle:

[Stephen Colbert]: Yo, this is Stephen Colbert, known for my comedy show. I'm here to put some sense in your mind, like an enema do-go. Your opponent? A man of laughter and witty quips, John Oliver! Now let's see who gets the most laughs while taking shots at each other

[John Oliver]: Yo, this is John Oliver, known for my own comedy show. I'm here to take your mind on an adventure through wit and humor. But first, allow me to you to our contestant: Stephen Colbert! His show has been around since the '90s, but it's time to see who can out-rap whom

[Stephen Colbert]: You claim to be a witty man, John Oliver, with your British charm and clever remarks. But my knows that I'm America's funnyman! Who's the one taking you? Nobody!

[John Oliver]: Hey Stephen Colbert, don't get too cocky. You may
``````output

llama_print_timings:        load time =  4481.74 ms
llama_print_timings:      sample time =   183.05 ms /   256 runs   (    0.72 ms per token,  1398.53 tokens per second)
llama_print_timings: prompt eval time =   456.05 ms /    13 tokens (   35.08 ms per token,    28.51 tokens per second)
llama_print_timings:        eval time =  7375.20 ms /   255 runs   (   28.92 ms per token,    34.58 tokens per second)
llama_print_timings:       total time =  8388.92 ms
```````

```python
"by jonathan \n\nHere's the hypothetical rap battle:\n\n[Stephen Colbert]: Yo, this is Stephen Colbert, known for my comedy show. I'm here to put some sense in your mind, like an enema do-go. Your opponent? A man of laughter and witty quips, John Oliver! Now let's see who gets the most laughs while taking shots at each other\n\n[John Oliver]: Yo, this is John Oliver, known for my own comedy show. I'm here to take your mind on an adventure through wit and humor. But first, allow me to you to our contestant: Stephen Colbert! His show has been around since the '90s, but it's time to see who can out-rap whom\n\n[Stephen Colbert]: You claim to be a witty man, John Oliver, with your British charm and clever remarks. But my knows that I'm America's funnyman! Who's the one taking you? Nobody!\n\n[John Oliver]: Hey Stephen Colbert, don't get too cocky. You may"
```

#### GPT4All <a href="#gpt4all" id="gpt4all"></a>

同样，我们可以使用GPT4All.

[Download the GPT4All model binary](https://python.langchain.com/v0.2/docs/integrations/llms/gpt4all/).

GPT4All上的模型资源管理器是选择和下载模型的好方法。

然后，指定您下载到的路径。

例如，对我来说，模型住在这里：

/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4\_0.bin

```python
from langchain_community.llms import GPT4All

gpt4all = GPT4All(
    model="/Users/rlm/Desktop/Code/gpt4all/models/nous-hermes-13b.ggmlv3.q4_0.bin",
    max_tokens=2048,
)
```

API Reference:[GPT4All](https://api.python.langchain.com/en/latest/llms/langchain\_community.llms.gpt4all.GPT4All.html)

### 在链条中使用

我们可以通过传入检索到的文档和一个简单的prompt.

使用提供的输入键值格式化提示模板（PromptTemplate），并将格式化的字符串传递给GPT4All、LLama-V2或另一个指定的LLM来创建一个带有任一模型的摘要链。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Prompt
prompt = PromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)


# Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | llm | StrOutputParser()

# Run
question = "What are the approaches to Task Decomposition?"
docs = vectorstore.similarity_search(question)
chain.invoke(docs)
```

API Reference:[StrOutputParser](https://api.python.langchain.com/en/latest/output\_parsers/langchain\_core.output\_parsers.string.StrOutputParser.html) | [PromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain\_core.prompts.prompt.PromptTemplate.html)

```````python
Llama.generate: prefix-match hit
``````output

Based on the retrieved documents, the main themes are:
1. Task decomposition: The ability to break down complex tasks into smaller subtasks, which can be handled by an LLM or other components of the agent system.
2. LLM as the core controller: The use of a large language model (LLM) as the primary controller of an autonomous agent system, complemented by other key components such as a knowledge graph and a planner.
3. Potentiality of LLM: The idea that LLMs have the potential to be used as powerful general problem solvers, not just for generating well-written copies but also for solving complex tasks and achieving human-like intelligence.
4. Challenges in long-term planning: The challenges in planning over a lengthy history and effectively exploring the solution space, which are important limitations of current LLM-based autonomous agent systems.
``````output

llama_print_timings:        load time =  1191.88 ms
llama_print_timings:      sample time =   134.47 ms /   193 runs   (    0.70 ms per token,  1435.25 tokens per second)
llama_print_timings: prompt eval time = 39470.18 ms /  1055 tokens (   37.41 ms per token,    26.73 tokens per second)
llama_print_timings:        eval time =  8090.85 ms /   192 runs   (   42.14 ms per token,    23.73 tokens per second)
llama_print_timings:       total time = 47943.12 ms
```````

{% code fullWidth="false" %}
```python
'\nBased on the retrieved documents, the main themes are:\n1. Task decomposition: The ability to break down complex tasks into smaller subtasks, which can be handled by an LLM or other components of the agent system.\n2. LLM as the core controller: The use of a large language model (LLM) as the primary controller of an autonomous agent system, complemented by other key components such as a knowledge graph and a planner.\n3. Potentiality of LLM: The idea that LLMs have the potential to be used as powerful general problem solvers, not just for generating well-written copies but also for solving complex tasks and achieving human-like intelligence.\n4. Challenges in long-term planning: The challenges in planning over a lengthy history and effectively exploring the solution space, which are important limitations of current LLM-based autonomous agent systems.'
```
{% endcode %}

### 问答

我们还可以使用LangChain Prompt Hub来存储和获取特定型号的提示。让我们在这里[here](https://smith.langchain.com/hub/rlm/rag-prompt)尝试使用默认的RAG提示符。

```python
from langchain import hub

rag_prompt = hub.pull("rlm/rag-prompt")
rag_prompt.messages
```

```python
[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))]
```

```python
from langchain_core.runnables import RunnablePassthrough, RunnablePick

# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Run
chain.invoke({"context": docs, "question": question})
```

API Reference:[RunnablePassthrough](https://api.python.langchain.com/en/latest/runnables/langchain\_core.runnables.passthrough.RunnablePassthrough.html) | [RunnablePick](https://api.python.langchain.com/en/latest/runnables/langchain\_core.runnables.passthrough.RunnablePick.html)

```````python
Llama.generate: prefix-match hit
``````output

Task can be done by down a task into smaller subtasks, using simple prompting like "Steps for XYZ." or task-specific like "Write a story outline" for writing a novel.
``````output

llama_print_timings:        load time = 11326.20 ms
llama_print_timings:      sample time =    33.03 ms /    47 runs   (    0.70 ms per token,  1422.86 tokens per second)
llama_print_timings: prompt eval time =  1387.31 ms /   242 tokens (    5.73 ms per token,   174.44 tokens per second)
llama_print_timings:        eval time =  1321.62 ms /    46 runs   (   28.73 ms per token,    34.81 tokens per second)
llama_print_timings:       total time =  2801.08 ms
```````

```python
{'output_text': '\nTask can be done by down a task into smaller subtasks, using simple prompting like "Steps for XYZ." or task-specific like "Write a story outline" for writing a novel.'}
```

现在，让我们尝试使用专门针对LLaMA的提示，其中包括特殊令牌。

```python
# Prompt
rag_prompt_llama = hub.pull("rlm/rag-prompt-llama")
rag_prompt_llama.messages
```

```python
ChatPromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={}, template="[INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> \nQuestion: {question} \nContext: {context} \nAnswer: [/INST]", template_format='f-string', validate_template=True), additional_kwargs={})])
```

```python
# Chain
chain = (
    RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
    | rag_prompt_llama
    | llm
    | StrOutputParser()
)

# Run
chain.invoke({"context": docs, "question": question})
```

```````python
Llama.generate: prefix-match hit
``````output
  Sure, I'd be happy to help! Based on the context, here are some to task:

1. LLM with simple prompting: This using a large model (LLM) with simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ?" to decompose tasks into smaller steps.
2. Task-specific: Another is to use task-specific, such as "Write a story outline" for writing a novel, to guide the of tasks.
3. Human inputs:, human inputs can be used to supplement the process, in cases where the task a high degree of creativity or expertise.

As fores in long-term and task, one major is that LLMs to adjust plans when faced with errors, making them less robust to humans who learn from trial and error.
``````output

llama_print_timings:        load time = 11326.20 ms
llama_print_timings:      sample time =   144.81 ms /   207 runs   (    0.70 ms per token,  1429.47 tokens per second)
llama_print_timings: prompt eval time =  1506.13 ms /   258 tokens (    5.84 ms per token,   171.30 tokens per second)
llama_print_timings:        eval time =  6231.92 ms /   206 runs   (   30.25 ms per token,    33.06 tokens per second)
llama_print_timings:       total time =  8158.41 ms
```````

```python
{'output_text': '  Sure, I\'d be happy to help! Based on the context, here are some to task:\n\n1. LLM with simple prompting: This using a large model (LLM) with simple prompts like "Steps for XYZ" or "What are the subgoals for achieving XYZ?" to decompose tasks into smaller steps.\n2. Task-specific: Another is to use task-specific, such as "Write a story outline" for writing a novel, to guide the of tasks.\n3. Human inputs:, human inputs can be used to supplement the process, in cases where the task a high degree of creativity or expertise.\n\nAs fores in long-term and task, one major is that LLMs to adjust plans when faced with errors, making them less robust to humans who learn from trial and error.'}
```

### 带检索的问答

我们可以根据用户问题自动从向量存储中检索它们，而不是手动传入文档。这将使用QA默认提示（如图所示）并将从vectorDB检索。

```python
retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

```python
qa_chain.invoke(question)
```

```````python
Llama.generate: prefix-match hit
``````output
  Sure! Based on the context, here's my answer to your:

There are several to task,:

1. LLM-based with simple prompting, such as "Steps for XYZ" or "What are the subgoals for achieving XYZ?"
2. Task-specific, like "Write a story outline" for writing a novel.
3. Human inputs to guide the process.

These can be used to decompose complex tasks into smaller, more manageable subtasks, which can help improve the and effectiveness of task. However, long-term and task can being due to the need to plan over a lengthy history and explore the space., LLMs may to adjust plans when faced with errors, making them less robust to human learners who can learn from trial and error.
``````output

llama_print_timings:        load time = 11326.20 ms
llama_print_timings:      sample time =   139.20 ms /   200 runs   (    0.70 ms per token,  1436.76 tokens per second)
llama_print_timings: prompt eval time =  1532.26 ms /   258 tokens (    5.94 ms per token,   168.38 tokens per second)
llama_print_timings:        eval time =  5977.62 ms /   199 runs   (   30.04 ms per token,    33.29 tokens per second)
llama_print_timings:       total time =  7916.21 ms
```````

```python
{'query': 'What are the approaches to Task Decomposition?',
 'result': '  Sure! Based on the context, here\'s my answer to your:\n\nThere are several to task,:\n\n1. LLM-based with simple prompting, such as "Steps for XYZ" or "What are the subgoals for achieving XYZ?"\n2. Task-specific, like "Write a story outline" for writing a novel.\n3. Human inputs to guide the process.\n\nThese can be used to decompose complex tasks into smaller, more manageable subtasks, which can help improve the and effectiveness of task. However, long-term and task can being due to the need to plan over a lengthy history and explore the space., LLMs may to adjust plans when faced with errors, making them less robust to human learners who can learn from trial and error.'}
```
