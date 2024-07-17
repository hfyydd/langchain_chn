# 如何创建子图

对于更复杂的系统，子图是一个有用的设计原则。子图允许您在图的不同部分创建和管理不同的状态。这允许您构建多代理团队之类的东西，其中每个团队都可以跟踪自己的单独状态。

<figure><img src="../.gitbook/assets/下载 (4) (1).png" alt=""><figcaption></figcaption></figure>

## 简单的例子

让我们考虑一个玩具示例：我有一个系统，它接受日志并执行两个独立的子任务。首先，它会总结它们。其次，它会总结日志中捕获的任何故障模式。我想在两个不同的子图中执行这两个操作。

最需要识别的是图之间的信息传递。入口图是父图，两个子图中的每一个都被定义为入口图Entry Graph中的节点。两个子图都继承父入口图Entry Graph的状态；我可以通过在子图状态中指定它来访问每个子图中的文档（见图表）。每个子图都可以有自己的私有状态。我想要传播回父入口图Entry Graph（用于最终报告）的任何值都需要在我的入口图Entry Graph状态中定义（例如，摘要报告`summary report` 和失败报告failure report）。

<figure><img src="../.gitbook/assets/下载 (5) (1).png" alt=""><figcaption></figcaption></figure>

```python
from operator import add
from typing import List, TypedDict, Optional, Annotated, Dict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# The structure of the logs
class Logs(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str    
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]  

# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    docs: List[Logs]
    failures: List[Logs]
    fa_summary: str

def get_failures(state):
    docs = state['docs']
    failures = [doc for doc in docs if "grade" in doc]
    return {"failures": failures}

def generate_summary(state):
    failures = state['failures']
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation." 
    return {"fa_summary": fa_summary}

fa_builder = StateGraph(FailureAnalysisState)
fa_builder.add_node("get_failures", get_failures)
fa_builder.add_node("generate_summary", generate_summary)
fa_builder.set_entry_point("get_failures")
fa_builder.add_edge("get_failures","generate_summary")
fa_builder.set_finish_point("generate_summary")

# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    docs: List[Logs]
    qs_summary: str
    report: str

def generate_summary(state):
    docs = state['docs']
    # Add fxn: summary = summarize(docs)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store." 
    return {"qs_summary": summary}

def send_to_slack(state):
    qs_summary = state['qs_summary']
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}

def format_report_for_slack(state):
    report = state['report']
    # Add fxn: formatted_report = report_format(report)
    formatted_report = "foo bar"
    return {"report": formatted_report}

qs_builder = StateGraph(QuestionSummarizationState)
qs_builder.add_node("generate_summary", generate_summary)
qs_builder.add_node("send_to_slack", send_to_slack)
qs_builder.add_node("format_report_for_slack", format_report_for_slack)
qs_builder.set_entry_point("generate_summary")
qs_builder.add_edge("generate_summary", "send_to_slack")
qs_builder.add_edge("send_to_slack", "format_report_for_slack")
qs_builder.add_edge("format_report_for_slack", END)
```

请注意，每个子图都有自己的状态、QuestionSummarizationState和FailureAnalysisState。

定义每个子图后，我们将所有内容放在一起。

```python
# Dummy logs
question_answer = Logs(
    id="1",
    question="How can I import ChatOllama?",
    answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
)

question_answer_feedback = Logs(
    id="2",
    question="How can I use Chroma vector store?",
    answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
    grade=0,
    grader="Document Relevance Recall",
    feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
)

# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: Annotated[List[Dict], add]
    docs: Annotated[List[Logs], add] # This will be used in sub-graphs
    fa_summary: str # This will be generated in the FA sub-graph
    report: str # This will be generated in the QS sub-graph

def convert_logs_to_docs(state):
    # Get logs
    raw_logs = state['raw_logs']
    docs = [question_answer,question_answer_feedback]
    return {"docs": docs} 

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("convert_logs_to_docs", convert_logs_to_docs)
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())

entry_builder.set_entry_point("convert_logs_to_docs")
entry_builder.add_edge("convert_logs_to_docs", "failure_analysis")
entry_builder.add_edge("convert_logs_to_docs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

graph = entry_builder.compile()

from IPython.display import Image, display
# Setting xray to 1 will show the internal structure of the nested graph
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

<figure><img src="../.gitbook/assets/下载 (1) (4).jpeg" alt=""><figcaption></figcaption></figure>

```python
raw_logs = [{"foo":"bar"},{"foo":"baz"}]
graph.invoke({"raw_logs": raw_logs}, debug=False)
```

```
{'raw_logs': [{'foo': 'bar'}, {'foo': 'baz'}],
 'docs': [{'id': '1',
   'question': 'How can I import ChatOllama?',
   'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
  {'id': '2',
   'question': 'How can I use Chroma vector store?',
   'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
   'grade': 0,
   'grader': 'Document Relevance Recall',
   'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'},
  {'id': '1',
   'question': 'How can I import ChatOllama?',
   'answer': "To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'"},
  {'id': '2',
   'question': 'How can I use Chroma vector store?',
   'answer': 'To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).',
   'grade': 0,
   'grader': 'Document Relevance Recall',
   'feedback': 'The retrieved documents discuss vector stores in general, but not Chroma specifically'}],
 'fa_summary': 'Poor quality retrieval of Chroma documentation.',
 'report': 'foo bar'}
```

## 自定义reducer功能来管理状态

现在，让我们突出显示当我们在多个子图中使用相同状态时可能存在的绊脚石。

我们将创建两个图：一个带有几个节点的父图和一个作为父图中的节点添加的子图。

我们为我们的状态定义了一个自定义的reducer函数。

```python
from typing import Annotated

from typing_extensions import TypedDict

def reduce_list(left: list | None, right: list | None) -> list:
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class ChildState(TypedDict):
    name: str
    path: Annotated[list[str], reduce_list]


class ParentState(TypedDict):
    name: str
    path: Annotated[list[str], reduce_list]


child_builder = StateGraph(ChildState)

child_builder.add_node("child_start", lambda state: {"path": ["child_start"]})
child_builder.add_edge(START, "child_start")
child_builder.add_node("child_middle", lambda state: {"path": ["child_middle"]})
child_builder.add_node("child_end", lambda state: {"path": ["child_end"]})
child_builder.add_edge("child_start", "child_middle")
child_builder.add_edge("child_middle", "child_end")
child_builder.add_edge("child_end", END)

builder = StateGraph(ParentState)

builder.add_node("grandparent", lambda state: {"path": ["grandparent"]})
builder.add_edge(START, "grandparent")
builder.add_node("parent", lambda state: {"path": ["parent"]})
builder.add_node("child", child_builder.compile())
builder.add_node("sibling", lambda state: {"path": ["sibling"]})
builder.add_node("fin", lambda state: {"path": ["fin"]})

# Add connections
builder.add_edge("grandparent", "parent")
builder.add_edge("parent", "child")
builder.add_edge("parent", "sibling")
builder.add_edge("child", "fin")
builder.add_edge("sibling", "fin")
builder.add_edge("fin", END)
graph = builder.compile()
```

```python
from IPython.display import Image, display

# Setting xray to 1 will show the internal structure of the nested graph
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

<figure><img src="../.gitbook/assets/下载 (2) (3).jpeg" alt=""><figcaption></figcaption></figure>

```python
graph.invoke({"name": "test"}, debug=True)
```

```
[0:tasks] Starting step 0 with 1 task:
- __start__ -> {'name': 'test'}
[0:writes] Finished step 0 with writes to 1 channel:
- name -> 'test'
[0:checkpoint] State at the end of step 0:
{'name': 'test', 'path': []}
[1:tasks] Starting step 1 with 1 task:
- grandparent -> {'name': 'test', 'path': []}
[1:writes] Finished step 1 with writes to 1 channel:
- path -> ['grandparent']
[1:checkpoint] State at the end of step 1:
{'name': 'test', 'path': ['grandparent']}
[2:tasks] Starting step 2 with 1 task:
- parent -> {'name': 'test', 'path': ['grandparent']}
[2:writes] Finished step 2 with writes to 1 channel:
- path -> ['parent']
[2:checkpoint] State at the end of step 2:
{'name': 'test', 'path': ['grandparent', 'parent']}
[3:tasks] Starting step 3 with 2 tasks:
- child -> {'name': 'test', 'path': ['grandparent', 'parent']}
- sibling -> {'name': 'test', 'path': ['grandparent', 'parent']}
[3:writes] Finished step 3 with writes to 2 channels:
- name -> 'test'
- path -> ['grandparent', 'parent', 'child_start', 'child_middle', 'child_end'], ['sibling']
[3:checkpoint] State at the end of step 3:
{'name': 'test',
 'path': ['grandparent',
          'parent',
          'grandparent',
          'parent',
          'child_start',
          'child_middle',
          'child_end',
          'sibling']}
[4:tasks] Starting step 4 with 1 task:
- fin -> {'name': 'test',
 'path': ['grandparent',
          'parent',
          'grandparent',
          'parent',
          'child_start',
          'child_middle',
          'child_end',
          'sibling']}
[4:writes] Finished step 4 with writes to 1 channel:
- path -> ['fin']
[4:checkpoint] State at the end of step 4:
{'name': 'test',
 'path': ['grandparent',
          'parent',
          'grandparent',
          'parent',
          'child_start',
          'child_middle',
          'child_end',
          'sibling',
          'fin']}
{'name': 'test',
 'path': ['grandparent',
  'parent',
  'grandparent',
  'parent',
  'child_start',
  'child_middle',
  'child_end',
  'sibling',
  'fin']}
```

请注意，这里的\["grandparent", "parent"]序列是重复的！

这是因为我们的子状态已经收到了完整的父状态，并在终止后返回完整的父状态。

为了避免重复或状态冲突，您通常会执行以下一项或多项操作：

1.在减速器函数中处理重复项。

2.从python函数中调用子图。在该函数中，根据需要处理状态。

3.更新子图键以避免冲突。

然而，您仍然需要确保父级可以解释输出。让我们使用技术（1）重新实现图，并为列表中的每个值添加唯一ID。这就是MessageGraph中所做的。

```python
import uuid


def reduce_list(left: list | None, right: list | None) -> list:
    """Append the right-hand list, replacing any elements with the same id in the left-hand list."""
    if not left:
        left = []
    if not right:
        right = []
    left_, right_ = [], []
    for orig, new in [(left, left_), (right, right_)]:
        for val in orig:
            if not isinstance(val, dict):
                val = {"val": val}
            if "id" not in val:
                val["id"] = str(uuid.uuid4())
            new.append(val)
    # Merge the two lists
    left_idx_by_id = {val["id"]: i for i, val in enumerate(left_)}
    merged = left_.copy()
    for val in right_:
        if (existing_idx := left_idx_by_id.get(val["id"])) is not None:
            merged[existing_idx] = val
        else:
            merged.append(val)
    return merged


class ChildState(TypedDict):
    name: str
    path: Annotated[list[str], reduce_list]


class ParentState(TypedDict):
    name: str
    path: Annotated[list[str], reduce_list]
```

```python
from IPython.display import Image, display

# Setting xray to 1 will show the internal structure of the nested graph
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
```

<figure><img src="../.gitbook/assets/下载 (3) (2).jpeg" alt=""><figcaption></figcaption></figure>

```python
graph.invoke({"name": "test"}, debug=True)
```

```
[0:tasks] Starting step 0 with 1 task:
- __start__ -> {'name': 'test'}
[0:writes] Finished step 0 with writes to 1 channel:
- name -> 'test'
[0:checkpoint] State at the end of step 0:
{'name': 'test', 'path': []}
[1:tasks] Starting step 1 with 1 task:
- grandparent -> {'name': 'test', 'path': []}
[1:writes] Finished step 1 with writes to 1 channel:
- path -> ['grandparent']
[1:checkpoint] State at the end of step 1:
{'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'}]}
[2:tasks] Starting step 2 with 1 task:
- parent -> {'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'}]}
[2:writes] Finished step 2 with writes to 1 channel:
- path -> ['parent']
[2:checkpoint] State at the end of step 2:
{'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
          {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'}]}
[3:tasks] Starting step 3 with 2 tasks:
- child -> {'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
          {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'}]}
- sibling -> {'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
          {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'}]}
[3:writes] Finished step 3 with writes to 2 channels:
- name -> 'test'
- path -> [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
 {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'},
 {'id': 'd1c1bab0-6e19-4846-a470-e9cc2eb85088', 'val': 'child_start'},
 {'id': 'e0fcb647-1e9e-4ae4-b560-0046515d5783', 'val': 'child_middle'},
 {'id': '669dd810-360f-4694-a9f3-49597f23376a', 'val': 'child_end'}], ['sibling']
[3:checkpoint] State at the end of step 3:
{'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
          {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'},
          {'id': 'd1c1bab0-6e19-4846-a470-e9cc2eb85088', 'val': 'child_start'},
          {'id': 'e0fcb647-1e9e-4ae4-b560-0046515d5783', 'val': 'child_middle'},
          {'id': '669dd810-360f-4694-a9f3-49597f23376a', 'val': 'child_end'},
          {'id': '137dbc2f-b33c-4ea4-8b04-a62215ba9718', 'val': 'sibling'}]}
[4:tasks] Starting step 4 with 1 task:
- fin -> {'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
          {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'},
          {'id': 'd1c1bab0-6e19-4846-a470-e9cc2eb85088', 'val': 'child_start'},
          {'id': 'e0fcb647-1e9e-4ae4-b560-0046515d5783', 'val': 'child_middle'},
          {'id': '669dd810-360f-4694-a9f3-49597f23376a', 'val': 'child_end'},
          {'id': '137dbc2f-b33c-4ea4-8b04-a62215ba9718', 'val': 'sibling'}]}
[4:writes] Finished step 4 with writes to 1 channel:
- path -> ['fin']
[4:checkpoint] State at the end of step 4:
{'name': 'test',
 'path': [{'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49', 'val': 'grandparent'},
          {'id': '2a6f0263-3949-4e47-a210-57f817e6097d', 'val': 'parent'},
          {'id': 'd1c1bab0-6e19-4846-a470-e9cc2eb85088', 'val': 'child_start'},
          {'id': 'e0fcb647-1e9e-4ae4-b560-0046515d5783', 'val': 'child_middle'},
          {'id': '669dd810-360f-4694-a9f3-49597f23376a', 'val': 'child_end'},
          {'id': '137dbc2f-b33c-4ea4-8b04-a62215ba9718', 'val': 'sibling'},
          {'id': 'a4328c5f-845a-43de-b3d7-53a39208e316', 'val': 'fin'}]}
```

```
{'name': 'test',
 'path': [{'val': 'grandparent', 'id': '79a81f03-d16d-4d12-94a6-4ba29fc9ce49'},
  {'val': 'parent', 'id': '2a6f0263-3949-4e47-a210-57f817e6097d'},
  {'val': 'child_start', 'id': 'd1c1bab0-6e19-4846-a470-e9cc2eb85088'},
  {'val': 'child_middle', 'id': 'e0fcb647-1e9e-4ae4-b560-0046515d5783'},
  {'val': 'child_end', 'id': '669dd810-360f-4694-a9f3-49597f23376a'},
  {'val': 'sibling', 'id': '137dbc2f-b33c-4ea4-8b04-a62215ba9718'},
  {'val': 'fin', 'id': 'a4328c5f-845a-43de-b3d7-53a39208e316'}]}
```
