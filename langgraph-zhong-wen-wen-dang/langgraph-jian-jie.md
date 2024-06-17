# LangGraph简介

在本教程中，我们将使用 LangGraph 构建一个支持聊天机器人，它可以：

1. 通过搜索网络回答常见问题
2. 在对话过程中保持状态
3. 将复杂的查询转交给人工审核
4. 使用自定义状态来控制其行为
5. 回溯并探索不同的对话路径

我们将从一个基础的聊天机器人开始，逐步添加更复杂的功能，并在此过程中介绍 LangGraph 的关键概念。

### Setup <a href="#setup" id="setup"></a>

首先，安装所需的包：

```python
%%capture --no-stderr
%pip install -U langgraph langsmith

# Used for this tutorial; not a requirement for LangGraph
%pip install -U langchain_anthropic
```

接下来，设置您的API密钥：

```python
import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
```

（鼓励）朗史密斯让人们更容易看到“引擎盖下”发生了什么

```python
_set_env("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Tutorial"
```
