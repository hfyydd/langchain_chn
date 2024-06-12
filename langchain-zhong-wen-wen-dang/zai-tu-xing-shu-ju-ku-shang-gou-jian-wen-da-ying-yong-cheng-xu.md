# 在图形数据库上构建问答应用程序

在本指南中，我们将介绍在图数据库上创建问答链的基本方法。这些系统将使我们能够针对图数据库中的数据提出问题，并获得自然语言的答案。

⚠️ 安全提示 ⚠️

构建图数据库的问答系统需要执行由模型生成的图查询。这其中存在固有风险。请确保数据库连接权限始终根据链/代理的需要尽可能缩小范围。这样可以减轻但不能消除构建模型驱动系统的风险。有关一般安全最佳实践的更多信息，请参见此处[see here](https://python.langchain.com/v0.2/docs/security/)。

### 架构

从高层次来看，大多数图链的步骤如下：

1. 将问题转换为图数据库查询：模型将用户输入转换为图数据库查询（例如，Cypher 查询）。
2. 执行图数据库查询：执行图数据库查询。
3. 回答问题：模型使用查询结果响应用户输入。

<figure><img src="https://python.langchain.com/v0.2/assets/images/graph_usecase-34d891523e6284bb6230b38c5f8392e5.png" alt=""><figcaption></figcaption></figure>

### Setup <a href="#setup" id="setup"></a>

首先，获取所需的包并设置环境变量。在这个例子中，我们将使用Neo4j图形数据库。

```python
%pip install --upgrade --quiet  langchain langchain-community langchain-openai neo4j
```

在本指南中，我们默认使用OpenAI模型。

```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Uncomment the below to use LangSmith. Not required.
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
```

接下来，我们需要定义Neo4j凭据。按照以下安装步骤设置Neo4j数据库。

```python
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
```

下面的示例将创建与Neo4j数据库的连接，并将使用有关电影及其演员的示例数据填充它。

```python
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

# Import movie information

movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

graph.query(movies_query)
```

API Reference:[Neo4jGraph](https://api.python.langchain.com/en/latest/graphs/langchain\_community.graphs.neo4j\_graph.Neo4jGraph.html)

### 图模式

为了让大型语言模型（LLM）能够生成 Cypher 语句，它需要了解图的架构信息。当你实例化一个图对象时，它会检索图架构的信息。如果你后来对图进行了任何更改，可以运行 `refresh_schema` 方法来刷新架构信息。

```python
graph.refresh_schema()
print(graph.schema)
```

```python
Node properties are the following:
Movie {imdbRating: FLOAT, id: STRING, released: DATE, title: STRING},Person {name: STRING},Genre {name: STRING},Chunk {id: STRING, question: STRING, query: STRING, text: STRING, embedding: LIST}
Relationship properties are the following:

The relationships are the following:
(:Movie)-[:IN_GENRE]->(:Genre),(:Person)-[:DIRECTED]->(:Movie),(:Person)-[:ACTED_IN]->(:Movie)
```

太好了！我们有一个可以查询的图形数据库。现在让我们尝试将其连接到LLM。

### 链

让我们使用一个简单的链，它接受一个问题，将其转换为Cypher查询，执行查询，并使用结果来回答原始问题。

<figure><img src="https://python.langchain.com/v0.2/assets/images/graph_chain-6379941793e0fa985e51e4bda0329403.webp" alt=""><figcaption></figcaption></figure>

LangChain为此工作流程提供了一个内置链，旨在与Neo4j一起使用：[GraphCypherQAChain](https://python.langchain.com/v0.2/docs/integrations/graphs/neo4j\_cypher/)

```python
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)
response = chain.invoke({"query": "What was the cast of the Casino?"})
response
```

API Reference:[GraphCypherQAChain](https://api.python.langchain.com/en/latest/chains/langchain\_community.chains.graph\_qa.cypher.GraphCypherQAChain.html) | [ChatOpenAI](https://api.python.langchain.com/en/latest/chat\_models/langchain\_openai.chat\_models.base.ChatOpenAI.html)

```python


[1m> Entering new GraphCypherQAChain chain...[0m
Generated Cypher:
[32;1m[1;3mMATCH (:Movie {title: "Casino"})<-[:ACTED_IN]-(actor:Person)
RETURN actor.name[0m
Full Context:
[32;1m[1;3m[{'actor.name': 'Joe Pesci'}, {'actor.name': 'Robert De Niro'}, {'actor.name': 'Sharon Stone'}, {'actor.name': 'James Woods'}][0m

[1m> Finished chain.[0m
```

```python
{'query': 'What was the cast of the Casino?',
 'result': 'The cast of Casino included Joe Pesci, Robert De Niro, Sharon Stone, and James Woods.'}
```

### 验证关系方向

LLM可能会与生成的Cypher语句中的关系方向作斗争。由于图模式是预定义的，我们可以使用validate\_cypher参数验证和可选地更正生成的Cypher语句中的关系方向。

```python
chain = GraphCypherQAChain.from_llm(
    graph=graph, llm=llm, verbose=True, validate_cypher=True
)
response = chain.invoke({"query": "What was the cast of the Casino?"})
response
```

```python


[1m> Entering new GraphCypherQAChain chain...[0m
Generated Cypher:
[32;1m[1;3mMATCH (:Movie {title: "Casino"})<-[:ACTED_IN]-(actor:Person)
RETURN actor.name[0m
Full Context:
[32;1m[1;3m[{'actor.name': 'Joe Pesci'}, {'actor.name': 'Robert De Niro'}, {'actor.name': 'Sharon Stone'}, {'actor.name': 'James Woods'}][0m

[1m> Finished chain.[0m
```

```python
{'query': 'What was the cast of the Casino?',
 'result': 'The cast of Casino included Joe Pesci, Robert De Niro, Sharon Stone, and James Woods.'}
```

#### 后续步骤[​](https://python.langchain.com/v0.2/docs/tutorials/graph/#next-steps) <a href="#next-steps" id="next-steps"></a>

对于更复杂的查询生成，我们可能希望创建少量提示或添加查询检查步骤。对于像这样的高级技术和更多检查：

* [Prompting strategies](https://python.langchain.com/v0.2/docs/how\_to/graph\_prompting/): 先进的快速工程技术。
* [Mapping values](https://python.langchain.com/v0.2/docs/how\_to/graph\_mapping/): 将值从问题映射到数据库的技术。
* [Semantic layer](https://python.langchain.com/v0.2/docs/how\_to/graph\_semantic/): 实现语义层的技术。
* [Constructing graphs](https://python.langchain.com/v0.2/docs/how\_to/graph\_constructing/): 构建知识图谱的技术。
