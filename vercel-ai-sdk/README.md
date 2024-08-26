# Vercel AI SDK

Vercel AI SDK是TypeScript工具包，旨在帮助开发人员使用React、Next. js、Vue、Svelte、Node.js等构建AI驱动的应用程序。

为什么使用 Vercel AI SDK?

将大型语言模型(LLMs)集成到应用程序中是复杂的,并且高度依赖于您使用的特定模型提供商。Vercel AI SDK 抽象了不同模型提供商之间的差异,消除了构建聊天机器人的样板代码,并允许您超越文本输出,生成丰富的交互式组件。

• AI SDK Core: 用于生成文本、结构化对象和工具调用的统一 API。

• AI SDK UI: 一套框架无关的钩子,用于快速构建聊天和生成式用户界面。

• AI SDK RSC: 一个用 React Server Components (RSC) 流式传输生成式用户界面的库。

### 概览

Vercel AI SDK标准化了跨支持提供商集成人工智能（AI）模型。这使开发人员能够专注于构建出色的AI应用程序，而不是在技术细节上浪费时间。例如，以下是如何使用Vercel AI SDK生成具有各种模型的文本：

```javascript
// OpenAI
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"
const { text } = await generateText({
model: openai("gpt-4-turbo"),
prompt: "What is love?"
})
```

```
Love is a complex and multifaceted emotion that can be felt and expressed in many different ways. It involves deep affection, care, compassion, and connection towards another person or thing. Love can take on various forms such as romantic love, platonic love, familial love, or self-love.
```

为了有效利用AI SDK，熟悉以下概念会有所帮助：

### 生成式人工智能

**生成式人工智能**指的是基于统计学上的可能性来预测和生成各种类型输出(如文本、图像或音频)的模型,这些模型从训练数据中学习到的模式中提取信息。例如:

* 给定一张照片,生成式模型可以生成一个描述。
* 给定一个音频文件,生成式模型可以生成一个转录。
* 给定一段文本描述,生成式模型可以生成一张图像。

### **大型语言模型**

**大型语言模型(LLM)是一种主要专注于文本**的生成式模型的子集。LLM 接收一系列单词作为输入,旨在预测最有可能跟随的序列。它为潜在的下一个序列分配概率,然后选择一个。模型会继续生成序列,直到满足指定的停止标准。

LLM 通过对海量文本集合进行训练来学习,这意味着它们更适合某些用例而非其他。例如,在 GitHub 数据上训练的模型会特别擅长理解源代码中序列的概率。

然而,理解 LLM 的局限性至关重要。当被问及较少知道或缺失的信息时,比如个人亲属的生日,LLM 可能会"产生幻觉"或编造信息。考虑你需要的信息在模型中的表示程度是很重要的。

### **嵌入模型**

**嵌入模型**用于将复杂数据(如单词或图像)转换为密集向量(一系列数字)表示,称为嵌入。与生成式模型不同,嵌入模型不生成新的文本或数据。相反,它们提供实体之间语义和语法关系的表示,这些表示可以用作其他模型的输入或用于其他自然语言处理任务。



