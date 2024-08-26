# 供应商和模型

像 OpenAI 和 Anthropic 这样的公司(提供商)通过自己的 API 提供访问一系列具有不同优势和能力的大型语言模型(LLM)的服务。

每个提供商通常都有自己独特的与其模型交互的方法,这使得切换提供商的过程变得复杂,并增加了供应商锁定的风险。

为了解决这些挑战,Vercel AI SDK Core 提供了一种与 LLM 交互的标准化方法,通过语言模型规范来抽象提供商之间的差异。这种统一的接口允许您轻松地在提供商之间切换,同时对所有提供商使用相同的 API。

以下是 AI SDK 提供商架构的概述:

<figure><img src="https://sdk.vercel.ai/_next/image?url=%2Fimages%2Fai-sdk-diagram.png&#x26;w=828&#x26;q=75&#x26;dpl=dpl_7CutpkL18zogkLLtc2pDzd8d9bGX" alt=""><figcaption></figcaption></figure>

### [AI SDK Providers](https://sdk.vercel.ai/docs/foundations/providers-and-models#ai-sdk-providers) <a href="#ai-sdk-providers" id="ai-sdk-providers"></a>

Vercel AI SDK 配备了几个提供商，您可以使用它们与不同的语言模型进行交互：

* [OpenAI Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/openai) (`@ai-sdk/openai`)
* [Azure OpenAI Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/azure) (`@ai-sdk/azure`)
* [Anthropic Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/anthropic) (`@ai-sdk/anthropic`)
* [Amazon Bedrock Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/amazon-bedrock) (`@ai-sdk/amazon-bedrock`)
* [Google Generative AI Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/google-generative-ai) (`@ai-sdk/google`)
* [Google Vertex Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/google-vertex) (`@ai-sdk/google-vertex`)
* [Mistral Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/mistral) (`@ai-sdk/mistral`)
* [Cohere Provider](https://sdk.vercel.ai/providers/ai-sdk-providers/cohere) (`@ai-sdk/cohere`)

您也可以使用与 OpenAI 兼容的 OpenAI 提供商：

* [Groq](https://sdk.vercel.ai/providers/ai-sdk-providers/groq)
* [Perplexity](https://sdk.vercel.ai/providers/ai-sdk-providers/perplexity)
* [Fireworks](https://sdk.vercel.ai/providers/ai-sdk-providers/fireworks)

我们的语言模型规范已经发布为一个开源软件包，您可以使用它来创建自定义提供商。

开源社区已经创建了以下提供商：

* [LLamaCpp Provider](https://sdk.vercel.ai/providers/community-providers/llama-cpp) (`nnance/llamacpp-ai-provider` )
* [Ollama Provider](https://sdk.vercel.ai/providers/community-providers/ollama) (`sgomez/ollama-ai-provider`)
* [ChromeAI Provider](https://sdk.vercel.ai/providers/community-providers/chrome-ai) (`jeasonstudio/chrome-ai`)

\
