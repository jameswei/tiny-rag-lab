# tiny-rag-lab

[English](README.md) · [项目主页](https://jameswei.github.io/tiny-rag-lab/)

> 一个以学习为先、可检查的经典 RAG 实验室：通过易读 Python、丰富的浏览器 Studio、
> 交互式检索课程、直接的 CLI 实验、真实语料 trace 与中英双语学习指南理解 RAG。

`tiny-rag-lab` 让用户问题、文档语料、检索证据、打包后的上下文与带引用答案之间的
完整路径清晰可见、可以检查。易读的 Python 直接呈现 RAG 机制；丰富的浏览器 Studio
将中间产物转化为引导回放和动手实验；直接的 CLI 支持可重复检查。当某个概念需要更
安静、深入的阅读时，可搜索的中英双语学习指南会在实验旁打开。

Studio 还把检索栈变成一套实时课程：检查 BM25 逐词贡献、稠密余弦计算、NumPy 与
可选 Qdrant 中的同一组向量、混合 RRF 融合、交叉编码器排名移动，以及在 16 道已
审核真实语料问题上的双配置评估。整个过程不需要 LLM 服务商。

它是学习工具，而不是生产级 RAG 平台。项目优先选择可见的机制，而非框架魔法；先评估再优化；先分析失败再引入高级特性。

![Guided Learn 回放展示真实检索证据](website/assets/screenshots/guided-retrieval.jpg)

## 通过真实证据学习，而非玩具示例

本地可视化工作区是一级学习环境，而不是 CLI 外面的一层仪表盘。它让学习者可以在真实技术文档上，沿着一条完整的本地路径理解经典 RAG。

学习者不再从合成的单文档回放开始，而是从固定的 40 篇 Cloudflare 语料中，跟随四个不依赖服务商的已保存课程。每个课程都会保留连接各个 RAG 阶段的真实产物：源文档、文本块、查询向量、排序后的候选证据、选入与省略的上下文、提示词、答案和引用。在这一引导基线上，学习者还可以自行检索、构建索引、导入小型语料、比较可检查的 NumPy 索引与可选 Qdrant，并且只在已测试可用的服务商存在时使用 Live Ask。

因此，项目同时以两种互补方式发挥价值：通过易读源码和直接的 CLI 学习 RAG 机制；通过交互式、实验性的 Web 应用观察这些机制如何在真实证据上运行。

当某个阶段需要更深入的概念解释时，本地 Studio 还会同时提供中英双语的**学习指南**：
可搜索的长文阅读会在实验旁打开，无需跳转到 GitHub。学习指南用于辅助两种入口，而
不是引入另一套 RAG 体验。

这里的**经典 RAG**指一条可见、可检查的路径：从语料中检索证据，选择并打包为上下文，然后生成带引用、基于证据的答案。经过测试的 OpenAI 兼容服务商会为 Live Ask 补全生成步骤；这不会使项目变成 Agentic RAG 或多步 RAG。

## 一个可检查的核心，两种学习入口

Web 应用和 CLI 是学习同一套项目自有 RAG 核心的互补方式，而不是拥有不同机制的两个独立产品。两者都使用相同的文档、文本块、嵌入、检索结果、选入的上下文、提示词、引用和 trace。

- **本地可视化工作区：** 推荐的起点。它通过真实语料回放引导学习者，让中间产物易于检查，并支持亲手进行检索、索引、失败场景和服务商实验。
- **CLI：** 直接、紧凑且可脚本化的入口。它适合重复某个配置、比较结果、检查原始输出，并逐条命令跟随 RAG 机制。

## 这个实验室的独特之处

许多 RAG 示例止步于一次框架调用、一个聊天界面，或一条顺利完成的答案路径。`tiny-rag-lab` 将学习者真正需要理解和推理的概念连接起来：

- **实现与学习体验保持连接。** CLI 和 Web 工作区都呈现来自同一套项目自有 RAG 核心的产物，而不是讲解与可执行代码脱节的抽象流程图。
- **真实产物就是课程。** 引导回放展示一次结果背后的真实源文档、文本块、向量、排序候选、选入与省略的上下文、提示词、答案、引用和耗时。
- **从引导走向实验。** 学习者可以从稳定、不依赖服务商的真实语料回放开始，再调整检索方式、构建索引、上传小型语料，或连接服务商进行 Live Ask。
- **失败也是学习内容。** 评估、trace 检查和精心设计的失败场景，让学习者能够追问结果为何出现，而不只是确认是否返回了答案。

## 从本地可视化实验室开始

最快的学习方式是启动内置的本地 Studio：

```bash
cp .env.example .env
docker compose up --build
```

然后在浏览器打开 http://127.0.0.1:8000。学习指南位于
http://127.0.0.1:8000/docs/，也可以从实验室中的相关阶段直接打开。交互式 Studio
只在本地运行，不需要公共部署或注册账号，Live Ask 只会联系你自行配置的服务商。学习指南
同时发布在[项目主页](https://jameswei.github.io/tiny-rag-lab/)上，无需运行实验室即可阅读。

推荐的首次使用路径如下：

1. **Home → Start guided lesson：** 从固定的 40 篇 Cloudflare State & Coordination 文档中，选择四个已保存课程之一进行回放。
2. **Learn：** 逐步查看语料、文本块、查询嵌入向量、检索候选、选入上下文的证据、基于证据的答案和引用。
3. **Retrieval：** 通过六个实时模块，从词法与稠密检索机制一路学习 NumPy/Qdrant
   对比、混合融合、重排序，以及在 16 道已审核问题上的浏览器 A/B 评估。
4. **Explore：** 提出题库问题或自由问题，比较稠密检索、BM25 和混合检索，可选地
   重排更大的候选池，并检查返回的 trace。只有希望进行 Live Ask 生成时，才需要
   配置并测试 OpenAI 兼容的 LLM 服务商。
5. **Build & Inspect：** 使用内置语料或小型 Markdown/纯文本上传构建索引，然后检查文档、文本块、向量和来源信息。
6. **Failure Lab：** 对比精心设计的失败场景及其改进方案。

当你希望在更安静的阅读环境中理解相应概念时，可以从 Learn、Retrieval、Explore 或
Failure Lab 打开**阅读学习指南**。它会在新标签页打开，并保留当前实验状态。

界面提供英文和简体中文。内置语料内容、问题、已记录答案和引用会保留其原始语言。

### 实验室包含什么

- 四个不依赖 LLM 服务商、带完整已保存产物的 Guided Learn 回放课程。
- 六个实时 Retrieval 模块，覆盖词法与稠密评分、本地向量与可选 Qdrant、混合融合、
  交叉编码器重排序，以及对 16 道已审核 Cloudflare 问题的 A/B 评估。
- 固定的 Cloudflare 学习语料，以及可直接使用的结构化分块和固定字符分块 NumPy 索引。
- 内置 watsonxDocsQA 源数据；完成显式的后台索引构建后，可以使用全部 75 个题库问题。
- 不配置 LLM 服务商也可以进行纯检索探索；通过连接测试后，可使用任何 OpenAI 兼容 Chat Completions 服务进行 Live Ask。
- 支持小型自定义语料上传：最多 100 个 Markdown 或纯文本文件，总大小不超过 100 MiB。
- 默认使用 NumPy/文件索引；可选的本地 Qdrant 后端只改变存储和向量搜索的执行方式，不改变本项目要讲解的文本块、嵌入、检索、上下文、引用和 trace 概念。
- 精心设计的失败课程、原始产物检查、来源溯源、候选证据与上下文选择的对比，以及支持减少动画偏好的回放体验。

默认的 `full` 镜像包含固定版本的本地嵌入模型和交叉编码器重排序模型，只使用 CPU，
不需要 GPU 或 CUDA 运行时。若想体验更小的镜像：

```bash
LAB_IMAGE_VARIANT=slim docker compose up --build
```

在 slim 镜像中，Guided Learn 回放和 BM25 检索仍然可用。设置页分别提供嵌入模型与
重排序模型的显式下载。稠密/混合检索和索引构建需要嵌入模型；交叉编码器实验需要
重排序模型。

如需使用可选的 Qdrant 对比后端：

```bash
docker compose --profile qdrant up --build
```

## 经典 RAG 流程

```text
本地语料 -> 文档 -> 规范化文本 -> 文本块 -> 嵌入向量
-> 本地向量索引 -> 查询嵌入 -> 检索
-> 选出的上下文 -> 基于证据的提示词 -> 带引用的答案
```

项目使每个阶段都可检查：

- **索引：** 文档加载、文本规范化、固定字符/结构化/实验性语义分块、元数据、嵌入和本地索引。
- **检索：** 稠密余弦检索、BM25 关键词检索、基于倒数排名融合（RRF）的混合检索，以及可选的二次重排序。
- **生成：** 显式上下文预算、提示词组装、OpenAI 兼容的生成边界、引用，以及证据不足时的拒答。
- **评估与可观测性：** 检索指标、LLM-as-judge 答案指标、可回放 trace 和精心设计的失败诊断。

## CLI：直接而可脚本化的入口

CLI 是可视化工作区直接、可重复的伙伴。需要通过紧凑命令体验同一套机制时，可以使用：

```bash
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
rag index --corpus PATH --index-dir .tiny-rag/index --chunking-strategy structural
rag index --corpus PATH --index-dir .tiny-rag/index --chunking-strategy semantic --semantic-similarity-threshold 0.5

rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever dense
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever bm25
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever hybrid
rag retrieve "question text" --index-dir .tiny-rag/index --top-k 5 --retriever hybrid --reranker cross-encoder --rerank-top-n 20

rag ask "question text" --index-dir .tiny-rag/index --top-k 5
rag ask "question text" --index-dir .tiny-rag/index --context-budget 8192 --output-format json

rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --top-k 5 --retriever hybrid
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl --index-dir .tiny-rag/index --judge fake --generator fake

rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index
```

每个命令都有针对性的帮助：

```bash
uv run rag --help
uv run rag index --help
uv run rag retrieve --help
uv run rag ask --help
uv run rag eval --help
uv run rag diagnose --help
```

## 开发

```bash
uv sync --group dev
uv run pytest --tb=short -q
```

如需从源码运行两个浏览器界面，请在两个终端中分别执行：

```bash
npm --prefix learning_materials install
npm --prefix learning_materials run dev

npm --prefix web install
npm --prefix web run dev
```

React 开发服务器会将 `/docs` 代理到 `127.0.0.1:4173` 上的 VitePress 服务，与打包后
的同源路径保持一致。

如需为独立 CLI 准备 watsonxDocsQA 语料：

```bash
uv run python scripts/prepare_watsonx_docsqa.py --inspect
uv run python scripts/prepare_watsonx_docsqa.py --output-dir corpus/watsonx-docsqa
```

本地生成的语料和索引会被 Git 忽略：

```text
corpus/
.tiny-rag/
```

## 技术选择

- Python · `argparse` CLI · FastAPI · React + TypeScript · VitePress · Docker Compose
- 本地嵌入模型：`sentence-transformers/all-MiniLM-L6-v2`
- 默认索引：可检查的 NumPy 文件；可选本地 Qdrant 适配器
- 生成：OpenAI 兼容 Chat Completions API
- 离线测试：fake embedder + fake generator
- 不使用 LangChain、LlamaIndex 或 Haystack 来封装学习关键路径上的 RAG 机制

## 文档

- [学习指南](learning_materials/zh/learning-roadmap.md)：CLI 与可视化实验室的概念
  伴读材料（由 Studio 在本地 `/docs/` 提供）
- [项目提案](docs/proposal.md)：项目目的、理念和非目标
- [架构](docs/architecture.md)：RAG 概念平面和边界
- [文件结构](docs/file-structure.md)：仓库地图
