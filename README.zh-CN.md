# tiny-rag-lab

[English](README.md) · [项目主页](https://jameswei.github.io/tiny-rag-lab/)

> 一个以学习为先、可检查的经典 RAG 实验室：易读的 Python、带实时检索课程的浏览器
> Studio、直接的 CLI、真实语料 trace，以及中英双语学习指南。

`tiny-rag-lab` 让从提问到带引用答案的完整路径清晰可见、可以检查：文档语料、检索
证据、打包后的上下文，以及基于证据的答案。易读的 Python 直接呈现机制；浏览器
Studio 将其转化为引导回放和一套实时检索课程——BM25 逐词得分、稠密余弦计算、混合
RRF 融合、交叉编码器重排序，以及 NumPy 与可选 Qdrant 中的同一组向量；直接的 CLI
支持可重复检查。可搜索的中英双语学习指南会在实验旁打开，供更深入的阅读；这一切
都不需要 LLM 服务商——只有 Live Ask 生成才需要一个已测试的 OpenAI 兼容服务商。

它是学习工具，而不是生产级 RAG 平台：优先选择可见的机制而非框架魔法，先评估再
优化，先分析失败再引入高级特性。

![Guided Learn 回放展示真实检索证据](website/assets/screenshots/guided-retrieval.jpg)

## 一个核心，两种学习方式

Web Studio 和 CLI 是学习同一套项目自有 RAG 核心的互补方式，而不是拥有不同机制的
两个独立产品。两者使用相同的文档、文本块、嵌入、检索结果、上下文、提示词、引用
和 trace。

- **Studio（推荐起点）：** 引导式真实语料回放，每个中间产物都可检查，并支持亲手
  进行检索、索引、失败场景和服务商实验。
- **CLI：** 直接、可脚本化的入口——重复某个配置、比较结果、检查原始输出，逐条命
  令跟随机制。

学习者不再从合成的单文档演示开始，而是从固定的 40 篇 Cloudflare 语料中跟随四个不依赖
服务商的已保存课程；每个课程都保留连接各个 RAG 阶段的真实产物：源文档、文本块、
向量、排序候选、选入与省略的上下文、提示词、答案和引用。在此基础上，你可以自行
检索、构建索引、导入小型语料、比较 NumPy 索引与可选 Qdrant，或连接已测试的服务商
使用 Live Ask。当某个阶段需要更深入的解释时，中英双语**学习指南**会在实验旁打
开，无需跳转到 GitHub。

这里的**经典 RAG** 指一条可见的路径：检索证据、打包上下文、生成带引用的答案。经
过测试的 OpenAI 兼容服务商为 Live Ask 补全生成步骤，这不会使项目变成 Agentic RAG
或多步 RAG。

## 这个实验室的独特之处

许多 RAG 示例止步于一次框架调用或一条顺利完成的答案路径。`tiny-rag-lab` 把这些
概念真正连接起来：

- **实现与学习体验保持连接。** CLI 和 Studio 都呈现来自同一套 RAG 核心的产物，而
  不是讲解与可执行代码脱节的抽象流程图。
- **真实产物就是课程。** 引导回放展示一次结果背后的真实文档、文本块、向量、排序
  候选、选入与省略的上下文、提示词、答案、引用和耗时。
- **从引导走向实验。** 从稳定、不依赖服务商的回放开始，再调整检索方式、构建索
  引、上传语料，或连接服务商进行 Live Ask。
- **失败也是学习内容。** 评估、trace 检查和精心设计的失败场景，让你能够追问结果
  *为什么*出现，而不只是确认是否返回了答案。

## 快速开始

```bash
cp .env.example .env
docker compose up --build
```

Docker Compose 默认会在 `http://127.0.0.1:8000` 提供 Studio（可通过
`TINY_RAG_LAB_PORT` 修改）。学习指南在每个实验室阶段都有链接，也同时发布在
[项目主页](https://jameswei.github.io/tiny-rag-lab/)上，无需运行实验室即可阅
读。Studio 本身只在本地运行——不需要账号，也不做公共部署——Live Ask 只会联系你
自行配置的服务商。

推荐的首次使用路径：

1. **Home → Start guided lesson** —— 从固定的 Cloudflare State & Coordination
   语料中，选择四个已保存课程之一进行回放。
2. **Learn** —— 逐步查看语料、文本块、查询嵌入向量、检索候选、选入的上下文、答
   案和引用。
3. **Retrieval** —— 六个实时模块：词法与稠密机制、NumPy 与 Qdrant 对比、混合融
   合、重排序，以及一次 16 题的浏览器 A/B 评估。
4. **Explore** —— 从 75 题的 watsonxDocsQA 题库或自由问题中提问，比较
   Dense/BM25/Hybrid 检索，可选地重排，然后检查 trace。只有需要 Live Ask 生成
   时，才配置已测试的 OpenAI 兼容服务商。
5. **Build & Inspect** —— 使用内置语料或小型 Markdown/纯文本上传（最多 100 个文
   件、共 100 MiB）构建索引，然后检查文本块、向量和来源信息。
6. **Failure Lab** —— 对比精心设计的失败场景及其改进方案。

每个阶段都有**阅读学习指南**链接，会在新标签页打开对应指南，不影响当前实验状
态。界面提供英文和简体中文；内置语料内容和已记录答案保留其原始语言。

默认使用 CPU 运行——`full` 镜像内置固定版本的嵌入模型和交叉编码器重排序模型，无
需 GPU。若想体验更小的镜像：

```bash
LAB_IMAGE_VARIANT=slim docker compose up --build
```

`slim` 镜像开箱即可使用 Guided Learn 回放和 BM25 检索；需要 Dense/Hybrid 检索或
交叉编码器重排序时，可在设置页单独下载嵌入模型和重排序模型。

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

- **索引：** 文档加载、文本规范化、固定字符/结构化/实验性语义分块、元数据、嵌入
  和本地索引。
- **检索：** 稠密余弦检索、BM25 关键词检索、基于倒数排名融合（RRF）的混合检索，
  以及可选的二次重排序。
- **生成：** 显式上下文预算、提示词组装、OpenAI 兼容的生成边界、引用，以及证据
  不足时的拒答。
- **评估与可观测性：** 检索指标、LLM-as-judge 答案指标、可回放 trace 和精心设计
  的失败诊断。

## CLI

Studio 直接、可重复的伙伴——同一套机制，紧凑的命令：

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

React 开发服务器会将 `/docs` 请求代理到由 `npm --prefix learning_materials run dev`
启动的 VitePress 开发服务，与打包后的同源路径保持一致。

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

## 技术栈

- Python · `argparse` CLI · FastAPI · React + TypeScript · VitePress · Docker Compose
- 本地嵌入模型：`sentence-transformers/all-MiniLM-L6-v2`
- 默认索引：可检查的 NumPy 文件；可选本地 Qdrant 适配器
- 生成：OpenAI 兼容 Chat Completions API
- 离线测试：fake embedder + fake generator
- 不使用 LangChain、LlamaIndex 或 Haystack 来封装学习关键路径上的 RAG 机制

## 文档

- [学习指南](learning_materials/zh/learning-roadmap.md)：CLI 与可视化实验室的概
  念伴读材料——由 Studio 在本地提供，也发布在项目主页上
- [项目提案](docs/proposal.md)：项目目的、理念和非目标
- [架构](docs/architecture.md)：RAG 概念平面和边界
- [文件结构](docs/file-structure.md)：仓库地图
