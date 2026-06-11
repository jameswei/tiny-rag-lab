# 学习路线图

这是 `learning_materials/` 目录下学习资料的推荐阅读顺序。从上到下依次阅读。

---

## 阅读顺序

| # | 文档 | 定位 |
|---|------|------|
| 1 | [RAG 数据流](the-rag-data-flow.md) | 建立四个核心数据结构的概念和术语体系 |
| 2 | [索引平面](the-indexing-plane.md) | 文档如何变成可搜索的向量 |
| 3 | [检索与生成](retrieval-and-generation.md) | 查询如何找到相关片段并生成答案 |
| 4 | [持久化与测试](persistence-and-testing.md) | 索引的磁盘格式、往返完整性以及假后端测试模式 |
| 5 | [评估检索质量](evaluating-retrieval.md) | 用指标回答"检索器到底好不好用" |
| 6 | [可观测性与调试](observability-and-debugging.md) | 用单次运行 trace 解释一次 retrieve 或 ask 命令 |

---

## 高层架构

整个管线分为两个主平面，外加基础设施和评估层：

```
                   ┌──────────────────────┐
                   │   1. 索引平面         │
                   │                      │
     语料库 ─────-─►│ 加载 → 规范化 →       │
     (.md,.txt)    │ 分块 → 嵌入          │────► .tiny-rag/index/
                   │                       │       (manifest.json,
                   │ documents.py          │        chunks.jsonl,
                   │ chunking.py           │        embeddings.npz)
                   │ embeddings.py         │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │   2. 检索与生成平面   │
                   │                       │
     用户 ────────►│ 嵌入查询 →           │
     问题          │ 余弦搜索 →           │────► 打印答案
                   │ 组装提示词 →         │     + 引文
                   │ 调用 LLM             │     + trace 输出
                   │                       │     + 可选 JSON trace
                   │ retrieval.py          │
                   │ prompting.py          │
                   │ generation.py         │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │   3. 评估层           │
                   │                       │
     qa.jsonl ────►│ 嵌入问题 →           │────► EvalReport
                   │ 检索 → 与标准答案     │     (命中率, MRR,
                   │ 文档列表对比          │      精确率, 召回率)
                   │                       │
                   │ eval.py               │
                   └──────────────────────┘

                   ┌──────────────────────┐
                   │   4. 可观测性层       │
                   │                       │
 retrieve / ask ──►│ 收集 chunks、prompt、 │────► RetrieveTrace /
                   │ answer、citations、   │     AskTrace
                   │ latency               │     (终端 + JSON)
                   │                       │
                   │ trace.py              │
                   └──────────────────────┘
```

两个平面通过磁盘上的索引连接——索引平面写入，检索平面读取。评估层复用了与用户
体验完全一致的检索路径。可观测性层记录一次 `retrieve` 或 `ask` 运行中发生的事，
便于事后调试。

---

## 数据流：从文档到答案

核心管线使用三个数据 dataclass，可观测性层再把命令输出转成 trace dataclass。
每一次箭头都是一次转换。

```
┌──────────┐     ┌──────────┐     ┌──────────────────┐
│ Document │ ──► │  Chunk   │ ──► │ RetrievalResult  │
└──────────┘     └──────────┘     └──────────────────┘
  索引阶段           索引阶段             检索阶段

                       ┌──────────────────────────────┐
                       │ RetrieveTrace / AskTrace     │
                       └──────────────────────────────┘
                         可观测性层
```

| 类型 | 关键字段 | 创建者 | 消费者 |
|---|---|---|---|
| **Document** | `doc_id`, `normalized_text`, `raw_hash`, `title`, `format` | `documents.load_document()` | 分块器 |
| **Chunk** | `chunk_id`, `doc_id`, `text`, `char_start`, `char_end`, `metadata` | `chunking.chunk_document()` | 嵌入器、检索器 |
| **RetrievalResult** | `chunk`, `score`, `rank`（从1开始） | `retrieval.retrieve_by_vector()` | 提示词组装 |
| **RetrieveTrace** | `query`, `retriever`, `top_k`, `chunks`, `latency_by_stage` | `cli.cmd_retrieve()` | 终端输出，可选 JSON trace |
| **AskTrace** | `query`, `retriever`, `top_k`, `chunks`, `prompt`, `answer`, `citations`, `latency_by_stage` | `cli.cmd_ask()` | 终端输出，可选 JSON trace |

核心不变式：`document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text`。
如果这个等式不成立，引文就会指向错误的文本。

---

## CLI 命令

```
rag index --corpus PATH --index-dir .tiny-rag/index --chunk-size 800 --chunk-overlap 120
rag retrieve "问题" --index-dir .tiny-rag/index --top-k 5 --trace-out /tmp/retrieve.json
rag ask "问题" --index-dir .tiny-rag/index --top-k 5 --trace-out /tmp/ask.json
rag eval --qa-file qa.jsonl --index-dir .tiny-rag/index --top-k 5
```

每个命令复用前一个命令的输出。`index` 构建索引，`retrieve` 搜索索引，`ask`
运行完整管线，`eval` 评估检索质量。

---

## 学习文档与管线的对应关系

| 学习文档 | 管线阶段 | 对应的源码模块 |
|---|---|---|
| RAG 数据流 | 架构总览 | `models.py`, `cli.py` |
| 索引平面 | 加载、规范化、分块、嵌入 | `documents.py`, `chunking.py`, `embeddings.py` |
| 检索与生成 | 余弦搜索、提示词组装、LLM 调用 | `retrieval.py`, `prompting.py`, `generation.py` |
| 持久化与测试 | 索引读写、往返完整性、假后端模式 | `index_writer.py`, `index_loader.py`, 测试套件 |
| 评估检索质量 | 检索质量指标 | `eval.py` |
| 可观测性与调试 | 单次运行 trace 记录和 JSON 产物 | `trace.py`, `cli.py` |
