# 可观测性与调试 —— 让一次 RAG 运行可解释

Phase 1.7 为单次 `retrieve` 和 `ask` 运行增加了可观测性层。目标不是仪表盘，
也不是评估报告。目标是一份可信的单次运行记录：使用了什么查询、返回了哪些块、
发给生成器的 prompt 是什么、模型返回了什么答案，以及每个阶段耗时多少。

---

## 为什么 trace 重要

一个 RAG 答案可能因为多种原因出错：

- 检索器返回了错误的块。
- 正确的块被返回了，但排名很低。
- prompt 没有包含足够上下文。
- 模型忽略了上下文。
- 模型引用了不能支撑答案的块。

没有 trace 时，这些失败会被压缩成一个模糊症状："答案不好"。有了 trace，你可以
逐阶段检查流水线。

Phase 1.6 的评估回答的是批量问题："检索器在一个数据集上表现如何？" Phase 1.7
的 trace 回答的是交互式问题："这一次运行到底发生了什么？"

---

## Trace 类型

所有 trace 类型都在 `tiny_rag_lab/trace.py` 中。它们刻意保持为普通 dataclass，
字段也都是 JSON 原生类型，所以 `dataclasses.asdict()` 和 `json.dumps()` 就足以
完成序列化。

### `ChunkTrace`

`ChunkTrace` 是一个检索块的紧凑视图：

```python
@dataclass
class ChunkTrace:
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    path: str
    score: float
    text_preview: str
```

它保留调试检索时最需要的字段：排名、分数、身份、来源文档和短预览。它不保存完整
块文本，因为 trace 应该保持可读且体积较小。

### `RetrieveTrace`

`RetrieveTrace` 记录一次 `rag retrieve` 运行：

```python
@dataclass
class RetrieveTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace]
    latency_by_stage: dict[str, float]
```

耗时字段如下：

| 检索器 | 耗时字段 |
|---|---|
| `dense` | `load`, `embed`, `retrieve` |
| `hybrid` | `load`, `embed`, `retrieve` |
| `bm25` | `load`, `retrieve` |

BM25 不包含 `embed`，因为它不使用嵌入模型。

### `AskTrace`

`AskTrace` 记录一次 `rag ask` 运行：

```python
@dataclass
class AskTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace]
    prompt: str
    answer: str
    citations: list[str]
    latency_by_stage: dict[str, float]
```

耗时字段是 `load`、`embed`、`retrieve`、`prompt_assembly` 和 `generate`。这把完整
RAG 流水线拆成了可检查的阶段。

---

## 终端输出与 JSON 输出

`rag retrieve` 和 `rag ask` 现在都会打印由 formatter 生成的 trace 输出：

```bash
rag retrieve "how do I deploy a model" --index-dir .tiny-rag/index
rag ask "how do I deploy a model" --index-dir .tiny-rag/index
```

加上 `--trace-out` 会把同一份 trace 契约写成 JSON 文件：

```bash
rag retrieve "how do I deploy a model" \
  --index-dir .tiny-rag/index \
  --trace-out /tmp/retrieve-trace.json

rag ask "how do I deploy a model" \
  --index-dir .tiny-rag/index \
  --trace-out /tmp/ask-trace.json
```

`--trace-out` 不改变运行本身，只是额外写出 JSON 产物。

---

## 如何阅读 Retrieve Trace

先看头部：

- `retriever`：确认本次运行使用的是 dense、BM25 还是 hybrid。
- `top_k`：确认命令请求了多少个块。
- `latency`：查看时间花在加载、嵌入还是排名上。

再看块列表：

- rank 1 的高 `score` 通常表示查询和块在嵌入空间中很接近。
- 所有块的分数都低，可能说明查询与语料不匹配。
- 正确的 `doc_id` 出现在低排名，说明检索器找到了答案但排序不够自信。
- 重复的 `doc_id` 有时有用，但也可能挤掉其他证据。

Retrieve trace 是在不涉及 LLM 的情况下调试检索的最快入口。

---

## 如何阅读 Ask Trace

Ask trace 包含检索信息，另外还有 prompt 和答案。

建议按这个顺序检查：

1. **Chunks**：检索是否带来了有用证据？
2. **Prompt**：prompt 组装是否包含预期的 source marker？
3. **Answer**：生成器是否遵守上下文？
4. **Citations**：答案中的引用是否指向检索到的 chunk ID？
5. **Latency**：哪个阶段占用了最多时间？

这个顺序很重要。如果 chunks 错了，模型从一开始就没有正确证据。如果 chunks 对了
但答案错了，问题更可能在 prompting 或 generation。

---

## 这如何连接到 Phase 1.8

Phase 1.8 会研究 RAG 失败模式。Phase 1.7 的 trace schema 是这项工作的输入契约。
失败分析需要持久产物：query、chunks、scores、prompt、answer、citations 和
latency。trace 文件正好提供这些内容，而且不需要数据库或报表 UI。

关键学习点：可观测性不只是 logging。它是在选择一组足够小、但能在事后解释系统
行为的字段。

