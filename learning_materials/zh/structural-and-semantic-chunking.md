# 结构化与语义分块

Phase 2.2 在现有的固定字符基准策略之外，新增了两种分块方式。分块策略是一个
**索引时的决策**——在运行 `rag index` 时选择，所有下游命令（`retrieve`、`ask`、
`eval`、`diagnose`）直接使用已在磁盘上的分块，不需要任何修改。

---

## 为什么分块策略很重要

切片不变式保证每个 chunk 的文本都是文档 `normalized_text` 的正确子串，但
"正确的子串"不等于"有用的子串"。

固定大小的字符窗口可能从句子中间截断：

```
Chunk A: "在批量导入之前，请将 timeout_ms 设置为 150"   ← 句子不完整
Chunk B: "00，并将 retry_mode 设置为 manual。"         ← 依赖 Chunk A
```

其中一个 chunk 仍然可能匹配关于 `timeout_ms` 的查询，但任何一个 chunk
单独来看都无法支撑完整答案，因为它们都没有完整指令。结构化和语义分块通过将
边界放在有意义的文档位置（而不是任意字符偏移量）来解决这个问题。

---

## 策略一：固定字符分块（基准）

```
chunk_document(doc, chunk_size=800, chunk_overlap=120)
```

以 `chunk_size` 个字符为窗口大小，每步向前滑动 `chunk_size - chunk_overlap`
个字符。相邻 chunk 共享 `chunk_overlap` 个字符。

**适用场景**：快速实验、验证管线端到端，或文档没有明显结构（纯散文、未格式化
的日志）。

---

## 策略二：结构化分块

```
chunk_document_structural(doc, chunk_size=800, chunk_overlap=120)
```

三层打包（详见 Phase 2.2 规范中的设计决策 2）：

1. **第一层——整块（block）。** 在空行边界处切分 `normalized_text`
   （`_split_blocks`）。如果某个块只包含一行 ATX 标题（`# …`），则与下一个
   块合并，确保标题不会单独成块。将整块贪婪地打包，直到加入下一块会超过
   `chunk_size`。

2. **第二层——句子。** 如果单个块本身宽于 `chunk_size`，则用 `_split_sentences`
   将其切成句子，再以同样的方式打包。这是"在条件允许时尊重句子边界"的层——
   只有当整块打包无法满足预算时才会触发。

3. **第三层——字符窗口。** 如果单个句子本身宽于 `chunk_size`（超长句或无标点的
   代码块），回退到 `_chunk_oversized_span`——与 `chunk_document` 相同的滑动
   窗口，但仅限于该句子的字符范围。这是**唯一**使用 `chunk_overlap` 的层。

第一层和第二层的 chunk **完全连续**（`chunk[i].char_end ==
chunk[i+1].char_start`）——无重叠、无间隔。第三层的 chunk 与固定字符分块一样
有重叠。

**适用场景**：格式化的 Markdown 语料（文档、维基、操作手册），标题与正文、
段落之间的边界是有意义的。

---

## 策略三：语义分块（实验性）

```
chunk_document_semantic(doc, embedder, chunk_size=800, chunk_overlap=120,
                        similarity_threshold=0.5)
```

步骤：

1. 将 `normalized_text` 切分成句子（`_split_sentences`）。
2. **用一次批量调用**嵌入所有句子——`embedder.embed([s1, s2, …])`。
   这是一次 embedder 调用，而不是每个句子调用一次。它仍然是额外的句子级
   embedding 步骤，发生在写入索引所需的常规 chunk 级 embedding 之前。
3. 按顺序遍历句子。当满足以下条件之一时，关闭当前 chunk 并开始新的：
   - 加入下一个句子会超过 `chunk_size`，**或**
   - 当前句子与前一个句子的嵌入余弦相似度低于 `similarity_threshold`。

余弦相似度直接用点积计算，因为 `Embedder` 的向量已经 L2 归一化——不需要额外的
归一化步骤。

单个句子宽于 `chunk_size` 时，回退到 `_chunk_oversized_span`（第三层）。

**适用场景**：没有明显结构的叙述性文本，主题切换可以通过嵌入相似度检测，但
无法通过空行判断。

**关键成本**：语义分块需要嵌入每个句子——比现有的块级嵌入步骤粒度更细、成本
更高。它被标记为实验性、需主动选择的模式。

---

## 共享的私有辅助函数

这些分块器复用一组小的私有辅助函数来保证正确性和代码复用。结构化与语义策略
额外使用边界感知相关的辅助函数：

| 辅助函数 | 作用 |
|---|---|
| `_validate_chunk_params` | 对无效的 `chunk_size` / `chunk_overlap` 抛出 `ValueError`——与 `chunk_document` 一直以来的报错文本完全相同 |
| `_chunk_metadata(doc)` | 返回 `{title, path, format, raw_hash}` 元数据字典 |
| `_split_sentences(text)` | 正则句子分割器；返回覆盖整个字符串、无间隙的 `[(start, end), …]` 区间列表 |
| `_split_blocks(text)` | 带有 ATX 标题合并的空行分割器；返回 `[(start, end), …]` 区间列表 |
| `_chunk_oversized_span(doc, start, end, chunk_size, chunk_overlap)` | 第三层回退：在 `text[start:end]` 范围内滑动字符窗口 |

---

## 策略分发

```python
chunk_document_with_strategy(
    doc,
    strategy="fixed_character",  # 或 "structural" 或 "semantic"
    chunk_size=800,
    chunk_overlap=120,
    embedder=None,                # strategy="semantic" 时必须提供
    similarity_threshold=0.5,
)
```

对未知策略名或 `strategy="semantic"` 且 `embedder is None` 的情况，抛出
`ValueError`。

`chunk_documents_with_strategy(docs, ...)` 依次对每个文档应用
`chunk_document_with_strategy`。

---

## 清单记录

分块策略**只记录一次**，存入索引清单（manifest）——不会写到每个 chunk 上。
这保持了 `Chunk` 数据类不变，意味着检索、trace 格式化、引文和提示词组装
不需要任何修改：

```json
{
  "chunking_strategy": "structural",
  "chunking_params": {}
}
```

对于 `semantic`：

```json
{
  "chunking_strategy": "semantic",
  "chunking_params": {"similarity_threshold": 0.5}
}
```

`chunking_params` 始终被序列化（永不缺席），消费者无需对其存在与否进行分支处理。

---

## CLI 命令

```bash
# 默认：固定字符分块（与之前所有 phase 行为相同）
rag index --corpus PATH --index-dir .tiny-rag/index

# 结构化分块
rag index --corpus PATH --index-dir .tiny-rag/structural \
    --chunking-strategy structural

# 语义分块（自定义阈值）
rag index --corpus PATH --index-dir .tiny-rag/semantic \
    --chunking-strategy semantic \
    --semantic-similarity-threshold 0.7
```

对于 `structural` 和 `semantic`，`--chunk-overlap` 仅在第三层字符窗口回退时
生效；这两种策略产生的大多数 chunk 没有重叠。

---

## 前后对比演示

Phase 2.2 在 `tests/fixtures/chunking_corpus/` 中提供了一个具体的演示，
走查文档在 `docs/phases/phase-2.2-chunking-comparison.md`。

核心结论：在 `chunk_size=75` 时，固定字符分块将黄金指令拆分到两个 chunk，
BM25 优先检索到干扰文档。结构化分块保持句子完整，`rag diagnose` 显示该失败
不再复现：

| 索引策略 | `rag diagnose` 结果 |
|---|---|
| `fixed_character` | `Confirmed: 1`——失败复现 |
| `structural` | `Confirmed: 0`——失败消失 |

---

## 这能教给你什么

- 分块边界是设计决策，不仅仅是实现细节。
- 相同的 `Chunk`、`RetrievalResult`、`AskTrace` 和 `rag diagnose`
  基础设施无需修改即可复用——分块纯粹是索引时的决策。
- 比较策略是一种使用模式：用不同的 `--chunking-strategy` 构建两个索引，
  对同一组查询运行，比较报告。
- 语义分块不总是更好——它会增加索引时的嵌入成本，且依赖嵌入模型检测主题
  切换的能力。在采用之前务必先测量。
