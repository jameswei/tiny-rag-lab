# 检索机制 —— Dense、BM25 与混合检索

Phase 1 只提供一种检索器：dense 余弦相似度搜索。Phase 1.5 在此基础上增加了两种：
基于关键词的 BM25，以及通过 Reciprocal Rank Fusion 将 dense 和 BM25 融合的混合
检索。两个新模块完成这项工作：`bm25.py`（关键词检索）和 `hybrid.py`（融合逻辑）。

---

## 为什么需要三种检索器？

Dense 检索擅长语义匹配："汽车保养"和"车辆维护"即使没有共同词汇也能得到高分。但
它有时会漏掉精确匹配——搜索"API key rotation"，可能会把关于"key management"的文档
排在字面包含"API key rotation"的页面之上。

BM25 恰恰相反。它关心精确的词汇重叠——如果查询词"rotation"在某文档中大量出现，
BM25 会提升该文档的排名。但它完全不理解同义词和改写。

混合检索结合了二者：dense 捕捉语义，BM25 捕捉关键词，Reciprocal Rank Fusion 将
两个排序列表合并为一个。

---

## BM25 关键词检索 (`bm25.py`)

### BM25 做了什么

BM25（Best Matching 25）是 TF-IDF 家族的一种排序函数。对于每个 chunk，它基于以下
因素计算得分：

- **词频（TF）**：每个查询词在 chunk 中出现的频率。出现次数越多 → 得分越高，但
  存在边际递减效应。
- **逆文档频率（IDF）**：每个查询词在整个语料库中的稀有程度。在大量 chunk 中出现
  的词（如"the"、"and"）会被降权。

实现使用 `rank_bm25`，一个纯 Python 的 BM25 库，无原生扩展、无 GPU、无模型下载。

### 分词器：`_tokenize`

```python
def _tokenize(text: str) -> list[str]:
    return text.lower().split()
```

这是整个模块中最重要的一行代码。它在转小写后按空白字符分割——不做词干提取、不做
停用词移除、不做标点剥离。这意味着：

- `"Watson?"` 和 `"watson"` 是**不同的 token**——问号粘在词上。查询 "watson" 不会
  匹配包含 "Watson?" 的 chunk。
- `"API"` 和 `"api"` 是**相同的 token**——大小写被归一化。
- 没有空格分隔的中文/日文文本**不会被切分**——整个句子变成一个 token。

这个分词器刻意保持简单。目标是让 BM25 的机制透明可见，而非最大化检索分数。生产
系统会剥离标点或使用专业分词器；那属于未来的实验。

### `BM25Retriever` 类

```python
class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        # 对所有 chunk 分词，构建 rank_bm25.BM25Okapi 索引
        # 如果所有 chunk 分词后都为空，self._bm25 保持为 None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        # 返回按 BM25 原始得分排序的结果，score 字段为原始 BM25 分数
```

关键行为：

- **返回 `RetrievalResult` 对象** —— 与管线其他部分使用相同的数据类。`score` 是原始
  BM25 得分（未归一化到 [-1,1]，与余弦相似度不同）。
- **空语料库 → `[]`**：如果 `_bm25` 为 `None`，`retrieve()` 返回空列表。
- **空查询 → `[]`**：空白或仅含空格的查询不返回结果。
- **`top_k < 0` → `ValueError`**：与 dense 检索器的契约一致。

### 一次构建，多次复用

`BM25Retriever.__init__` 会对所有 chunk 构建倒排索引——它分词每个 chunk 并预计算
文档统计信息。这是 O(N) 的操作（N 为语料库大小），应该只执行一次，而非每次查询
都重复。需要运行多次查询的调用方（如 `rag eval`）构建一个 `BM25Retriever` 后复用。
CLI 每次命令构建一个新的，因为单次查询的开销很低。

### BM25 比 dense 更擅长的场景

BM25 在查询包含精确、独特词汇时表现优异：

| 查询 | Dense 可能…… | BM25 倾向于…… |
|---|---|---|
| "`POST /v2/accounts`" | 检索通用 API 文档 | 找到确切的端点文档 |
| "`ValueError: chunk_size must be positive`" | 检索通用错误处理 | 找到具体的错误消息 |
| "`ibm cloud api key rotation`" | 分散到多个主题 | 聚焦于 "rotation" + "key" |

Phase 1.6 的评估工具可以用数字验证这些直觉——在同一个 QA 集上运行
`rag eval --retriever dense` 和 `rag eval --retriever bm25`，比较命中率。

---

## Reciprocal Rank Fusion (`hybrid.py`)

### RRF 公式

```python
def reciprocal_rank_fusion(
    results_lists: list[list[RetrievalResult]],
    top_k: int,
    k: int = 60,
) -> list[RetrievalResult]:
```

对于出现在任一结果列表中的每个唯一 chunk，其融合得分为：

```
rrf_score(chunk) = sum( 1 / (k + rank_i)  对 chunk 出现的每个列表 i 求和 )
```

`rank_i` 从 1 开始，与 `RetrievalResult.rank` 一致。常数 `k = 60` 来自 RRF 原始
论文的平滑参数——它防止某个列表中 rank-1 的结果完全压倒另一列表中 rank-2 的结果。

**示例。** 假设 chunk A 在 dense 中 rank 1、在 BM25 中 rank 3；chunk B 仅在 dense
中 rank 2：

```
rrf(A) = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
rrf(B) = 1/(60+2)               = 0.01613
```

chunk A 胜出，因为它出现在两个列表中，尽管 chunk B 在 dense 中的排名比 A 在 BM25
中的排名更高。

### 融合的具体步骤

1. **累加分数。** 遍历每个结果列表。对于每个 `RetrievalResult`，将
   `1/(k + result.rank)` 加到该 chunk 的累计总分中。
2. **记录首次出现。** `chunk` 对象取自各列表中首次出现该 chunk 的列表——所有
   列表中的 `Chunk` 引用来自同一个索引，所以这只是形式上的选择。
3. **降序排序。** 按融合得分从高到低排序。
4. **重新排名。** `rank` 按融合后的顺序从 1 重新编号。

### 平局处理

当两个 chunk 的融合得分相同时，Python 的稳定排序 `sorted` 保持它们在第一个结果
列表中的相对顺序。由于 dense 结果总是作为第一个列表传入，平局时 dense 排名更高的
chunk 优先。

### 为什么用 RRF 而不是分数归一化

Dense 余弦相似度和 BM25 得分处于不同的数值范围。Dense 大致在 -1 到 1（实践中通常
0.0 到 0.7）；BM25 无上限，可能是 0 到 20+。将分数归一化到统一尺度需要假设它们的
分布特性。RRF 完全避免了归一化——它只使用排名，而排名始终可比较。

---

## 混合检索：`retrieve_hybrid`

```python
def retrieve_hybrid(
    query: str,
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int = 5,
    bm25_retriever: BM25Retriever | None = None,
) -> list[RetrievalResult]:
```

该函数执行以下步骤：

1. **构建 BM25**（如果 `bm25_retriever` 为 `None`）：`BM25Retriever(index.chunks)`。
   需要运行多次查询的调用方应预先构建一个 retriever，通过 `bm25_retriever` 参数
   注入，避免每次查询重建。
2. **运行 dense 检索**：调用 `retrieval.py` 中的 `retrieve(query, index, embedder, top_k=top_k)`。
3. **运行 BM25 检索**：调用 `bm25_retriever.retrieve(query, top_k=top_k)`。
4. **融合结果**：调用 `reciprocal_rank_fusion([dense_results, bm25_results], top_k=top_k)`。
5. **返回**恰好 `top_k` 个结果，score 为融合后的 RRF 分数，rank 从 1 重新编号。

返回的 `RetrievalResult.score` 是融合后的 RRF 分数——一个小正数，而非余弦相似度。

---

## CLI 使用方式

### `rag retrieve` 选择检索器

```bash
# Dense（默认）
rag retrieve "what is watson assistant?" --retriever dense

# BM25 关键词检索 —— 不需要嵌入模型
rag retrieve "what is watson assistant?" --retriever bm25

# 混合检索：dense + BM25，通过 RRF 融合
rag retrieve "what is watson assistant?" --retriever hybrid
```

使用 `--retriever bm25` 时，CLI 完全跳过加载嵌入模型——BM25 不使用嵌入。

### `rag eval` 选择检索器

```bash
rag eval --qa-file qa.jsonl --retriever dense   # Phase 1 基线
rag eval --qa-file qa.jsonl --retriever bm25    # 纯关键词
rag eval --qa-file qa.jsonl --retriever hybrid  # 组合检索
```

评估报告头部包含检索器名称：

```
Evaluation report  (n=847, top_k=5, retriever=hybrid)
──────────────────────────────────────────────────────
Hit Rate @ 5      :  0.751
MRR               :  0.603
Context Precision :  0.328
Context Recall    :  0.667
```

### `rag diagnose` 的逐案例检索器配置

失败实验室（`rag diagnose`）不接受 `--retriever` 参数。每个失败案例在 `cases.jsonl`
中定义自己的 `baseline` 和 `intervention` 检索器配置，因此可以在单次诊断运行中逐
案例对比 dense、BM25 和 hybrid。

---

## 源码模块对应关系

| 模块 | 提供的内容 |
|---|---|
| `tiny_rag_lab/bm25.py` | `_tokenize()`、`BM25Retriever` 类 |
| `tiny_rag_lab/hybrid.py` | `reciprocal_rank_fusion()`、`retrieve_hybrid()` |
| `tiny_rag_lab/retrieval.py` | `retrieve()`、`retrieve_by_vector()` —— hybrid 使用的 dense 路径 |

`models.py`、`index_loader.py` 和 `index_writer.py` 无需任何修改——`RetrievalResult`
和 `Chunk` 数据类原生支持所有三种检索器。

---

## 接下来该看什么

阅读完本文后，最有价值的实验是：

1. **在同一查询上对比检索器。** 用 `--retriever dense` 运行 `rag retrieve "精确术语"`，
   再用 `--retriever bm25` 运行。观察哪些 chunk 同时出现在两个列表中，哪些只出现在
   其中一个。
2. **检查评估数据。** 在同一个 QA 集上分别用三种检索器运行 `rag eval`。看看每种
   检索器答对了哪些问题——它会告诉你哪些类型的查询更依赖关键词，哪些更依赖语义。
3. **追踪一次混合检索运行。** 在 `rag retrieve` 中加上 `--retriever hybrid`，查看
   trace 输出。dense 和 BM25 各自的得分在 RRF 融合之前是分开可见的。

---

## 相关文档

- [检索与生成](retrieval-and-generation.md) —— dense 余弦检索路径的详细说明。
- [评估检索质量](evaluating-retrieval.md) —— 如何衡量哪种检索器效果最好。