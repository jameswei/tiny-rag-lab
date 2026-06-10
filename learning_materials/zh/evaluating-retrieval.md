# 评估检索 — 优化之前先度量

Phase 1.6 为 RAG 流水线增加了一个度量层。在你调整分块大小、替换嵌入模型或引入
BM25 之前，你需要一个可以改进的数字。本文解释这个数字是什么、如何计算，以及它
告诉你什么。

---

## 为什么要先度量？

完成 Phase 1 之后，你可以提问并得到答案。但你无法知道检索器是否找到了正确的文档
块。也许它每次都能检索到正确文档——或者它有 40% 的时间会失误，而大模型用幻觉掩
盖了这一点。

`qa.jsonl` 文件在 Phase 1（`P1-T02`）中专门为这个时刻准备：它包含一批问题，以及
这些问题对应的正确来源文档（已知的标准答案）。这就是评估集——一个可以与检索结
果进行比对的真值基准。

---

## 评估数据集（`qa.jsonl`）

`qa.jsonl` 中每行是一个 JSON 对象：

```json
{
  "question_id": "q001",
  "question": "What topics does the sample document cover?",
  "answer": "It covers several topics useful for retrieval testing.",
  "gold_doc_ids": ["with_h1.md"]
}
```

关键字段是 `gold_doc_ids`。这些是相对于语料库根目录的路径（与索引中的
`Document.doc_id` 完全相同），指向真正包含答案的文档。检索器工作正常时，至少有
一个检索到的文档块应来自这些文档之一。

这种直接匹配——`chunk.doc_id` 与 `gold_doc_ids` 对比——使得所有指标都是确定性
的，不需要大模型判断。

---

## 数据契约（`eval.py`）

三个数据类在流水线中传递评估状态：

```python
@dataclass
class EvalSample:
    question_id: str
    question: str
    answer: str               # 保留，供未来的答案质量指标使用
    gold_doc_ids: list[str]   # 检索器应当找到的内容
```

```python
@dataclass
class EvalResult:
    question_id: str
    question: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]  # 检索器实际找到的内容，按排名顺序
    hit: bool                     # 是否至少有一个检索结果命中了 gold 文档？
    reciprocal_rank: float        # 1/首次命中排名；未命中则为 0.0
    context_precision: float      # 检索结果中相关文档的比例
    context_recall: float         # gold 文档中被检索到的比例
```

```python
@dataclass
class EvalReport:
    n_questions: int
    top_k: int
    hit_rate: float               # 所有问题的命中率均值
    mrr: float                    # 平均倒数排名
    mean_context_precision: float
    mean_context_recall: float
    per_question: list[EvalResult]
```

注意这三层的分工：`EvalSample` 是输入（来自数据集），`EvalResult` 是单题输出
（来自一次检索），`EvalReport` 是汇总（跨所有问题）。每种类型只包含它所需的字段。

---

## 四个指标

`eval.py` 中的四个指标函数都是纯函数——接受两个字符串列表，返回一个数字。调用
方在传入之前负责将检索结果截取到所需的 `k` 条，`k` 隐含在列表长度中。

### Hit Rate @ k（命中率）

```python
def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> bool:
    return any(d in gold_doc_ids for d in retrieved_doc_ids)
```

最简单的问题：检索器在 top-k 中是否找到了*任何*有用内容？只要有一个检索到的文档
块来自 gold 文档，就算命中。

**直觉理解：** 把这看成每题的通过/不通过。命中率就是通过的题目比例。

### MRR — 平均倒数排名

```python
def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / i
    return 0.0
```

命中率不区分答案排在第 1 位还是第 5 位，MRR 会区分。单题的倒数排名（RR）是
`1/首次命中排名`：

| 首次命中排名 | RR |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| 未命中 | 0.000 |

MRR 是所有题目 RR 的均值。一个能稳定把正确文档块排在第 1 位的系统，MRR 接近
1.0；经常把答案排在很靠后或 top-k 中没有命中的系统，MRR 接近 0。

**直觉理解：** MRR 衡量"第一个有用的文档块在结果中排多靠前"。它会惩罚那种能
找到答案但只排在第 4 位的检索器。

### Context Precision @ k（上下文精确率）

```python
def context_precision_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d in gold_doc_ids)
    return hits / len(retrieved_doc_ids)
```

检索到的 k 个文档块中，有多少比例来自 gold 文档？一个主要检索到无关内容的检索
器会浪费大模型的上下文窗口，甚至干扰它。

**一个细节：** 精确率在文档块级别计算（每个检索位置独立计数），而非文档级别。
如果同一个 gold 文档出现在第 2 位和第 4 位，两者都算命中。这是 v1 基准的有意
设计——它反映了提示中的实际内容，而非唯一文档覆盖率。

**直觉理解：** 高精确率意味着大模型看到的是干净的信号；低精确率意味着提示充
满噪音。

### Context Recall @ k（上下文召回率）

```python
def context_recall_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0
    covered = len(set(retrieved_doc_ids) & set(gold_doc_ids))
    return covered / len(gold_doc_ids)
```

在 gold 文档中，有多少被 top-k 检索覆盖了？这里使用集合交集——同一个文档重复
出现在检索结果中只计一次。

**直觉理解：** 高召回率意味着检索器找到了大模型所需的所有证据；低召回率意味着
某些必要证据未出现在 top-k 中，无论生成多好都无法给出完整答案。

### 精确率与召回率的权衡

随着 `k` 增大，这两个指标会朝相反方向变化：

| 增大 k | 效果 |
|---|---|
| 召回率 ↑ | 覆盖更多 gold 文档（好事） |
| 精确率 ↓ | 引入更多无关文档块（坏事） |

选择 `k` 是一种权衡。评估框架让这种权衡变得可见，让你根据数据而非直觉做决定。

---

## 运行器（`run_retrieval_eval`）

```python
def run_retrieval_eval(
    samples: list[EvalSample],
    index: LoadedIndex,
    embedder: Embedder,
    top_k: int,
) -> EvalReport:
```

运行器遍历每个 `EvalSample`，执行检索，然后汇总：

```python
for sample in samples:
    query_vec = embedder.embed([sample.question])[0]
    results = retrieve_by_vector(query_vec, index, top_k=top_k)
    retrieved_doc_ids = [r.chunk.doc_id for r in results]

    hit = hit_at_k(retrieved_doc_ids, sample.gold_doc_ids)
    rr  = reciprocal_rank(retrieved_doc_ids, sample.gold_doc_ids)
    cp  = context_precision_at_k(retrieved_doc_ids, sample.gold_doc_ids)
    cr  = context_recall_at_k(retrieved_doc_ids, sample.gold_doc_ids)
    ...
```

有三点值得注意：

1. **完全复用了现有的检索路径。** `retrieve_by_vector` 就是 `rag ask` 使用的同
   一个函数。评估度量的正是用户实际体验到的行为。

2. **提取的是 `doc_id`，而非 `chunk_id`。** 同一个文档可能产生多个文档块；指标
   关心的是文档级覆盖率。

3. **`retrieve_by_vector` 在函数体内延迟导入。** 这避免了循环导入：
   `eval.py` → `retrieval.py` → `index_loader.py` → 回到其他地方定义的类型。
   文件顶部的 `TYPE_CHECKING` 块处理类型提示，不产生运行时开销。

---

## 读懂输出

下面的基准来自本地缓存的 `ibm-research/watsonxDocsQA` 快照：

```text
Documents : 1144
Chunks    : 8648
Questions : 75
Model     : sentence-transformers/all-MiniLM-L6-v2
```

```
Evaluation report  (n=75, top_k=5)
────────────────────────────────────
Hit Rate @ 5     :  0.867
MRR               :  0.756
Context Precision :  0.365
Context Recall    :  0.867
```

如何解读这些数字：

- **命中率 0.867** — 87% 的问题在 top-5 中至少有一个相关文档块。13% 的问题在
  top-5 中没有任何相关内容。
- **MRR 0.756** — 第一个相关文档块通常出现在较靠前的位置。很多首次命中在第 1
  名，也有一些更靠后的命中把均值拉低。
- **上下文精确率 0.365** — top-5 中约 37% 的文档块来自 gold 文档，其余是进入提示
  的干扰内容。
- **上下文召回率 0.867** — top-5 覆盖了 87% 的 gold 文档。

这是一个可复现的本地基准，不是基准榜单结论。它依赖准备好的数据快照、分块参数、
嵌入模型和 `top_k`。

同一组本地运行也展示了精确率和召回率之间的权衡：

| top_k | 命中率 | MRR | 上下文精确率 | 上下文召回率 |
|---:|---:|---:|---:|---:|
| 1 | 0.680 | 0.680 | 0.680 | 0.680 |
| 3 | 0.840 | 0.749 | 0.453 | 0.840 |
| 5 | 0.867 | 0.756 | 0.365 | 0.867 |
| 10 | 0.907 | 0.762 | 0.249 | 0.907 |

随着 `k` 增大，命中率和召回率提高，因为检索器有更多机会包含 gold 文档；精确率下降，
因为更多非 gold 文档块进入提示。Phase 1.5（BM25、混合检索）将尝试改善这些数字，
现在你有了判断改进是否有效的工具。

---

## CLI 命令

```bash
rag eval --qa-file corpus/watsonx-docsqa/qa.jsonl \
         --index-dir .tiny-rag/index \
         --top-k 5
```

`cli.py` 中的 `cmd_eval` 遵循与 `cmd_retrieve` 相同的模式：

```python
def cmd_eval(args):
    index = load_index(Path(args.index_dir))
    embedder = _make_embedder(index.manifest.get("embedding_model"))
    samples = load_eval_samples(Path(args.qa_file))
    report = run_retrieval_eval(samples, index, embedder, top_k=args.top_k)
    print(format_eval_report(report))
```

`_make_embedder` 工厂与 `cmd_retrieve` 和 `cmd_ask` 使用的完全相同。测试时用
`FakeEmbedder` 替换它——不需要下载模型，不需要网络。

---

## 本文的学习要点

评估回答了 Phase 1 流水线留下的问题：**检索器是否真的在正常工作？**

核心洞察：检索质量和答案质量是两件独立的事。一个糟糕的检索器配上优秀的大模型，
会产生措辞流畅但内容错误的答案。一个优秀的检索器配上普通的大模型，至少能产生
有据可查的正确答案。评估框架通过单独度量检索来分离这两者——不需要大模型参与。

四个指标各自揭示一种不同的失效模式：

| 指标 | 低分意味着什么 |
|---|---|
| 命中率 | 检索器的 top-k 中经常没有任何相关文档 |
| MRR | 相关文档被检索到了，但排名很靠后（第 4 名而非第 1 名） |
| 上下文精确率 | top-k 充满噪音——大模型在大量无关内容中挣扎 |
| 上下文召回率 | 部分必要证据未出现在 top-k 中（可能存在于更靠后的排名） |

不同的失效模式对应不同的修复方案。改善召回率可能意味着增大 `k` 或使用混合检索；
改善精确率可能意味着减小 `k` 或引入重排序。没有指标，你只是在猜测。
