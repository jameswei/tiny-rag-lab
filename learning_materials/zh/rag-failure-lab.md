# RAG 失败实验室 —— 把检索错误变成可重复测试用例

Phase 1.8 增加了一个小型失败实验室。目标不是把所有坏答案都归为“效果不好”，而是用
可重复的案例对常见的检索侧失败进行分类。

---

## 失败实验室增加了什么

核心实现位于 `tiny_rag_lab/failure.py`：

| 组件 | 作用 |
|---|---|
| `FailureCase` | 一个策划好的问题、gold 文档列表、期望标签、baseline 配置和 intervention 配置 |
| `RetrieverConfig` | 每个 case 自己的检索策略：`dense`、`bm25` 或 `hybrid`，以及 `top_k` |
| `detect_failure_label()` | 根据 retrieved doc ids 和 gold doc ids 分配启发式标签 |
| `run_diagnosis()` | 对每个 case 分别运行 baseline 和 intervention 检索 |
| `DiagnosisReport` | 汇总 confirmed、fixed、moved 以及每个 case 的结果 |
| `format_diagnosis_report()` | 打印可读的终端报告 |

CLI 入口是：

```bash
rag diagnose --cases-file tests/fixtures/failure/cases.jsonl --index-dir .tiny-rag/index
```

和 `rag eval` 不同，这个命令没有 `--retriever` 或 `--top-k` 参数。这些设置写在每个
`FailureCase` 里，因为这个命令的重点就是比较某个 baseline 检索配置和一个具体的
intervention 配置。

---

## 五个启发式标签

Phase 1.8 只标注能从检索结果中判断的失败：

| 标签 | 含义 |
|---|---|
| `missing_evidence` | 检索结果中没有任何文档命中 gold 文档列表 |
| `low_rank_evidence` | gold 文档被找到了，但第一次命中的排名低于阈值 |
| `distractor_evidence` | gold 文档排名还可以，但检索上下文里噪音太多 |
| `unanswerable_query` | 这个 case 有意设置为语料库中没有 gold 文档 |
| `no_failure` | 检索结果满足配置的阈值 |

还有两个重要失败被记录下来，但没有实现为启发式检测：`unsupported_answer` 和
`citation_mismatch`。它们需要判断答案本身，通常要靠人工或 LLM-as-judge，因为只看
retrieved ids 无法证明生成答案是否忠实于上下文。

---

## 检测逻辑如何复用评估指标

`detect_failure_label()` 刻意复用 `eval.py` 中的指标函数：

```python
hit_at_k(retrieved_doc_ids, gold_doc_ids)
reciprocal_rank(retrieved_doc_ids, gold_doc_ids)
context_precision_at_k(retrieved_doc_ids, gold_doc_ids)
```

这样检索指标只有一个事实来源。

检测顺序很重要：

1. gold 列表为空且期望标签是 `unanswerable_query`，则是不可回答问题。
2. 没有命中 gold 文档，则是 missing evidence。
3. 第一次命中低于排名阈值，则是 low-rank evidence。
4. 排名可接受但 precision 过低，则是 distractor evidence。
5. 其他情况为 no retrieval failure。

低排名检测先于 distractor 检测。一个 gold 文档在第 4 名且周围很多噪音的 case，应该
教会我们“证据被埋得太深”，而不只是“上下文有噪音”。

---

## 如何阅读诊断报告

报告会比较每个 case 的两次运行：

```text
Diagnosis report  (n=6)
--------------------------------------------
  Confirmed  : 4
  Fixed      : 1
  Moved      : 1
--------------------------------------------
Case fc001  expected=missing_evidence
  baseline   : missing_evidence       hit=0.000  prec=0.000  recall=0.000  mrr=0.000
  interv.    : no_failure             hit=1.000  prec=0.500  recall=1.000  mrr=0.500
  FIXED
```

结果词是最重要的学习信号：

| 结果 | 含义 |
|---|---|
| `CONFIRMED` | baseline 复现了期望的失败标签 |
| `FIXED` | baseline 失败，但 intervention 变成了 `no_failure` |
| `MOVED` | intervention 把失败变成了另一种失败 |
| `UNCHANGED` | 结果既没有匹配期望失败，也没有被修复或移动 |

关键习惯是把标签和指标一起读。`mrr` 说明证据是否排得更靠前，
`context_precision` 说明噪音是否增加，`context_recall` 说明 gold 文档是否被覆盖。

---

## 它和评估有什么不同

`rag eval` 问的是：“这个检索器在一个数据集上整体表现如何？”

`rag diagnose` 问的是：“针对这个已知失败，这个 intervention 是修复了它、移动了它，
还是没有改变它？”

评估是宽范围测量，诊断是聚焦学习。两者结合后，RAG 不再只是黑盒：指标揭示模式，
trace 展示单次运行，失败案例把模式变成可重复观察的东西。
