# 答案质量评判：衡量生成出来的答案

Phase 2.0 增加了答案质量评判。检索指标回答的是“有没有取到正确证据”，答案评判
回答的是“最终答案有没有正确使用这些证据”。

Judge 是可选的。没有传 `--judge`，或者使用 `--judge none` 时，已有命令的行为保持
不变。

---

## 为什么只有检索指标还不够

检索器找到正确 chunk 后，答案仍然可能出错：

- 答案编造了上下文里没有的内容
- 答案没有真正回答用户问题
- 引用的来源并不支撑对应陈述
- 和参考答案相比事实错误

Phase 1.6 衡量检索。Phase 2.0 衡量“检索 + 生成”之后得到的最终答案。

---

## Judge 接口

Judge 合同在 `tiny_rag_lab/judge.py` 中。

```python
@dataclass
class JudgeVerdict:
    faithfulness: float
    answer_relevance: float
    citation_support: float
    answer_correctness: float | None
    judge_name: str
    latency: float
    notes: str = ""
```

四个分数的含义：

| 分数 | 含义 |
|---|---|
| `faithfulness` | 答案是否忠实于检索到的上下文 |
| `answer_relevance` | 答案是否回应了问题 |
| `citation_support` | 引用来源是否支撑答案中的陈述 |
| `answer_correctness` | 在有参考答案时，答案是否接近参考答案 |

当没有 `reference_answer` 时，`answer_correctness` 是 `None`。这和 `0.0` 不同：
缺少参考答案表示“没有测量”，不是“答案错误”。

当前有两个 judge 实现：

| Judge | 用途 |
|---|---|
| `FakeJudge` | 离线、确定性的测试和示例 |
| `OpenAIJudge` | OpenAI 兼容接口上的真实 JSON-mode judge |

`FakeJudge.verdict_map` 用答案字符串作为 key。这对失败实验室很重要，因为 baseline
和 intervention 可以使用同一个问题，但脚本化答案不同。

---

## 答案评估

`rag eval` 现在可以在一个命令里打印两段报告：

```bash
rag eval \
  --qa-file tests/fixtures/eval/qa.jsonl \
  --index-dir .tiny-rag/index \
  --judge fake \
  --generator fake
```

检索报告和答案质量报告是分开的：

```text
Evaluation report  (...)
...

Answer quality report  (n=3, judge=fake)
--------------------------------------------
Faithfulness      :  1.000
Answer Relevance  :  1.000
Citation Support  :  1.000
```

这种分离是有意设计。检索质量和答案质量可以独立变化：reranker 可能提高 hit rate，
但生成器仍然给出不被上下文支持的答案；prompt 改进也可能提升 faithfulness，但不改变
检索指标。

`qa.jsonl` 中可以增加可选的答案侧字段：

```json
{
  "question_id": "q1",
  "question": "What does the document cover?",
  "answer": "It covers retrieval testing.",
  "gold_doc_ids": ["with_h1.md"],
  "reference_answer": "The document covers retrieval testing topics.",
  "expected_facts": ["retrieval testing"]
}
```

旧数据仍然可以加载。`reference_answer` 默认是 `None`，`expected_facts` 默认是空列表。

---

## Ask Trace 中的 Verdict

`rag ask` 可以评判单次生成答案：

```bash
rag ask "sample document" \
  --index-dir .tiny-rag/index \
  --judge fake \
  --generator fake \
  --trace-out /tmp/ask.json
```

启用 judge 时，`AskTrace.verdict` 会被填充，可读 trace 中会出现 `Judge verdict` 段。
禁用 judge 时，JSON 中的 `verdict` 是 `null`，终端输出不会多出 verdict 段。

这样答案质量就和检索过程一样可调试：chunks、prompt、answer、citations、latency
和 judge verdict 都能在同一次 trace 中看到。

---

## 答案侧失败诊断

Phase 1.8 记录了两个无法只靠 retrieved doc id 判断的失败：

| 标签 | 含义 |
|---|---|
| `unsupported_answer` | 答案包含上下文不支持的内容 |
| `citation_mismatch` | 引用没有支撑对应陈述 |

Phase 2.0 用 judge 实现它们：

```python
detect_answer_failure_label(verdict, thresholds)
```

检测顺序很简单：

1. `faithfulness` 过低 -> `unsupported_answer`
2. `citation_support` 过低 -> `citation_mismatch`
3. 否则 -> `no_failure`

`rag diagnose --judge fake --generator fake` 现在会先打印检索诊断，再为答案侧案例打印
答案诊断段。

fixture 增加了两个案例：

| Case | 失败类型 |
|---|---|
| `fc008` | unsupported answer |
| `fc009` | citation mismatch |

这些案例在 JSONL 中直接写入 `baseline_answer` 和 `intervention_answer`。字段存在时，
`run_answer_diagnosis` 会跳过 generator，直接评判脚本化答案。这样失败实验室仍然是
确定性的、离线可运行的。

---

## 核心收获

生产级 RAG 至少需要两层测量：

| 层 | 问题 |
|---|---|
| 检索评估 | 有没有取到正确证据？ |
| 答案评判 | 答案有没有正确使用证据？ |

Phase 2.0 保持这两层都清晰可见。检索指标、答案指标、trace 和失败案例仍然是分开的
对象，因此回归发生时更容易定位原因。
