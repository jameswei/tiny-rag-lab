# 上下文预算与结构化答案——控制哪些内容进入提示词

Phase 2.1 将生成侧原本不可见的机制暴露出来：哪些检索到的 chunk 真正进入了提示词，
以及哪些因为预算限制被排除在外。此外还为 `rag ask` 增加了机器可读的 JSON 输出模式。

---

## 检索与提示词之间的空白

检索结束后，你得到一份有序的 chunk 列表。在 Phase 2.1 之前，这些 chunk 全部原封不动地进入提示词。当 `top_k` 较小、chunk 较短时，这行得通。但上下文窗口是有限的：

- 4 096 token 的窗口放不下十个 600 token 的 chunk
- reranker 可能返回大量候选，但最终提示词只能容纳其中一部分
- 从输出中根本看不出哪些 chunk 被包含了

Phase 2.1 让这个选择步骤变得明确且可检查。

---

## 上下文打包步骤

`tiny_rag_lab/context.py` 中的 `pack_context` 采用贪心算法进行选择：

```python
def pack_context(
    results: list[RetrievalResult],
    budget: int,
    counter: TokenCounter,
    question: str = "",
) -> ContextPackResult:
    ...
```

算法流程：

1. 从预算中扣除 `PROMPT_OVERHEAD`（100 token）和问题的 token 数，为提示词模板预留空间。
2. 按排名顺序遍历 chunk。
3. 对每个 chunk，使用与 `assemble_prompt` 完全相同的格式进行格式化，统计 token 数，若剩余预算足够则选中它。
4. 返回包含 `selected`、`omitted` 和 `estimated_tokens` 的 `ContextPackResult`。

调用方（`cmd_ask`、`run_answer_eval`、`run_answer_diagnosis`）在调用 `assemble_prompt` 之前，将结果列表过滤为 `selected` chunk ID。`prompting.py` 本身不做任何修改。

---

## Token 计数器

提供两种实现：

| 计数器 | 精度 | 依赖 |
|---|---|---|
| `FakeTokenCounter` | 约 4 个字符对应 1 个 token | 无（始终可用） |
| `TiktokenCounter` | 精确的 tiktoken 编码 | `pip install tiktoken` |

`cli.py` 自动选择：`_make_token_counter()` 首先尝试 `TiktokenCounter`，若 tiktoken 未安装则回退到 `FakeTokenCounter`。测试直接使用 `FakeTokenCounter`；在生产环境中，回退是透明的。

关键约束：`FakeTokenCounter` 快速且离线，但可能多选或少选几个 token。`TiktokenCounter` 对所选 tiktoken 模型编码更精确。

---

## Trace 中显示的内容

当 `--context-budget > 0` 时，`AskTrace.context_pack` 会被填充：

```json
{
  "context_pack": {
    "selected": ["chunk0000000001", "chunk0000000002"],
    "omitted":  ["chunk0000000003"],
    "estimated_tokens": 847,
    "budget": 8192,
    "counter_name": "tiktoken-gpt-4o-mini"
  }
}
```

人类可读的 trace 会在 chunk 列表与答案之间插入一个块：

```text
Context packing  (budget=8192, counter=tiktoken-gpt-4o-mini)
  Selected  : 2 chunks   (~847 tokens used)
  Omitted   : 1 chunk
    - chunk0000000003
```

当 `--context-budget 0`（默认值）时，`context_pack` 为 `null`，该块不会出现。Phase 2.0 的输出完全不变。

---

## CLI 用法

```bash
# 默认：不设预算，与 Phase 2.0 完全相同
rag ask "问题" --index-dir .tiny-rag/index

# 设置预算：trace 中出现打包块
rag ask "问题" --index-dir .tiny-rag/index --context-budget 8192

# JSON 输出：将完整 AskTrace 以缩进 JSON 格式输出到 stdout
rag ask "问题" --index-dir .tiny-rag/index --context-budget 8192 --output-format json

# eval 和 diagnose 也支持 --context-budget
rag eval --qa-file qa.jsonl --index-dir .tiny-rag/index --judge fake --generator fake --context-budget 8192
rag diagnose --cases-file cases.jsonl --index-dir .tiny-rag/index --judge fake --context-budget 8192
```

`--context-budget -1` 会立即抛出 `ValueError`。`--context-budget 0` 完全跳过打包步骤。

---

## JSON 输出模式

`--output-format json` 将完整的 `AskTrace` 字典输出到 stdout，而非人类可读的 trace。这使得下游脚本无需解析文本即可消费完整 trace 和答案字段：

```bash
rag ask "问题" --index-dir .tiny-rag/index \
  --context-budget 8192 \
  --output-format json | jq '.context_pack.omitted'
```

无论 `--output-format` 设置为何值，`--trace-out PATH` 始终将 JSON 写入文件。

---

## 核心结论

以下三点在此前是不可见的，Phase 2.1 将其暴露出来：

| Phase 2.1 之前 | Phase 2.1 之后 |
|---|---|
| 所有检索到的 chunk 都进入提示词 | 预算按排名顺序贪心选择 chunk |
| 没有记录哪些被包含或排除 | `context_pack.selected` 和 `omitted` 列表 |
| `rag ask` 只有文本输出 | 可通过 `--output-format json` 获得 JSON 输出 |

紧张的预算让你看到：排名靠前不代表能进入上下文。一个高排名的 chunk 仍可能因为前面的 chunk 用尽了剩余预算而被排除。这是 Phase 2.1 让生产现实变得可观测的关键所在。
