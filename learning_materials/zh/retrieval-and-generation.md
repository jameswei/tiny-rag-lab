# 检索与生成 —— 从搜索到答案

索引构建完成后，检索平面找到相关块，生成平面将它们转化为答案。三个模块完成
这项工作：`retrieval.py`（余弦搜索）、`prompting.py`（上下文组装）和
`generation.py`（LLM 调用）。

---

## 检索：余弦相似度搜索（`retrieval.py`）

### 两个函数，一个给人类，一个给机器

```python
def retrieve(query_text, index, embedder, top_k=5) -> list[RetrievalResult]:
    # 嵌入 query_text，然后委托给 retrieve_by_vector

def retrieve_by_vector(query_vec, index, top_k=5) -> list[RetrievalResult]:
    # 纯排名 —— 接受预先计算的向量，返回排名的块
```

这种分离是有原因的。`retrieve` 是面向用户的函数 —— 你传入自然语言，它在内部
处理嵌入。`retrieve_by_vector` 是面向测试的函数 —— 你传入一个已知向量（例如，
测试语料库中块 3 的嵌入），然后验证它把块 3 排在第一位。

这种分离意味着你可以在不测试嵌入器的情况下测试检索排名。如果排名错误，你就知道
bug 在检索数学中，而不是在嵌入模型中。

### 余弦相似度在代码中如何工作

`retrieve_by_vector` 的核心：

```python
query_vec = np.asarray(query_vec, dtype=np.float32)
q_norm = float(np.linalg.norm(query_vec))
if q_norm == 0.0:
    return []                              # 零查询 → 无结果

query_unit = query_vec / q_norm            # 将查询归一化到单位长度

emb = index.embeddings.astype(np.float32)  # (N, dim) — 所有块向量
norms = np.linalg.norm(emb, axis=1, keepdims=True)  # (N, 1)
safe_norms = np.where(norms == 0.0, 1.0, norms)     # 避免除零
emb_unit = emb / safe_norms                # 归一化所有块向量

scores = emb_unit @ query_unit             # (N,) — 通过点积计算余弦相似度
scores[norms[:, 0] == 0.0] = 0.0           # 零范数的块得分为 0

top_indices = np.argsort(scores)[::-1][:actual_k]   # 最高分排最前
```

让我们逐步拆解：

### 第一步：为什么要归一化？

两个向量之间的余弦相似度定义为：

```
cos(a, b) = (a · b) / (||a|| × ||b||)
```

如果两个向量都已经是单位长度（||a|| = ||b|| = 1），分母就是 1，余弦相似度退化
为一个点积：`cos(a, b) = a · b`。

嵌入器（假的和真的）都已经产生单位向量。但检索代码无论如何都重新归一化 —— 这是
一种纵深防御措施。如果将来的嵌入器意外地产生非单位向量，检索仍然能正确工作，
而不是悄悄地产生错误的分数。

### 第二步：矩阵-向量乘法

`emb_unit @ query_unit` 是关键操作。`emb_unit` 的形状是 `(N, dim)`，`query_unit`
的形状是 `(dim,)`。NumPy 将点积广播到所有 N 行，产生一个 1-D 的 N 个分数数组
—— 每个块一个余弦相似度。这是一个用优化的 C 语言一次性完成 N 个独立点积的调用，
比 Python 循环快得多。

### 第三步：零向量处理

代码处理两种零向量情况：

**零查询向量。** 如果用户提交了空查询（或嵌入器由于某种原因产生全零），
`q_norm == 0.0` 触发提前返回 `[]`。没有有意义的方法可以对零向量进行块排名。

**零范数的索引行。** 如果某个嵌入行是全零的（在真实嵌入器中不应该发生，但代码
不信任这一点），`safe_norms` 将零替换为 1.0 以避免除零。点积之后，这些行被显式
设置为 `score = 0.0`，因此它们在并列中排在最后。

### 第四步：排名

`np.argsort(scores)` 返回按分数递增排序的索引。`[::-1]` 反转数组使最高分排在
最前。`[:actual_k]` 取前 k 个结果。然后结果被包装成 `RetrievalResult`，带有从
1 开始的排名。

### 好的分数看起来是什么样的

使用真实嵌入时，相关块的余弦相似度通常在 0.3 到 0.7 之间。0.9+ 的分数异常
强烈；0.0 左右的分数意味着块和查询在语义上无关。你可以用 `rag retrieve` 直接
查看这些：

```
Rank 1  score=0.6234  chunk_id=a1b2c3d4e5f67890
  Title : API Authentication Guide
  Path  : docs/auth.md
  To authenticate with the watsonx API, you need an API key...
```

---

## Prompt 组装：将块转化为上下文（`prompting.py`）

### prompt 模板

`prompting.py` 包含一个可见的模块级字符串 —— 不隐藏在配置文件里，不是由框架
生成的。整个模板是一个 Python 字符串，你可以在 30 秒内读完：

```python
PROMPT_TEMPLATE = """\
You are a retrieval-augmented assistant. Answer the question using only the
provided context.

If the context is insufficient, say that the provided context does not contain
enough information to answer. Do not use outside knowledge.

Cite every factual claim with the source marker for the context block that
supports it.

Question:
{question}

Context:
{context_blocks}

Answer:"""
```

四条指令，每条都有选择的原因：

| 指令 | 它防止什么 |
|---|---|
| "仅使用提供的上下文回答" | 模型从训练数据中产生幻觉 |
| "当上下文不足时说明" | 模型在不该编造时编造答案 |
| "不要使用外部知识" | 模型完全忽略检索到的块 |
| "为每个事实声明引用来源" | 模型产生无支持的说法 |

### 上下文块格式

每个检索到的块在 prompt 中变成一个块：

```
[Source: a1b2c3d4e5f67890]
Title: API Authentication Guide
Path: docs/auth.md

To authenticate with the watsonx API, you need an API key...
```

`[Source: <chunk_id>]` 标记是 prompt 和答案之间的契约。模型被指示在回答中包含
这些标记，CLI 用正则表达式提取它们来构建引用列表和源表格。

### 源表格

`format_source_table` 产生 `rag ask` 输出底部的人类可读摘要：

```
Sources:
  [Source: a1b2c3d4e5f67890]  API Authentication Guide  (docs/auth.md)
  [Source: f9e8d7c6b5a43210]  Rate Limits              (docs/limits.md)
```

这个表格将每个块 ID（在答案中作为引用出现）映射到人类可读的标题和路径。用户
可以看到像 `[Source: a1b2c3d4e5f67890]` 这样的引用，并立刻找到它来自哪个文档。

### 为什么模板是模块级字符串

将 prompt 模板放在源代码中可见的位置，使其易于审计、修改和实验。你不需要在配置
文件中搜索或解开嵌套的函数调用。你只需打开 `prompting.py` 然后阅读。这是有意
为之：后续阶段将实验不同的 prompt 策略，可见的模板使得变化清晰明了。

---

## 生成：调用 LLM（`generation.py`）

### Generator 接口

```python
class Generator(ABC):
    def generate(self, prompt: str) -> str:
        ...
```

一个方法，一个字符串输入，一个字符串输出。流水线不知道也不关心答案是来自真实
API、本地模型还是假的。

### FakeGenerator —— 为测试提供确定性答案

```python
class FakeGenerator(Generator):
    _SOURCE_RE = re.compile(r"\[Source: ([^\]]+)\]")

    def generate(self, prompt: str) -> str:
        markers = self._SOURCE_RE.findall(prompt)
        if not markers:
            return "Based on the provided context, "
                   "the context does not contain enough information to answer."
        cited = " ".join(f"[Source: {m}]" for m in markers)
        return f"Based on the provided context: {cited}"
```

这在特定意义上很巧妙：它扫描 prompt 中的 `[Source: ...]` 标记，并在答案中回显
它们。这让测试可以验证：

- prompt 正确地包含了检索到的块的源标记。
- 引用提取正则表达式（`_CITATION_RE`）正确地在答案中找到它们。
- 完整的 `ask` 流水线 —— 从检索到引用 —— 端到端工作，无需任何网络。

当没有源标记时（空检索结果），假生成器返回"上下文不足"消息，这会测试拒答路径。

### OpenAIGenerator —— 真实后端

```python
class OpenAIGenerator(Generator):
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model=None, api_key=None, base_url=None):
        self._client = OpenAI(api_key=..., base_url=...)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

设计是最小化的 —— 它恰好包装了一个 OpenAI SDK 调用，没有中间件，没有重试逻辑，
没有流式传输。对于学习实验室，这是正确的级别：你可以确切地看到什么发送到了
API，什么返回了。

**配置优先级：** 构造函数参数覆盖环境变量（`OPENAI_API_KEY`、
`OPENAI_BASE_URL`），环境变量覆盖 SDK 的默认值。CLI 层在构造生成器之前从命令
行标志和 `os.environ` 读取。

**为什么用 `gpt-4o-mini`？** 它便宜、快速，对学习实验室来说足够好。`--model`
标志让你切换到任何 OpenAI 兼容的端点 —— 本地 Ollama、Azure OpenAI 或不同的云
提供商 —— 无需更改任何代码。

---

## 这教会我们什么

检索到生成的流水线是三个清晰的阶段：

```
嵌入查询 → 余弦搜索 → 组装 prompt → 调用 LLM
```

每个阶段是一个单一的函数。契约是简单的类型：`np.ndarray` → `list[RetrievalResult]`
→ `str` → `str`。没有共享状态，没有全局配置，没有隐藏的耦合。

最重要的概念：**检索和生成是独立的关注点**。你可以用 `rag retrieve` 调试检索
—— 不涉及 LLM。你可以用假生成器测试 prompt 组装 —— 不涉及网络。你可以在不触动
对方的情况下切换嵌入模型或 LLM 提供商。这种分离是 RAG 可调试的原因，它在
这三个模块的每一行中都是可见的。
