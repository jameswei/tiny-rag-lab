# RAG 数据流 —— 数据如何从文件变成答案

实现围绕一小组 dataclass 构建。它们不只是类型定义 —— 它们是流水线
每个阶段之间的**契约**。如果你理解了这些类型以及它们如何在 CLI 中流转，
你就理解了整个架构。

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

每个箭头都是一条有明确规则的转换。让我们逐一走过。

---

## Document —— 加载到内存中的一个文件

`Document` 代表语料库中的一个源文件。你可以在 `tiny_rag_lab/models.py` 中找到它。

```python
@dataclass
class Document:
    doc_id: str           # 例如 "docs/getting-started.md"
    path: str             # 例如 "/home/user/corpus/docs/getting-started.md"
    title: str            # 例如 "Getting Started with watsonx"
    format: str           # "markdown" 或 "text"
    raw_text: str         # 文件的精确内容，未做任何改动
    normalized_text: str  # 清洗后的版本，供切块使用
    raw_hash: str         # raw_text 的 SHA-256 哈希，64 个十六进制字符
```

### 为什么 `doc_id` 是 POSIX 路径而非绝对路径

当你运行 `rag index --corpus corpus/watsonx-docsqa` 时，代码相对于语料库根目录计算 `doc_id`：

```python
doc_id = path.relative_to(corpus_root).as_posix()
# 输入:  /home/user/tiny-rag-lab/corpus/watsonx-docsqa/docs/getting-started.md
# 输出: "docs/getting-started.md"
```

这有两个原因。首先，索引变得可移植 —— 你可以移动语料库并重建索引，而不会改变
任何标识符。其次，当 CLI 打印检索结果时，你看到的是像 `docs/faq.md` 这样干净的
路径，而不是没人想读的长绝对路径。

### 为什么同时保留 `raw_text` 和 `normalized_text`

`raw_text` 是磁盘上的精确文件内容。`normalized_text` 是切块器实际处理的文本。
它们不同是因为流水线会清理换行符、尾部空格和空行序列（详见索引深度剖析中的
`normalize_text`）。但你仍然需要保留 `raw_text` —— `raw_hash` 是根据它计算的，
所以如果磁盘上的文件发生了变化，你可以检测到变化，而不必重新读取所有内容。

### 标题是如何提取的

对于 Markdown 文件，加载器扫描**第一个 `#` 标题**：

```python
if stripped.startswith("# "):
    return stripped[2:].strip()
```

如果没有 H1（或文件是纯文本），标题回退到**文件名主干** —— `faq.md` 变成
`"faq"`。这很简单，但是有意为之：Phase 1 不需要花哨的标题提取。关键点是每个
文档都有一个人类可读的标签，用于 trace 输出。

---

## Chunk —— 原子检索单元

一个 `Document` 可能有几页长。LLM 的 prompt 只能容纳有限的上下文。所以我们
将每个文档拆分成有重叠的**块（chunk）**—— 检索系统可以返回的最小单元。

```python
@dataclass
class Chunk:
    chunk_id: str      # 例如 "a1b2c3d4e5f67890"
    doc_id: str        # 这个块来自哪个文档
    text: str          # 块的实际内容
    char_start: int    # 这个块在 normalized_text 中的起始位置
    char_end: int      # 结束位置（不包含，Python 切片风格）
    metadata: dict     # 始终包含 title、path、format、raw_hash
```

### 关键不变量

每个块必须满足：

```python
document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
```

这在 `tests/test_chunking.py` 中被测试，并在切块器的实现中得到保证。如果这个
不变量被破坏，检索结果会指向错误的文本，引用就变成了谎言。这个不变量是块和
其源文档之间唯一的真值锚点。

### 确定性的 `chunk_id`

ID 通过一个 SHA-256 哈希生成，输入是可预测的：

```python
def make_chunk_id(doc_id: str, char_start: int, chunk_text: str) -> str:
    raw = f"{doc_id}:{char_start}:{chunk_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

需要注意的几点：

- ID 只依赖于 **doc_id、偏移量和文本** —— 不依赖随机种子、时间戳或嵌入模型。
- 用相同的块大小重新索引相同的语料库会产生**相同的块 ID**。这对于引用在重建
  后保持稳定至关重要。
- 它使用前 16 个十六进制字符（64 位）。在几千个块中碰撞概率可以忽略，而且 16
  个字符比完整的 64 字符哈希在终端中更容易阅读。

### metadata 必须包含什么

`metadata` 字典始终包含：

| 键 | 值 | 原因 |
|---|---|---|
| `title` | 文档标题 | 显示在 trace 输出中 |
| `path` | 文件系统路径 | 让用户可以找到源文件 |
| `format` | `"markdown"` 或 `"text"` | 保留原始文件类型 |
| `raw_hash` | raw_text 的 SHA-256 | 以后检测源变化 |

---

## RetrievalResult —— 带分数的排名块

当检索引擎搜索索引时，它按与用户查询的相似度对块进行排名。每个结果将块、
分数和排名捆绑在一起：

```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float    # 余弦相似度，范围 [-1, 1]
    rank: int       # 从 1 开始的排名（rank 1 = 最佳匹配）
```

### 为什么 rank 从 1 开始

排名从 1 开始，因为这是人类期望的 —— "顶部结果"意味着排名 1。检索代码使用
`enumerate(results, start=1)` 生成它们。这是小事，但在 CLI 输出中很重要：
`Rank 1  score=0.9234` 一眼就能读懂。

### score 从何而来

`score` 是查询向量和块的嵌入向量之间的余弦相似度。值接近 1.0 表示块与问题在
语义上相似；接近 0 表示无关；负值表示方向相反（理论上可能，但真实嵌入中很少见）。
精确的数学在检索深度剖析中讲解。

---

## RetrieveTrace 和 AskTrace —— 一次运行的记录

Phase 1.7 将可观测性类型移动到了 `tiny_rag_lab/trace.py`。`rag retrieve`
构建 `RetrieveTrace`，`rag ask` 构建 `AskTrace`。两个命令都会打印由 formatter
生成的 trace 输出，也都可以通过 `--trace-out` 写出 JSON。

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

```python
@dataclass
class RetrieveTrace:
    query: str
    retriever: str
    top_k: int
    chunks: list[ChunkTrace]
    latency_by_stage: dict[str, float]
```

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

`RetrieveTrace` 用于在不生成答案的情况下检查搜索结果。`AskTrace` 用于检查完整
RAG 路径：检索、prompt 组装、生成和引用。

`latency_by_stage` 让阶段边界变得明确：

| 键 | 测量的内容 |
|---|---|
| `"load"` | 从磁盘加载索引的时间 |
| `"embed"` | 计算查询嵌入的时间 |
| `"retrieve"` | 对块进行排名的时间 |
| `"prompt_assembly"` | 为 `rag ask` 构建 prompt 的时间 |
| `"generate"` | LLM 生成答案的时间 |

BM25 的 retrieve trace 不包含 `"embed"`，因为 BM25 不使用嵌入模型。dense 和
hybrid retrieve trace 会包含它。

这些数字在 retrieve 和 ask trace 中可见，但两者的字段不同：

```
Retrieve trace:
latency   : load=0.006s  embed=0.012s  retrieve=0.001s

Ask trace:
latency   : load=0.006s  embed=0.012s  retrieve=0.001s  prompt_assembly=0.000s  generate=1.234s
```

这立刻告诉你流水线在哪里花费时间。在 Phase 1 中使用本地嵌入器和 API 生成器时，
`generate` 会以数量级优势占据主导。

---

## CLI 如何将它们串联起来

CLI 在 `tiny_rag_lab/cli.py` 中。三个命令，每个都建立在前一个之上：

### `rag index` —— 构建索引

```
load_documents(corpus_root)          → list[Document]
  └─ chunk_documents(docs, ...)      → list[Chunk]
       └─ embedder.embed(texts)      → np.ndarray
            └─ write_index(...)      → .tiny-rag/index/
```

关键洞察：索引是一个**流水线**，不是一个整体。每个函数做一件事。CLI 只是按顺序
调用它们，使用用户的参数。

### `rag retrieve` —— 搜索但不生成

```
load_index(index_dir)               → LoadedIndex
  └─ 按需嵌入查询                  → query vector
       └─ 检索 top-k 块             → list[RetrievalResult]
            └─ 构建 RetrieveTrace   → 终端输出 / 可选 JSON
```

检索是 RAG 的核心。`rag retrieve` 让你在 LLM 介入之前检查搜索结果。这是调试的
入口：如果返回了错误的块，没有任何 prompt 能修复它。

### `rag ask` —— 完全端到端

```
load_index → 嵌入查询 → 检索 → 组装 prompt → 生成 → 构建 AskTrace
```

`cmd_ask` 调用 `retrieve_by_vector`（纯排名函数）而不是 `retrieve`（后者也会
嵌入），因为它要分别测量嵌入和检索的时间。prompt 组装和生成与检索完全解耦
—— 它们只依赖 `list[RetrievalResult]`。

---

## 这教会我们什么

这些 dataclass 是这个学习实验室的**真正架构**。CLI 只是按顺序调用它们的薄壳。
如果你能凭记忆画出 Document → Chunk → RetrievalResult → RetrieveTrace / AskTrace 链条，
你就能推理流水线的任何变化 —— 新的切块器、不同的检索器、更好的 prompt、
更丰富的 trace —— 因为你确切知道每个阶段接收和产生什么数据。

接下来的深度剖析将逐一深入各个阶段：

- **索引平面** —— 文档如何变成块和向量
- **检索与生成** —— 相似度搜索和 prompt 组装的原理
- **持久化与测试** —— 索引如何保存、加载和验证
- **可观测性与调试** —— trace 记录如何解释一次运行
