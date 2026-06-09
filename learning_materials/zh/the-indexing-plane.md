# 索引平面 —— 文档、标准化、切块、嵌入

索引平面将磁盘上的原始文件转化为可搜索的向量。三个模块完成这项工作：
`documents.py`（加载 + 清洗）、`chunking.py`（切分成块）和 `embeddings.py`
（将文本转化为数字）。

---

## 第一步：加载文档（`documents.py`）

### 文件发现

流水线只接受 `.md` 和 `.txt` 文件：

```python
_SUPPORTED_SUFFIXES = {".md", ".txt"}
```

当你运行 `rag index --corpus corpus/watsonx-docsqa` 时，加载器用
`corpus_root.rglob("*")` 遍历整个目录树，挑选后缀在允许集合中的每个文件，
按路径排序（保证顺序确定），然后逐一加载。排序很关键 —— 不排序的话，即使
文件没有变化，索引中块的顺序也可能在不同运行之间改变。

### `load_document` 对每个文件做了什么

```python
raw_text = path.read_text(encoding="utf-8")
fmt = "markdown" if path.suffix.lower() == ".md" else "text"
```

然后构建一个包含七个字段的 `Document`：

| 字段 | 如何设置 |
|---|---|
| `doc_id` | `path.relative_to(corpus_root).as_posix()` —— 干净的相对路径 |
| `path` | 绝对路径字符串 |
| `title` | 第一个 `#` 标题（Markdown）或文件名主干 |
| `format` | `"markdown"` 或 `"text"` |
| `raw_text` | 精确的文件字节，用 UTF-8 解码 |
| `normalized_text` | `normalize_text(raw_text)` —— 清洗后的版本 |
| `raw_hash` | `sha256(raw_text)` —— 64 个十六进制字符 |

一个微妙的点：`raw_hash` 是根据 `raw_text` 计算的，不是 `normalized_text`。
这意味着哈希精确地捕获了磁盘上的文件内容。如果你更改了文件的换行符或尾部
空格，哈希就会改变 —— 即使标准化步骤产生了相同的 `normalized_text`，你也能
知道源文件发生了变化。

---

## 第二步：文本标准化

标准化在同一个文件中，使整个文本准备的过程在一个地方可见。四条规则，按顺序应用：

```python
# 规则 1：统一换行符
text = text.replace("\r\n", "\n").replace("\r", "\n")

# 规则 2：去除每行的尾部空格
lines = [line.rstrip() for line in text.split("\n")]

# 规则 3：将超过 2 个连续空行压缩为恰好 2 个
for line in lines:
    if line == "":
        blank_run += 1
        if blank_run <= 2:
            result.append(line)
    else:
        blank_run = 0
        result.append(line)

# 规则 4：保留 Markdown 标题和标点符号（隐式的 —— 不修改任何非空白字符）
```

### 每条规则存在的原因

**规则 1 —— 换行符。** 在 Windows（`\r\n`）上准备的语料库和在 macOS（`\n`）
上索引的语料库应该产生相同的块。没有这个，字符偏移会漂移。

**规则 2 —— 尾部空格。** 尾部空格是不可见的噪音。它们会加宽块而不增加信息，
并且可能导致块边界在不同编辑器之间落在不同的位置。

**规则 3 —— 空行压缩。** 一个文档可能在节之间用 5 个空行（对人类来说可读）
或 0 个空行（紧凑）分隔。压缩到 ≤2 个空行使切块器获得一致的段落边界，而不会
移除视觉上的分隔。

**规则 4 —— 保留内容。** 我们从不触碰实际的文本字符。标题、代码块、引用和
特殊标点符号都完好无损。这是一个字符级流水线 —— 不涉及分词，不涉及解析。

### 前后对比


前（为清晰起见做了夸大）：
```
# Introduction\r\n\r\n\r\n\r\nFirst paragraph.   \r\n\r\nSecond.\r\n\r\n\r\n\r\nThird.
```

后：
```
# Introduction

First paragraph.

Second.

Third.
```

标准化后的版本更短、更可预测，并为切块做好了准备。

---

## 第三步：字符切块（`chunking.py`）

### 滑动窗口算法

`chunk_document` 接受文档的 `normalized_text`，并用滑动窗口遍历它：

```python
step = chunk_size - chunk_overlap  # 例如 800 - 120 = 680

start = 0
while start < len(text):
    end = min(start + chunk_size, len(text))
    chunk_text = text[start:end]

    if chunk_text.strip():          # 跳过空或纯空白的窗口
        chunks.append(Chunk(...))

    if end == len(text):
        break                       # 到达末尾 —— 停止

    start += step
```

默认参数（`chunk_size=800`，`chunk_overlap=120`）下，步长为 680 个字符。这意味
着每个块与下一个块共享 120 个字符。

### 为什么重叠很重要

想象一个文档，其中一个关键事实跨越字符 790-810 —— 正好在两个不重叠块的边界
上。有了重叠，该事实会出现在块 1 的尾部（字符 680-800 够不到它，等一下——事实
上块 1 是 0-800，正好包含 790）和块 2 的开头。更具体地说：

```
块 1: 字符 [0,   800)
块 2: 字符 [680, 1480)   ← 与块 1 重叠 120 个字符
块 3: 字符 [1360, 2160)
```

120 字符的重叠意味着任何靠近一个块末尾开始的句子也会开始下一个块 —— 检索系统
有两次机会找到它。

权衡：更多重叠意味着更多块（索引更大，检索更慢），但更少的"中间切断"遗漏。
Phase 1 使用 120 作为合理的默认值；Phase 1.5 将使此参数可配置以便实验。

### 切片不变量实战

每个块存储 `char_start` 和 `char_end` —— 指向 `normalized_text` 的 Python 风格
左闭右开偏移量。`test_chunking.py` 中的测试验证：

```python
assert document.normalized_text[chunk.char_start:chunk.char_end] == chunk.text
```

这在结构上是成立的 —— 切块器就是字面地切片 `text[start:end]` —— 但在测试中验证
它可以捕获回归，如果将来有人在切块和存储之间添加了标准化步骤。

### 处理的边界情况

**短文档。** 如果文档比 `chunk_size` 短，你会得到一个包含整个文本的块 ——
`end = min(start + chunk_size, len(text))` 自然处理了这种情况。然后循环看到
`end == len(text)`，生成这个块然后退出。

**纯空白块。** `if chunk_text.strip()` 跳过只包含空格、制表符或换行符的窗口。
这防止索引被永远不会被有效检索到的噪音填满。

**尾部窗口。** 当最后一个窗口的末尾恰好等于 `len(text)` 时，循环立即退出 ——
不会向前滑动创建完全包含在当前窗口中的冗余尾部块。

### chunk_size 与 chunk_overlap 的不变量

实现强制执行：

```python
if chunk_overlap >= chunk_size:
    raise ValueError(...)
```

如果 overlap 等于或超过 chunk_size，步长会变为零或负数，滑动窗口将永远不会前进。
这在开始时被捕获，使错误消息清晰明了。

---

## 第四步：嵌入（`embeddings.py`）

### 接口契约

每个嵌入器承诺一件事：

```python
class Embedder(ABC):
    def embed(self, texts: list[str]) -> np.ndarray:
        # 返回形状 (len(texts), dim)，dtype float32
```

检索代码从不检查是哪个类产生的向量。它只关心形状。这是索引平面和检索平面之间的
**接口边界**。

### FakeEmbedder —— 确定性且免费

假嵌入器使用哈希将任何文本转化为单位向量：

```python
seed = int.from_bytes(sha256(text).digest()[:4], byteorder="little")
rng = np.random.default_rng(seed)
vec = rng.standard_normal(dim).astype(np.float32)
vec = vec / np.linalg.norm(vec)  # L2 归一化到单位长度
```

关键属性：
- **相同文本 → 相同向量。** 总是如此。哈希是确定性的，NumPy 的
  `default_rng(seed)` 跨运行可重现。
- **不同文本 → 不相关的向量。** SHA-256 的雪崩效应意味着甚至 "hello" 和
  "hello!" 也会产生不相关的种子。
- **向量是单位长度。** L2 归一化使余弦相似度可以作为一个简单的点积计算
  —— 查询时不需要除以范数。
- **无需下载、无网络、无 GPU。** 纯 Python + NumPy。所有 241 个 Phase 1
  测试都使用这个嵌入器。

### SentenceTransformerEmbedder —— 真家伙

```python
class SentenceTransformerEmbedder(Embedder):
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name=..., local_files_only=False):
        self._model = SentenceTransformer(model_name, local_files_only=...)
```

几个设计说明：

**为什么用 `all-MiniLM-L6-v2`？** 它是最小的可行嵌入模型之一（384 维，约 80MB），
可以在 CPU 上运行，并产生合理的语义向量。对于学习实验室，小且本地胜过大型且依赖
API。

**`normalize_embeddings=True`** 在 `encode` 调用中设置：

```python
vecs = self._model.encode(
    texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
)
```

这使真实嵌入器的输出与假嵌入器的输出一致 —— 两者都产生单位向量，因此相同的检索
代码可以与两者配合使用。

**`dim` 是动态发现**的，通过 `get_embedding_dimension()` 从加载的模型中获取，
而不是硬编码。如果以后切换到不同的模型，维度会自动更新。

**只需要一次网络连接。** 首次使用时，模型权重从 Hugging Face 下载并缓存到本地。
之后，所有后续运行都是完全离线的。

---

## 这教会我们什么

索引平面有三个清晰的阶段：加载 → 切块 → 嵌入。每个阶段都有一个做一件事的函数。
阶段之间的契约是简单的类型（`Document`、`Chunk`、`np.ndarray`）—— 没有回调、
没有注册表、没有框架连接。这就是"可读代码"在实践中意味着什么：你可以打开
`documents.py`，从头到尾读 `load_document`，然后确切地理解一个文件发生了什么。

将嵌入器放在 ABC 后面的设计选择意味着相同的检索代码可以配合一个 10 行的哈希
函数（测试）和一个 80MB 的 transformer 模型（生产）工作。这种模式 —— 一个狭窄
的接口加上一个确定性的假实现 —— 贯穿 Phase 1，使整个流水线无需网络访问即可测试。
