# 持久化与测试 —— 保存索引并证明它有效

构建索引代价昂贵（用真实模型嵌入数千个块需要时间），但重用便宜。两个模块处理
持久化：`index_writer.py` 保存索引，`index_loader.py` 将其读回。

整个流水线无需网络或 API 密钥即可测试。本文档涵盖索引格式、往返契约以及假后端
如何使测试成为可能。

---

## 索引格式：三个文件，一个契约

`rag index` 完成后，`.tiny-rag/index/` 包含三个文件：

```
.tiny-rag/index/
  manifest.json     — 索引运行的元数据
  chunks.jsonl      — 每行一个块，不含向量
  embeddings.npz    — float32 矩阵 + 并行的 chunk_ids
```

### 为什么是三个文件而不是一个

将块（JSONL）与嵌入（NPZ）分开，使你可以使用 `cat` 或 `head` 检查块 —— 它们是
纯文本。嵌入是以压缩的 NumPy 格式存储的二进制浮点数 —— 人类不可读，但紧凑且加载
快速。

如果它们合并在一起，你每次都需要一个 Python 脚本才能查看索引中的内容。对于学习
实验室，可检查性比最小文件数更重要。

### `manifest.json` —— 索引的出生证

```json
{
  "schema_version": "1.0",
  "corpus_root": "/home/user/tiny-rag-lab/corpus/watsonx-docsqa",
  "created_at": "2026-06-09T10:30:00+00:00",
  "document_count": 42,
  "chunk_count": 312,
  "chunk_size": 800,
  "chunk_overlap": 120,
  "embedding_backend": "SentenceTransformerEmbedder",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "corpus_files": [
    {"doc_id": "docs/auth.md", "path": "/home/...", "raw_hash": "abc123..."},
    ...
  ]
}
```

manifest 记录了产生索引的一切 —— 语料库路径、块参数、嵌入模型以及每个源文件的
哈希。这意味着你可以查看三周前构建的索引，而不必翻找 shell 历史就能确切知道
它是如何制作的。

`schema_version` 字段（"1.0"）存在是为了让后续阶段可以演化格式。如果 Phase 1.5
向索引添加了 BM25 分数，它可以提升版本号，加载器可以同时处理旧格式和新格式。

### `chunks.jsonl` —— 每行一个 JSON 对象

每行是一个序列化的 `Chunk`：

```json
{"chunk_id":"a1b2c3...","doc_id":"docs/auth.md","text":"To authenticate...","char_start":0,"char_end":800,"metadata":{"title":"API Auth","path":"/home/...","format":"markdown","raw_hash":"abc123..."}}
```

JSONL（每行一个 JSON 对象，用换行分隔）被选择而不是 JSON 数组，有两个原因。
第一，它是可流式传输的 —— 可以逐行读取而无需将整个文件加载到内存中。第二，它
可以轻松地用 `head chunks.jsonl` 或 `wc -l chunks.jsonl` 检查（行数就是块数量）。

嵌入向量**不**存储在 chunks.jsonl 中 —— 它们存在于 NPZ 文件中。这种分离保持
人类可读的一侧轻量，二进制一侧高效。

### `embeddings.npz` —— 向量矩阵

```python
np.savez(
    "embeddings.npz",
    embeddings=embeddings,    # float32 数组，形状 (N, dim)
    chunk_ids=chunk_ids,      # 字符串数组，形状 (N,)
)
```

两个数组存储在一个压缩文件中。关键契约：**`embeddings` 中的第 i 行必须对应
`chunk_ids[i]`**。写入器以与块列表相同的顺序存储 `chunk_ids`，加载器通过比较
NPZ 中的 `chunk_ids` 和 chunks.jsonl 中的 `chunk_id` 字段来验证这个顺序。

### 行顺序契约

这是整个持久化层中最重要的约束：

```python
# 写入器中：
chunk_ids = np.array([c.chunk_id for c in chunks])
np.savez(..., embeddings=embeddings, chunk_ids=chunk_ids)

# 加载器中 —— 已验证：
chunk_ids_jsonl = [c.chunk_id for c in chunks]
chunk_ids_npz = [str(cid) for cid in data["chunk_ids"]]
if chunk_ids_jsonl != chunk_ids_npz:
    raise ValueError("chunk_ids mismatch between chunks.jsonl and embeddings.npz")
```

如果这个顺序发生漂移，检索将返回错误的块 —— 块 3 的嵌入行将匹配块 7 的文本，
引用将变成谎言。加载器在读取时捕获这一点，而不是悄悄地产生错误结果。

---

## 往返完整性：最重要的测试

`tests/test_persistence_roundtrip.py`（P1-T19）证明了保存再加载不会破坏检索：

```
1. 使用 FakeEmbedder 从测试语料库构建索引
2. 用 write_index() 将其写入磁盘
3. 用 load_index() 将其加载回来
4. 检索一个查询
5. 断言相同的块排名最高
```

这是一个覆盖四个模块（写入器、加载器、检索、模型）的测试，可以捕获整类 bug：
错误的序列化、顺序漂移、缺失字段或编码问题。如果这个测试通过，你就可以信任
索引格式有效。

该测试只使用 `tests/fixtures/corpus/` 下的测试语料库 —— 几个签入 git 的小型
Markdown 和纯文本文件。没有 `watsonxDocsQA` 下载，没有真实嵌入，没有网络。
这是有意为之：往返测试应该始终运行，即使在零设置的首次克隆上也能运行。

---

## 假后端模式

Phase 1 使用两个假后端：`FakeEmbedder` 和 `FakeGenerator`。它们共同使所有
241 个测试在没有网络、没有 API 密钥、无需下载 80MB transformer 模型的情况下
运行。

### 这个模式为什么有效

两个假实现共享相同的设计：

1. **狭窄的接口。** `Embedder.embed()` 和 `Generator.generate()` 是流水线调用
   的唯一方法。假实现只需要满足这些。
2. **确定性。** 相同的输入总是产生相同的输出。这意味着测试可以断言精确值，
   而不仅仅是"足够接近"的范围。
3. **自包含。** 除了 Python 的标准库和 NumPy 之外没有外部依赖。没有下载，
   没有配置文件，没有环境变量。

### 用假实现可以测试什么

| 测试关注点 | 假实现如何实现 |
|---|---|
| 块 ID 稳定性 | FakeEmbedder 不影响 ID；ID 是基于哈希的 |
| 块 → 嵌入流水线 | FakeEmbedder.embed() 用正确的文本调用 |
| 检索排名 | FakeEmbedder 产生已知向量 → 已知排名 |
| Prompt 组装 | FakeGenerator 回显 prompt 中的源标记 |
| 引用提取 | FakeGenerator 包含标记 → 正则找到它们 |
| CLI 命令连接 | FakeEmbedder 和 FakeGenerator 通过 monkeypatch 替换 |
| 索引持久化 | FakeEmbedder 创建小型确定性 NPZ 文件 |
| 空结果处理 | FakeGenerator 的无标记路径测试拒答 |

### 用假实现不能测试什么

| 限制 | 原因 |
|---|---|
| 嵌入质量 | 假向量是随机哈希，不是语义表示 |
| 真实检索相关性 | 假向量上的余弦相似度没有意义 |
| 答案忠实度 | FakeGenerator 不读上下文 —— 它回显标记 |
| LLM prompt 合规性 | 假实现不遵循基于上下文的指令 |
| 嵌入模型行为 | 只有 `SentenceTransformerEmbedder` 测试覆盖这些 |

这些差距是有意的。Phase 1.6（评估框架）将用真实嵌入添加真实语料库测试来衡量
实际质量。假后端用于**正确性** —— 代码是否工作？真后端用于**质量** —— 它是否
工作得好？

### CLI 测试如何交换后端

CLI 测试使用 pytest 的 `monkeypatch` 来替换工厂函数：

```python
# 在 conftest 或测试文件中：
monkeypatch.setattr(
    "tiny_rag_lab.cli._make_embedder",
    lambda model_name=None: FakeEmbedder(dim=8)
)
monkeypatch.setattr(
    "tiny_rag_lab.cli._make_generator",
    lambda args: FakeGenerator()
)
```

这就是为什么 `cli.py` 将 `_make_embedder()` 和 `_make_generator()` 作为独立
函数，而不是内联构造。这是一个故意的接缝 —— 一个可以在不改变 CLI 逻辑的情况下
将真实实现替换为测试替身的点。

---

## 加载器中的验证：纵深防御

`load_index()` 验证的内容不止行顺序契约：

```python
# 所有三个文件必须存在
for p in (manifest_path, chunks_path, embeddings_path):
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {p}")

# 嵌入行数必须匹配块数量
if embeddings.shape[0] != len(chunks):
    raise ValueError(
        f"embeddings row count {embeddings.shape[0]} != chunk count {len(chunks)}"
    )
```

这些检查捕获常见的失败模式：

- **缺失文件。** 有人在 `rag index` 之前运行了 `rag retrieve`，或手动删除了
  一个文件。
- **行数不匹配。** NPZ 文件在失败的写入中被截断，或旧的 chunks.jsonl 与新的
  embeddings.npz 配对。
- **ID 顺序不匹配。** 写入器中的 bug 以错误的顺序存储了块 ID。

每个检查产生一条带有实际值的特定错误消息，因此你确切知道出了什么问题，而不必
猜测。

---

## 这教会我们什么

持久化层大约 150 行（写入器 + 加载器合计），但它教会了两个重要的系统设计概念：

**将人类可读数据与二进制数据分开。** chunks.jsonl 是为人类和简单工具准备的；
embeddings.npz 是为机器准备的。合并它们会使两者都更难使用。

**在加载时验证，而不仅仅在保存时。** 写入器假定正确性；加载器证明它。每个在
加载时检查的不变量都是一个在悄悄产生错误答案之前被捕获的 bug。带交叉验证的
三文件格式是防御性 I/O 的一个微型案例研究。

假后端模式 —— 狭窄的接口、确定性的实现、为测试设计的接缝 —— 是 241 个测试能
在首次克隆时以 3.6 秒运行的原因。这是一种可以扩展到你流水线中任何真实后端缓慢、
昂贵或不确定的阶段的模式。
