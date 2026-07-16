const translations = {
  en: {
    // Nav
    'nav.about':        'About',
    'nav.pipeline':     'Pipeline',
    'nav.evidence':     'Evidence',
    'nav.studio':       'Visual Lab',
    'nav.capabilities': 'Capabilities',
    'nav.stack':        'Stack',
    'nav.cli':          'CLI',
    'nav.github':       'GitHub →',

    // Hero
    'hero.kicker':    'A LEARNING-FIRST RAG LAB',
    'hero.tagline':   'See classic RAG actually run — real documents, real retrieval, real citations.',
    'hero.why':       'Most tutorials hide retrieval behind a framework call. This lab keeps the calculations, candidate movement, evidence, and generation boundary visible so you can reason about each result.',
    'hero.cta.github': 'View source on GitHub',
    'hero.cta.studio': 'Run the local lab ↓',

    // Trace hero (documents -> chunks -> evidence -> answer narrative)
    'trace.label': 'From documents to a cited answer',
    'trace.stage.documents': 'Documents',
    'trace.stage.chunks': 'Chunks',
    'trace.stage.evidence': 'Evidence',
    'trace.stage.answer': 'Answer',
    'trace.stage.answer.icon': 'ANSWER',
    'evidence.label.question': 'Question',

    // About
    'about.label': 'ONE CORE · TWO WAYS TO LEARN',
    'about.title': 'Learn the mechanics, then make them move.',
    'about.subtitle': 'The visual workspace and CLI expose the same project-owned RAG artifacts. Start guided; go deeper when you are ready.',
    'about.web.label': 'START HERE',
    'about.web.title': 'Visual workspace',
    'about.web.body': 'Replay real-corpus lessons, follow a live retrieval course, and run your own index, reranking, evaluation, and provider experiments.',
    'about.web.cta': 'Explore the learning workspace →',
    'about.cli.label': 'GO DEEPER',
    'about.cli.title': 'Direct CLI',
    'about.cli.body': 'Repeat configurations, compare results, inspect raw output, and follow the same mechanics command by command.',
    'about.cli.cta': 'See the CLI path →',

    // Stats
    'stat.planes':    'RAG Architecture Planes',
    'stat.entrypoints':'Local Learning Entrypoints',
    'stat.retrievers':'Interactive Retrieval Modules',
    'stat.chunking':  'Chunking Strategies',

    // What makes it different (About)
    'diff.label':   'WHAT MAKES IT DIFFERENT',
    'diff.1.title': 'Real evidence, not a toy demo',
    'diff.1.proof': 'Four guided replays and six live retrieval modules use 40 real Cloudflare docs — not a synthetic one-file example.',
    'diff.2.title': 'One core, two ways in',
    'diff.2.proof': 'The visual lab and CLI use the same project-owned RAG components and expose the same mechanics and artifacts — there is no demo-only pipeline.',
    'diff.3.title': 'Failure is part of the curriculum',
    'diff.3.proof': 'A dedicated Failure Lab replays bad chunking, buried evidence, and stale documents — and shows whether a fix actually helps.',
    'diff.4.title': 'No framework black box',
    'diff.4.proof': 'Chunking, retrieval, prompting, and citations are hand-written and readable — no LangChain, LlamaIndex, or Haystack wrapper hiding the mechanics.',

    // Pipeline
    'pipeline.title':    'The RAG pipeline',
    'pipeline.subtitle': 'Every stage is implemented directly — no framework wrapper, no hidden magic.',

    // SVG labels
    'svg.indexing':      'Indexing',
    'svg.querytime':     'Query time',
    'svg.corpus':        'Corpus',
    'svg.corpus.sub':    'docs / Markdown',
    'svg.chunks':        'Chunks',
    'svg.chunks.sub1':   'fixed / structural',
    'svg.chunks.sub2':   '/ semantic',
    'svg.embeddings':    'Embeddings',
    'svg.embeddings.sub':'MiniLM vectors',
    'svg.index':         'Local Index',
    'svg.index.sub':     'NumPy · BM25',
    'svg.query':         'Query',
    'svg.query.sub':     'user question',
    'svg.retrieve':      'Retrieve',
    'svg.retrieve.sub':  'dense · BM25 · hybrid',
    'svg.generate':      'Generate',
    'svg.generate.sub':  'grounded prompt → LLM',
    'svg.answer':        'Answer',
    'svg.answer.sub':    '+ citations',

    // Planes
    'plane.indexing':   'Indexing plane',
    'plane.retrieval':  'Retrieval plane',
    'plane.generation': 'Generation plane',
    'plane.eval':       'Evaluation & observability plane',

    // Real Evidence Showcase
    'evidence.label':    'REAL EVIDENCE',
    'evidence.title':    'The artifacts are the lesson',
    'evidence.subtitle': 'Every citation below is a real retrieval over the bundled Cloudflare corpus — not a mockup.',
    'evidence.1.question': 'How can a Worker use a Durable Object namespace, stable ID, and stub to send requests for the same entity to one stateful coordinator?',
    'evidence.1.excerpt':  'Bind a Durable Object namespace to the Worker, derive a stable Durable Object ID for the entity, obtain that object\'s <mark class="highlight">stub</mark>, and send every request for the same entity to that one coordinator.',
    'evidence.1.footer':   'Retrieved, packed into context, and cited — not paraphrased from memory.',
    'evidence.2.question': 'What is the tradeoff between Workers KV eventual consistency for global configuration and R2 object storage for mutable files?',
    'evidence.2.excerpt':  'Use KV for configuration that is read globally and <mark class="highlight">tolerates eventual consistency</mark>. Use R2 for mutable file or object content that needs strong read-after-write behavior.',
    'evidence.2.footer':   'Two source documents, one grounded comparison.',

    // Visual lab
    'studio.title':        'A rich local workspace for learning RAG',
    'studio.subtitle':     'Learn through real documents: start with guided replays, inspect retrieval math live, then run experiments of your own.',
    'studio.replay.title': 'Start with a replay',
    'studio.replay.desc':  'Follow a recorded real-corpus lesson from source documents to a cited answer, one visible stage at a time.',
    'studio.inspect.title':'Make retrieval visible',
    'studio.inspect.desc': 'Follow six live modules through BM25 math, cosine similarity, NumPy and Qdrant, hybrid fusion, reranking, and evaluation.',
    'studio.compare.title':'Learn from comparisons',
    'studio.compare.desc': 'Compare two retrieval configurations over 16 reviewed questions, inspect every result, then carry reranking into free-form Explore.',
    'studio.shot.guided': 'Guided lesson: follow ranked evidence back to its source.',
    'studio.shot.explore':'Explore: change the retrieval setup and inspect what comes back.',
    'studio.shot.failure':'Failure Lab: compare a baseline run against a fix, side by side.',
    'studio.shot.inspect':'Build & Inspect: examine chunks, vectors, and source provenance directly.',
    'studio.cta':         'Run the local studio with Docker Compose →',
    'studio.local-note':  'Core lab workflows run locally with no account required. Live Ask contacts only the provider you configure.',

    // Capabilities
    'cap.label':    'WHAT IT COVERS',
    'cap.title':    'Four planes, one inspectable pipeline',
    'cap.subtitle': 'Everything maps to one visible architecture and the same project-owned artifacts.',
    'cap.indexing.title':   'Indexing',
    'cap.indexing.points':  '<li>Document loading, normalization, and metadata</li><li>Fixed-character, structural, and experimental semantic chunking</li><li>Local embeddings (MiniLM) and an inspectable NumPy index</li>',
    'cap.retrieval.title':  'Retrieval',
    'cap.retrieval.points': '<li>Live BM25, cosine, and hybrid RRF explanations</li><li>The same vectors through inspectable NumPy and optional Qdrant</li><li>Cross-encoder candidate-to-final reranking</li>',
    'cap.generation.title':  'Generation',
    'cap.generation.points': '<li>Token-budgeted context packing with traceable omitted chunks</li><li>Grounded prompt assembly and OpenAI-compatible generation</li><li>Citations and abstention when evidence is insufficient</li>',
    'cap.eval.title':  'Evaluation & Observability',
    'cap.eval.points': '<li>Browser A/B retrieval evaluation over 16 reviewed questions</li><li>Answer metrics via LLM-as-judge: faithfulness, relevance, correctness</li><li>Per-query traces and a curated Failure Lab</li>',

    // Stack
    'stack.title':    'Tech stack',
    'stack.subtitle': 'Kept deliberately minimal — every moving part is visible in the code.',
    'stack.l.language':   'Language',
    'stack.l.interface':  'Interface',
    'stack.v.interface':  'CLI + local visual lab',
    'stack.l.embeddings': 'Embeddings',
    'stack.l.index':      'Vector index',
    'stack.v.index':      'NumPy + optional local Qdrant',
    'stack.l.generation': 'Generation',
    'stack.v.generation': 'OpenAI-compatible API',
    'stack.l.test':       'Test backends',
    'stack.v.test':       'Fake embedder + fake generator (fully offline)',
    'stack.l.corpus':     'Corpus',
    'stack.v.corpus':     'Cloudflare docs + watsonxDocsQA',
    'stack.l.deps':       'Dependencies',
    'stack.note': 'No LangChain, LlamaIndex, or Haystack wrapper. The core RAG mechanics are implemented directly.',

    // CLI
    'cli.title':    'A clean CLI interface',
    'cli.subtitle': 'The direct, repeatable companion to the visual workspace: index a corpus, retrieve evidence, and ask a question with the same inspectable mechanics.',
    'cli.link':     'Full CLI reference and docs on GitHub →',

    // CTA
    'cta.title': 'Interested in the source?',
    'cta.desc':  'Clone it, run it, or just read how it\'s built — all on GitHub.',
    'cta.btn':   'View on GitHub',

    // Footer
    'footer.built': 'Built by',
  },

  zh: {
    // Nav
    'nav.about':        '关于',
    'nav.pipeline':     '流程',
    'nav.evidence':     '证据',
    'nav.studio':       '可视化实验室',
    'nav.capabilities': '功能',
    'nav.stack':        '技术栈',
    'nav.cli':          'CLI',
    'nav.github':       'GitHub →',

    // Hero
    'hero.kicker':    '一个以学习为先的 RAG 实验室',
    'hero.tagline':   '亲眼看经典 RAG 运行——真实文档、真实检索、真实引用。',
    'hero.why':       '许多教程把检索隐藏在一次框架调用里。这个实验室让计算、候选排名移动、证据和生成边界保持可见，使你能够解释每一次结果。',
    'hero.cta.github': '在 GitHub 查看源码',
    'hero.cta.studio': '运行本地实验室 ↓',

    // Trace hero
    'trace.label': '从文档到有据可依的答案',
    'trace.stage.documents': '文档',
    'trace.stage.chunks': '文本块',
    'trace.stage.evidence': '证据',
    'trace.stage.answer': '答案',
    'trace.stage.answer.icon': '答案',
    'evidence.label.question': '问题',

    // About
    'about.label': '一个核心 · 两种学习方式',
    'about.title': '理解机制，再亲手让它运行。',
    'about.subtitle': '可视化工作区和 CLI 都暴露同一套项目自有 RAG 产物。先从引导开始，再逐步深入。',
    'about.web.label': '从这里开始',
    'about.web.title': '可视化工作区',
    'about.web.body': '回放真实语料课程，跟随实时检索课程，并亲手进行索引、重排序、评估和服务商实验。',
    'about.web.cta': '探索学习工作区 →',
    'about.cli.label': '继续深入',
    'about.cli.title': '直接使用 CLI',
    'about.cli.body': '重复配置、比较结果、检查原始输出，并逐条命令跟随同一套机制。',
    'about.cli.cta': '查看 CLI 路径 →',

    // Stats
    'stat.planes':    'RAG 架构层',
    'stat.entrypoints':'本地学习入口',
    'stat.retrievers':'交互式检索模块',
    'stat.chunking':  '分块策略',

    // What makes it different (About)
    'diff.label':   '它的独特之处',
    'diff.1.title': '真实证据，而非玩具演示',
    'diff.1.proof': '四个引导回放和六个实时检索模块使用 40 篇真实 Cloudflare 文档——不是合成的单文件示例。',
    'diff.2.title': '一套核心，两种入口',
    'diff.2.proof': '可视化工作区和 CLI 使用同一套项目自有 RAG 组件，并暴露相同的机制与产物——不存在仅供演示的独立流水线。',
    'diff.3.title': '失败也是课程的一部分',
    'diff.3.proof': '专门的失败实验室回放糟糕的分块、被埋没的证据和过期文档——并展示修复是否真的有效。',
    'diff.4.title': '没有框架黑箱',
    'diff.4.proof': '分块、检索、提示词组装和引用均为手写且可读——没有 LangChain、LlamaIndex 或 Haystack 封装掩盖机制。',

    // Pipeline
    'pipeline.title':    'RAG 流程',
    'pipeline.subtitle': '每个阶段直接实现——没有框架包装，没有隐藏魔法。',

    // SVG labels
    'svg.indexing':      '索引阶段',
    'svg.querytime':     '查询阶段',
    'svg.corpus':        '语料库',
    'svg.corpus.sub':    '文档 / Markdown',
    'svg.chunks':        '文本块',
    'svg.chunks.sub1':   '固定 / 结构化',
    'svg.chunks.sub2':   '/ 语义',
    'svg.embeddings':    '嵌入向量',
    'svg.embeddings.sub':'MiniLM 向量',
    'svg.index':         '本地索引',
    'svg.index.sub':     'NumPy · BM25',
    'svg.query':         '查询',
    'svg.query.sub':     '用户问题',
    'svg.retrieve':      '检索',
    'svg.retrieve.sub':  '稠密 · BM25 · 混合',
    'svg.generate':      '生成',
    'svg.generate.sub':  '提示词 → LLM',
    'svg.answer':        '答案',
    'svg.answer.sub':    '+ 引用来源',

    // Planes
    'plane.indexing':   '索引层',
    'plane.retrieval':  '检索层',
    'plane.generation': '生成层',
    'plane.eval':       '评估与可观测层',

    // Real Evidence Showcase
    'evidence.label':    '真实证据',
    'evidence.title':    '产物本身就是课程',
    'evidence.subtitle': '下面的每一条引用都是对内置 Cloudflare 语料的真实检索——不是模拟数据。',
    'evidence.1.question': 'Worker 如何通过 Durable Object 命名空间、稳定 ID 和 stub，将同一实体的请求都发送给同一个有状态协调者？',
    'evidence.1.excerpt':  '将 Durable Object 命名空间绑定到 Worker，为该实体推导出稳定的 Durable Object ID，获取该对象的 <mark class="highlight">stub</mark>，并将同一实体的每个请求都发送给这一个协调者。',
    'evidence.1.footer':   '经过检索、打包进上下文并标注引用——不是凭记忆转述。',
    'evidence.2.question': '对于面向全局读取、可以容忍最终一致性的配置数据使用 Workers KV，与需要强读写一致性的可变文件使用 R2 对象存储相比，两者的权衡是什么？',
    'evidence.2.excerpt':  '将 KV 用于全局读取、可以容忍<mark class="highlight">最终一致性</mark>的配置数据。将 R2 用于需要强读写一致性的可变文件或对象内容。',
    'evidence.2.footer':   '两份源文档，一次有据可依的对比。',

    // Visual lab
    'studio.title':        '用于学习 RAG 的丰富本地工作区',
    'studio.subtitle':     '通过真实文档学习：从引导回放开始，实时检查检索计算，再进行自己的实验。',
    'studio.replay.title': '从回放开始',
    'studio.replay.desc':  '沿着真实语料的已记录课程，从源文档按可见步骤一路跟随到带引用的答案。',
    'studio.inspect.title':'让检索机制可见',
    'studio.inspect.desc': '通过六个实时模块学习 BM25 计算、余弦相似度、NumPy 与 Qdrant、混合融合、重排序和评估。',
    'studio.compare.title':'通过对比学习',
    'studio.compare.desc': '在 16 道已审核问题上比较两种检索配置，检查每道题的结果，再把重排序带入自由 Explore。',
    'studio.shot.guided': '引导课程：沿着排序后的证据回到它的来源。',
    'studio.shot.explore':'探索：改变检索配置，检查返回了什么。',
    'studio.shot.failure':'失败实验室：并排比较基线运行与修复后的结果。',
    'studio.shot.inspect':'构建与检查：直接查看分块、向量和来源出处。',
    'studio.cta':         '使用 Docker Compose 运行本地实验室 →',
    'studio.local-note':  '核心实验流程在本地运行，无需账号。只有 Live Ask 会联系你自行配置的服务商。',

    // Capabilities
    'cap.label':    '功能覆盖',
    'cap.title':    '四个层面，一条可检查的流水线',
    'cap.subtitle': '所有能力都映射到同一套可见架构和项目自有产物。',
    'cap.indexing.title':   '索引',
    'cap.indexing.points':  '<li>文档加载、文本规范化与元数据</li><li>固定字符、结构化与实验性语义分块</li><li>本地嵌入模型（MiniLM）与可检查的 NumPy 索引</li>',
    'cap.retrieval.title':  '检索',
    'cap.retrieval.points': '<li>实时解释 BM25、余弦相似度与混合 RRF</li><li>让同一组向量通过可检查 NumPy 与可选 Qdrant</li><li>交叉编码器候选到最终证据的重排序</li>',
    'cap.generation.title':  '生成',
    'cap.generation.points': '<li>带 Token 预算的上下文打包，省略的 chunk 可追踪</li><li>基于检索的提示词组装与 OpenAI 兼容生成</li><li>引用来源，证据不足时拒答</li>',
    'cap.eval.title':  '评估与可观测性',
    'cap.eval.points': '<li>在 16 道已审核问题上的浏览器检索 A/B 评估</li><li>基于 LLM 评判的答案指标：忠实度、相关性、正确性</li><li>逐查询 trace 与精心设计的失败实验室</li>',

    // Stack
    'stack.title':    '技术栈',
    'stack.subtitle': '刻意保持精简——每个组件在代码中清晰可见。',
    'stack.l.language':   '语言',
    'stack.l.interface':  '接口',
    'stack.v.interface':  'CLI + 本地可视化实验室',
    'stack.l.embeddings': '嵌入模型',
    'stack.l.index':      '向量索引',
    'stack.v.index':      'NumPy + 可选本地 Qdrant',
    'stack.l.generation': '生成',
    'stack.v.generation': 'OpenAI 兼容 API',
    'stack.l.test':       '测试后端',
    'stack.v.test':       'Fake 嵌入 + Fake 生成器（完全离线）',
    'stack.l.corpus':     '语料库',
    'stack.v.corpus':     'Cloudflare 文档 + watsonxDocsQA',
    'stack.l.deps':       '依赖管理',
    'stack.note': '不依赖 LangChain、LlamaIndex 或 Haystack 封装。核心 RAG 机制直接实现。',

    // CLI
    'cli.title':    '简洁的 CLI 接口',
    'cli.subtitle': '可视化工作区直接、可重复的伙伴：以同一套可检查机制索引语料、检索证据并发起问答。',
    'cli.link':     '完整 CLI 参考文档请访问 GitHub →',

    // CTA
    'cta.title': '对源码感兴趣？',
    'cta.desc':  '克隆它、运行它，或者直接看看它是如何构建的——一切都在 GitHub 上。',
    'cta.btn':   '在 GitHub 查看',

    // Footer
    'footer.built': '作者',
  }
};

const savedLang = localStorage.getItem('tiny-rag-lang');
let currentLang = savedLang === 'zh' ? 'zh' : 'en';

function applyTranslations(lang) {
  // Plain text elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (translations[lang][key] !== undefined) {
      el.textContent = translations[lang][key];
    }
  });

  // HTML content elements (contain tags like <strong>, <code>, <mark>)
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const key = el.dataset.i18nHtml;
    if (translations[lang][key] !== undefined) {
      el.innerHTML = translations[lang][key];
    }
  });

  // Page language attribute and title
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.title = lang === 'zh'
    ? 'tiny-rag-lab — 可检查的经典 RAG 学习实验室'
    : 'tiny-rag-lab — An Inspectable Classic RAG Learning Lab';

  // Toggle button label
  const btn = document.getElementById('lang-toggle');
  if (btn) {
    btn.textContent = lang === 'zh' ? 'English' : '中文';
    btn.setAttribute(
      'aria-label',
      lang === 'zh' ? '切换到英文' : 'Switch to Chinese',
    );
  }
}

function setLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('tiny-rag-lang', lang);
  applyTranslations(lang);
}

document.getElementById('lang-toggle').addEventListener('click', () => {
  setLanguage(currentLang === 'en' ? 'zh' : 'en');
});

// Apply on initial load
applyTranslations(currentLang);
