const translations = {
  en: {
    // Nav
    'nav.about':        'About',
    'nav.pipeline':     'Pipeline',
    'nav.capabilities': 'Capabilities',
    'nav.stack':        'Stack',
    'nav.cli':          'CLI',
    'nav.github':       'GitHub →',

    // Hero
    'hero.tagline':   'A learning-first RAG laboratory — building retrieval-augmented generation from scratch to make the full pipeline visible and inspectable.',
    'hero.cta.github': 'View on GitHub',
    'hero.cta.learn':  'Learn more ↓',

    // About
    'about.title': 'About this project',
    'about.p1':    'RAG (retrieval-augmented generation) is a foundational pattern behind many AI tools and assistants — but most tutorials hide the interesting parts behind framework calls. <strong>tiny-rag-lab</strong> is my attempt to build every stage by hand: document loading, chunking, embedding, local vector search, retrieval, prompt assembly, answer generation, evaluation, and failure diagnosis.',
    'about.p2':    'The goal is not a production platform. The goal is deep, inspectable understanding — readable code over framework magic, evaluation before optimization, failure analysis before advanced features.',
    'about.p3':    'Built across nine incremental phases, the project now covers classic dense retrieval, BM25 keyword search, hybrid fusion, reranking, answer quality judging, context budgeting, and structural/semantic chunking — all wired through a clean Python CLI with trace output at every stage.',
    'about.cta':   'Explore the source on GitHub →',

    // Stats
    'stat.planes':    'RAG Architecture Planes',
    'stat.commands':  'CLI Commands',
    'stat.retrievers':'Retrieval Strategies',
    'stat.chunking':  'Chunking Strategies',

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

    // Capabilities
    'cap.title':    'What it covers',
    'cap.subtitle': 'Built phase-by-phase, each capability measurable and debuggable.',
    'cap.rag.title':       'Classic RAG baseline',
    'cap.rag.desc':        'Full pipeline: document loading, text normalization, chunking, embedding, local cosine-similarity retrieval, grounded prompt assembly, answer generation, and citations.',
    'cap.retrieval.title': 'Retrieval mechanics',
    'cap.retrieval.desc':  'Dense vector search, BM25 keyword retrieval, and hybrid fusion via Reciprocal Rank Fusion — all configurable and comparable with a single CLI flag.',
    'cap.eval.title': 'Evaluation harness',
    'cap.eval.desc':  '<code>rag eval</code> measures retrieval quality (hit rate, MRR, context precision/recall) and answer quality (faithfulness, relevance, correctness) against a prepared QA set.',
    'cap.obs.title': 'Observability & failure lab',
    'cap.obs.desc':  'Per-query trace output exposes retriever, scores, latency, and prompt context. <code>rag diagnose</code> runs curated failure cases with baseline vs. intervention comparison.',
    'cap.rerank.title': 'Reranking & answer judging',
    'cap.rerank.desc':  'Optional second-pass reranking (fake or cross-encoder). LLM-as-judge interface for answer-quality verdicts — both fakeable for offline testing.',
    'cap.budget.title': 'Context budget & chunking',
    'cap.budget.desc':  'Token-budget context packing with traceable omitted chunks. Three chunking strategies: fixed-character, Markdown-structural, and experimental semantic topic-shift.',

    // Stack
    'stack.title':    'Tech stack',
    'stack.subtitle': 'Kept deliberately minimal — every moving part is visible in the code.',
    'stack.l.language':   'Language',
    'stack.l.interface':  'Interface',
    'stack.l.embeddings': 'Embeddings',
    'stack.l.index':      'Vector index',
    'stack.v.index':      'Local NumPy (no vector DB)',
    'stack.l.generation': 'Generation',
    'stack.v.generation': 'OpenAI-compatible API',
    'stack.l.test':       'Test backends',
    'stack.v.test':       'Fake embedder + fake generator (fully offline)',
    'stack.l.corpus':     'Corpus',
    'stack.l.deps':       'Dependencies',
    'stack.note': 'No LangChain, LlamaIndex, or Haystack wrapper. The core RAG mechanics are implemented directly.',

    // CLI
    'cli.title':    'A clean CLI interface',
    'cli.subtitle': 'Index a corpus, retrieve evidence, ask a question — each step inspectable.',
    'cli.link':     'Full CLI reference and docs on GitHub →',

    // CTA
    'cta.title': 'Interested in the source?',
    'cta.desc':  'All code, phase specs, and implementation notes are on GitHub.',
    'cta.btn':   'View on GitHub',

    // Footer
    'footer.built': 'Built by',
    'footer.repo':  'Repository',
  },

  zh: {
    // Nav
    'nav.about':        '关于',
    'nav.pipeline':     '流程',
    'nav.capabilities': '功能',
    'nav.stack':        '技术栈',
    'nav.cli':          'CLI',
    'nav.github':       'GitHub →',

    // Hero
    'hero.tagline':   '以学习为先的 RAG 实验室——从零手工实现检索增强生成的完整流程，让每个阶段清晰可见、可调试。',
    'hero.cta.github': '在 GitHub 查看',
    'hero.cta.learn':  '了解更多 ↓',

    // About
    'about.title': '关于此项目',
    'about.p1':    'RAG（检索增强生成）是当前众多 AI 工具和助手背后的核心模式——但大多数教程将关键细节隐藏在框架调用之后。<strong>tiny-rag-lab</strong> 是我手工实现每个阶段的尝试：文档加载、分块、嵌入、本地向量检索、召回、提示词组装、答案生成、评估与失败诊断。',
    'about.p2':    '目标不是构建生产级平台，而是深入、可观察的理解——可读代码优于框架魔法，先评估再优化，先分析失败再引入高级特性。',
    'about.p3':    '项目按九个递进阶段构建，目前已覆盖经典稠密检索、BM25 关键词搜索、混合融合、重排序、答案质量评判、上下文预算以及结构化/语义分块——全部通过简洁的 Python CLI 串联，每个阶段均有 trace 输出。',
    'about.cta':   '在 GitHub 查看源码 →',

    // Stats
    'stat.planes':    'RAG 架构层',
    'stat.commands':  'CLI 命令',
    'stat.retrievers':'检索策略',
    'stat.chunking':  '分块策略',

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

    // Capabilities
    'cap.title':    '功能覆盖',
    'cap.subtitle': '按阶段迭代构建，每项能力均可测量和调试。',
    'cap.rag.title':       '经典 RAG 基线',
    'cap.rag.desc':        '完整流程：文档加载、文本规范化、分块、嵌入、本地余弦相似度检索、基于检索的提示词组装、答案生成与引用。',
    'cap.retrieval.title': '检索机制',
    'cap.retrieval.desc':  '稠密向量检索、BM25 关键词检索，以及基于倒数排名融合（RRF）的混合检索——通过单个 CLI 参数即可配置和对比。',
    'cap.eval.title': '评估框架',
    'cap.eval.desc':  '<code>rag eval</code> 基于预置 QA 集衡量检索质量（命中率、MRR、上下文精确率/召回率）和答案质量（忠实度、相关性、正确性）。',
    'cap.obs.title': '可观测性与失败实验室',
    'cap.obs.desc':  '逐查询 trace 输出暴露检索器、分数、延迟和提示词上下文。<code>rag diagnose</code> 运行精心设计的失败案例，进行基线 vs. 干预对比。',
    'cap.rerank.title': '重排序与答案评判',
    'cap.rerank.desc':  '可选的二次重排序（fake 或 cross-encoder）。基于 LLM 的答案质量评判接口，生成忠实度、相关性等指标——均可用 fake 后端离线测试。',
    'cap.budget.title': '上下文预算与分块策略',
    'cap.budget.desc':  'Token 预算的上下文打包，省略的 chunk 可追踪。三种分块策略：固定字符、Markdown 结构感知，以及实验性的语义话题切换。',

    // Stack
    'stack.title':    '技术栈',
    'stack.subtitle': '刻意保持精简——每个组件在代码中清晰可见。',
    'stack.l.language':   '语言',
    'stack.l.interface':  '接口',
    'stack.l.embeddings': '嵌入模型',
    'stack.l.index':      '向量索引',
    'stack.v.index':      '本地 NumPy（不用向量数据库）',
    'stack.l.generation': '生成',
    'stack.v.generation': 'OpenAI 兼容 API',
    'stack.l.test':       '测试后端',
    'stack.v.test':       'Fake 嵌入 + Fake 生成器（完全离线）',
    'stack.l.corpus':     '语料库',
    'stack.l.deps':       '依赖管理',
    'stack.note': '不依赖 LangChain、LlamaIndex 或 Haystack 封装。核心 RAG 机制直接实现。',

    // CLI
    'cli.title':    '简洁的 CLI 接口',
    'cli.subtitle': '索引语料、检索证据、发起问答——每一步均可审查。',
    'cli.link':     '完整 CLI 参考文档请访问 GitHub →',

    // CTA
    'cta.title': '对源码感兴趣？',
    'cta.desc':  '所有代码、阶段规格说明和实现笔记均在 GitHub 上。',
    'cta.btn':   '在 GitHub 查看',

    // Footer
    'footer.built': '作者',
    'footer.repo':  '代码仓库',
  }
};

let currentLang = localStorage.getItem('tiny-rag-lang') || 'en';

function applyTranslations(lang) {
  // Plain text elements
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (translations[lang][key] !== undefined) {
      el.textContent = translations[lang][key];
    }
  });

  // HTML content elements (contain tags like <strong>, <code>)
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const key = el.dataset.i18nHtml;
    if (translations[lang][key] !== undefined) {
      el.innerHTML = translations[lang][key];
    }
  });

  // Page language attribute and title
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
  document.title = lang === 'zh'
    ? 'tiny-rag-lab — 以学习为先的 RAG 实验室'
    : 'tiny-rag-lab — A Learning-First RAG Laboratory';

  // Toggle button label
  const btn = document.getElementById('lang-toggle');
  if (btn) btn.textContent = lang === 'zh' ? 'English' : '中文';
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
