import { ChangeEvent, Fragment, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Lang = "en" | "zh";
type Corpus = { id: string; name: string; kind: string; file_count: number };
type IndexItem = { id: string; manifest: Record<string, unknown> };
type LabRun = { trace: Record<string, any>; evidence: any[]; index: any; run_id: string };
type FailureEvidence = { rank: number; score: number; doc_id: string; text: string };
type FailureSide = { config: Record<string, string | number>; trace: { evidence: FailureEvidence[]; context_pack: { selected: string[]; omitted: string[] }; answer: string; citations: string[]; outcome_label: string } };
type FailureLesson = { id: string; label: string; question: string; explanation: { en: string; zh: string }; baseline: FailureSide; intervention: FailureSide };

const copy = {
  en: {
    title: "tiny-rag-lab", subtitle: "A local visual laboratory for classic RAG.",
    start: "Start Lab", corpus: "Corpus Library", explore: "Index Explorer", run: "Run Workspace", failure: "Failure Lab", settings: "Settings",
    upload: "Add Markdown or text files", build: "Build index", question: "Ask a question", retrieve: "Retrieve", ask: "Live Ask",
    noIndex: "Upload a small corpus and build an index to start exploring.",
    stages: ["Corpus", "Chunks", "Embeddings", "Retrieve", "Context", "Answer"],
    evidence: "Retrieved evidence", prompt: "Grounded prompt", answer: "Answer", vectors: "Vector components", manifest: "Index manifest",
    provider: "Live Ask needs an OpenAI-compatible provider configured for this local lab.",
    quick: "The starter lessons replay saved traces offline. Add watsonxDocsQA or your own small corpus for a real run.", replay: "Replay starter trace", watson: "Download watsonxDocsQA",
    failureText: "Failure lessons compare a baseline with an intervention using the project’s curated cases.",
    providerUrl: "Provider base URL", providerModel: "Model", providerKey: "API key (this session only)", backend: "Vector index backend", modelReady: "Default embedding model is ready", modelMissing: "Default embedding model must be downloaded before custom indexing", downloadModel: "Download default embedding model", retriever: "Retriever", topK: "Top-k", contextBudget: "Context budget (0 = unlimited)", rerankerDeferred: "Reranker controls are not yet available in the visual lab; use the CLI for this advanced comparison.", selected: "Selected for context", omitted: "Omitted from context", vector: "Query vector", timings: "Stage timings", baseline: "Baseline", intervention: "Intervention", learn: "Read the learning material",
  },
  zh: {
    title: "tiny-rag-lab", subtitle: "用于理解经典 RAG 的本地可视化实验室。",
    start: "开始实验", corpus: "语料库", explore: "索引浏览", run: "运行工作区", failure: "失败实验室", settings: "设置",
    upload: "添加 Markdown 或文本文件", build: "构建索引", question: "输入问题", retrieve: "检索", ask: "实时问答",
    noIndex: "上传一个小型语料库并构建索引，即可开始探索。",
    stages: ["语料", "分块", "嵌入", "检索", "上下文", "答案"],
    evidence: "检索到的证据", prompt: "基于证据的提示词", answer: "答案", vectors: "向量分量", manifest: "索引清单",
    provider: "实时问答需要为本地实验室配置 OpenAI 兼容的模型服务。",
    quick: "入门课程可离线回放已保存的 trace。请添加 watsonxDocsQA 或自己的小型语料库来运行真实流程。", replay: "回放入门 trace", watson: "下载 watsonxDocsQA",
    failureText: "失败课程使用项目中精心设计的案例，对比基线与干预措施。",
    providerUrl: "模型服务地址", providerModel: "模型", providerKey: "API 密钥（仅当前会话）", backend: "向量索引后端", modelReady: "默认嵌入模型已就绪", modelMissing: "索引自定义语料库前需要下载默认嵌入模型", downloadModel: "下载默认嵌入模型", retriever: "检索器", topK: "Top-k", contextBudget: "上下文预算（0 = 不限）", rerankerDeferred: "可视化实验室暂不提供重排序控制；请使用 CLI 进行此高级对比。", selected: "已选入上下文", omitted: "未选入上下文", vector: "查询向量", timings: "阶段耗时", baseline: "基线", intervention: "干预", learn: "阅读学习材料",
  },
} as const;

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
  return response.json();
}

function App() {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem("tiny-rag-lab-lang") as Lang) || "en");
  const [tab, setTab] = useState("start");
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [indexes, setIndexes] = useState<IndexItem[]>([]);
  const [corpusId, setCorpusId] = useState("");
  const [indexId, setIndexId] = useState("");
  const [question, setQuestion] = useState("");
  const [retriever, setRetriever] = useState<"dense" | "bm25" | "hybrid">("dense");
  const [topK, setTopK] = useState(5);
  const [contextBudget, setContextBudget] = useState(0);
  const [backend, setBackend] = useState<"numpy" | "qdrant">("numpy");
  const [providerUrl, setProviderUrl] = useState(() => localStorage.getItem("tiny-rag-lab-provider-url") || "");
  const [providerModel, setProviderModel] = useState(() => localStorage.getItem("tiny-rag-lab-provider-model") || "");
  const [providerKey, setProviderKey] = useState("");
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [lessons, setLessons] = useState<FailureLesson[]>([]);
  const [lessonId, setLessonId] = useState("");
  const [activeStage, setActiveStage] = useState(3);
  const [run, setRun] = useState<LabRun | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [message, setMessage] = useState("");
  const t = copy[lang];

  const refresh = async () => {
    const [c, i] = await Promise.all([api<{ items: Corpus[] }>("/corpora"), api<{ items: IndexItem[] }>("/indexes")]);
    setCorpora(c.items); setIndexes(i.items);
  };
  useEffect(() => { void refresh().catch((e) => setMessage(e.message)); }, []);
  useEffect(() => { localStorage.setItem("tiny-rag-lab-lang", lang); document.documentElement.lang = lang === "zh" ? "zh-CN" : "en"; }, [lang]);
  useEffect(() => { localStorage.setItem("tiny-rag-lab-provider-url", providerUrl); localStorage.setItem("tiny-rag-lab-provider-model", providerModel); }, [providerUrl, providerModel]);
  useEffect(() => { if (indexId) void api(`/indexes/${indexId}`).then(setDetail).catch((e) => setMessage(e.message)); }, [indexId]);
  useEffect(() => { void api<{ ready: boolean }>("/models/default/status").then((status) => setModelReady(status.ready)).catch((e) => setMessage(e.message)); }, []);
  useEffect(() => { void api<{ items: FailureLesson[] }>("/failure-lessons").then((data) => { setLessons(data.items); setLessonId(data.items[0]?.id || ""); }).catch((e) => setMessage(e.message)); }, []);

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files; if (!files?.length) return;
    const form = new FormData(); Array.from(files as FileList).forEach((file: File) => form.append("files", file)); form.append("name", "My corpus");
    try { const result = await api<Corpus>("/corpora/upload", { method: "POST", body: form }); setCorpusId(result.id); setMessage("Corpus added."); await refresh(); } catch (e: any) { setMessage(e.message); }
  };
  const build = async () => {
    try {
      const job = await api<{ id: string }>("/indexes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corpus_id: corpusId, index_backend: backend }) });
      setMessage("Indexing locally…");
      const timer = window.setInterval(async () => { const state = await api<any>(`/jobs/${job.id}`); if (state.status === "complete") { window.clearInterval(timer); setIndexId(state.index_id); setMessage("Index ready."); await refresh(); } if (state.status === "failed") { window.clearInterval(timer); setMessage(state.error); } }, 700);
    } catch (e: any) { setMessage(e.message); }
  };
  const execute = async (kind: "retrieve" | "ask") => {
    try { setRun(await api<LabRun>(`/runs/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ index_id: indexId, query: question, retriever, top_k: topK, context_budget: contextBudget, provider: kind === "ask" ? { base_url: providerUrl || undefined, model: providerModel || undefined, api_key: providerKey || undefined } : undefined }) })); setTab("run"); } catch (e: any) { setMessage(e.message); }
  };
  const importWatson = async () => {
    try { const job = await api<{ id: string; status: string }>("/corpora/watsonxdocsqa/import", { method: "POST" }); setMessage(`watsonxDocsQA: ${job.status}`); } catch (e: any) { setMessage(e.message); }
  };
  const downloadModel = async () => {
    try { const job = await api<{ status: string }>("/models/default/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }); setMessage(`${t.downloadModel}: ${job.status}`); } catch (e: any) { setMessage(e.message); }
  };
  const navigation = useMemo<Array<[string, string]>>(() => [["start", t.start], ["corpus", t.corpus], ["explore", t.explore], ["run", t.run], ["failure", t.failure], ["settings", t.settings]], [t]);

  return <main className="app-shell">
    <header className="app-header"><div className="brand"><h1>{t.title}</h1><p>{t.subtitle}</p></div><button className="language-toggle" onClick={() => setLang(lang === "en" ? "zh" : "en")}>{lang === "en" ? "中文" : "English"}</button></header>
    <nav>{navigation.map(([id, label]) => <button className={tab === id ? "active" : ""} key={id} onClick={() => setTab(id)}>{label}</button>)}</nav>
    {message && <p className="notice">{message}</p>}
    {tab === "start" && <section className="studio-panel start-panel"><span className="eyebrow">01 / {t.start}</span><h2>{t.start}</h2><p className="lead">{t.quick}</p><div className="action-row"><button className="primary-action" onClick={() => api<LabRun>("/starter-run").then((value) => { setRun(value); setTab("run"); }).catch((e) => setMessage(e.message))}>{t.replay}</button></div><Pipeline labels={t.stages} /></section>}
    {tab === "corpus" && <section className="studio-panel"><span className="eyebrow">02 / {t.corpus}</span><h2>{t.corpus}</h2><div className="action-row"><label className="upload">{t.upload}<input type="file" multiple accept=".md,.txt,text/plain,text/markdown" onChange={upload} /></label><button onClick={importWatson}>{t.watson}</button></div><div className="control-row"><select value={corpusId} onChange={(e) => setCorpusId(e.target.value)}><option value="">—</option>{corpora.map((corpus) => <option key={corpus.id} value={corpus.id}>{corpus.name} ({corpus.file_count})</option>)}</select><label>{t.backend}<select value={backend} onChange={(e) => setBackend(e.target.value as "numpy" | "qdrant")}><option value="numpy">NumPy</option><option value="qdrant">Qdrant (optional)</option></select></label><button className="primary-action" disabled={!corpusId || !modelReady} onClick={build}>{t.build}</button></div></section>}
    {tab === "explore" && <section className="studio-panel"><span className="eyebrow">03 / {t.explore}</span><h2>{t.explore}</h2><select value={indexId} onChange={(e) => setIndexId(e.target.value)}><option value="">—</option>{indexes.map((index) => <option key={index.id} value={index.id}>{index.id}</option>)}</select>{detail && <><h3>{t.manifest}</h3><pre>{JSON.stringify(detail.manifest, null, 2)}</pre><h3>{t.vectors}</h3>{detail.chunks.slice(0, 8).map((chunk: any) => <article key={chunk.chunk_id}><strong>{chunk.doc_id}</strong><p>{chunk.text}</p><code>[{chunk.vector.slice(0, 8).map((v: number) => v.toFixed(3)).join(", ")}…]</code></article>)}</>}</section>}
    {tab === "run" && <section className="studio-panel run-panel"><span className="eyebrow">04 / {t.run}</span><h2>{t.run}</h2><Pipeline labels={t.stages} activeStage={activeStage} onSelect={setActiveStage} /><div className="control-row"><label>{t.retriever}<select value={retriever} onChange={(e) => setRetriever(e.target.value as "dense" | "bm25" | "hybrid")}><option value="dense">Dense</option><option value="bm25">BM25</option><option value="hybrid">Hybrid (RRF)</option></select></label><label>{t.topK}<input type="number" min="1" max="50" value={topK} onChange={(e) => setTopK(Number(e.target.value))} /></label><label>{t.contextBudget}<input type="number" min="0" value={contextBudget} onChange={(e) => setContextBudget(Number(e.target.value))} /></label></div><label className="question-field"><span>{t.question}</span><input aria-label={t.question} placeholder={t.question} value={question} onChange={(e) => setQuestion(e.target.value)} /></label><div className="action-row run-actions"><button disabled={!indexId || !question} onClick={() => execute("retrieve")}>{t.retrieve}</button><button className="primary-action" disabled={!indexId || !question} onClick={() => execute("ask")}>{t.ask}</button></div><p className="hint">{t.provider}</p><p className="hint">{t.rerankerDeferred}</p>{run && <RunView run={run} t={t} activeStage={activeStage} />}{!indexId && <p>{t.noIndex}</p>}</section>}
    {tab === "failure" && <section className="studio-panel"><span className="eyebrow">05 / {t.failure}</span><h2>{t.failure}</h2><p className="lead">{t.failureText}</p><select value={lessonId} onChange={(e) => setLessonId(e.target.value)}>{lessons.map((lesson) => <option key={lesson.id} value={lesson.id}>{lesson.label}</option>)}</select>{lessons.filter((lesson) => lesson.id === lessonId).map((lesson) => <article key={lesson.id}><h3>{lesson.label}</h3><p>{lesson.question}</p><p>{lesson.explanation[lang]}</p><div className="comparison"><FailureSideView title={t.baseline} side={lesson.baseline} t={t} /><FailureSideView title={t.intervention} side={lesson.intervention} t={t} /></div></article>)}<a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${lang}/rag-failure-lab.md`} target="_blank">{t.learn}</a></section>}
    {tab === "settings" && <section className="studio-panel"><span className="eyebrow">06 / {t.settings}</span><h2>{t.settings}</h2><p className="hint">{modelReady ? t.modelReady : t.modelMissing}</p>{!modelReady && <button className="primary-action" onClick={downloadModel}>{t.downloadModel}</button>}<div className="settings-fields"><label>{t.providerUrl}<input value={providerUrl} onChange={(e) => setProviderUrl(e.target.value)} placeholder="http://127.0.0.1:11434/v1" /></label><label>{t.providerModel}<input value={providerModel} onChange={(e) => setProviderModel(e.target.value)} placeholder="model-name" /></label><label>{t.providerKey}<input type="password" value={providerKey} onChange={(e) => setProviderKey(e.target.value)} /></label></div><p className="hint">{t.provider}</p></section>}
    <footer className="app-footer"><div><strong>tiny-rag-lab</strong><span>{lang === "en" ? "Created by James Wei · Learn classic RAG, locally." : "由 James Wei 创建 · 在本地学习经典 RAG。"}</span></div><div className="footer-links"><a href="https://github.com/jameswei/tiny-rag-lab" target="_blank" rel="noreferrer">GitHub</a><a href="https://jameswei.github.io/tiny-rag-lab/" target="_blank" rel="noreferrer">{lang === "en" ? "Project site" : "项目主页"}</a></div></footer>
  </main>;
}

function Pipeline({ labels, activeStage, onSelect }: { labels: readonly string[]; activeStage?: number; onSelect?: (index: number) => void }) {
  return <div className="pipeline">{labels.map((label, index) => <Fragment key={label}><button className={`pipeline-stage ${activeStage === index ? "stage-active" : ""}`} disabled={!onSelect} onClick={() => onSelect?.(index)}><small>{String(index + 1).padStart(2, "0")}</small>{label}</button>{index < labels.length - 1 && <span className="pipeline-arrow" aria-hidden="true">→</span>}</Fragment>)}</div>;
}
function RunView({ run, t, activeStage }: { run: LabRun; t: any; activeStage: number }) {
  const trace = run.trace;
  return <div className="run">
    {activeStage === 0 && <><pre>{JSON.stringify(run.index.manifest, null, 2)}</pre><a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${document.documentElement.lang === "zh-CN" ? "zh" : "en"}/the-indexing-plane.md`} target="_blank">{t.learn}</a></>}
    {activeStage === 1 && <><h3>{t.evidence}</h3>{run.evidence.map((item) => <article key={item.chunk_id}><strong>{item.doc_id}</strong><p>{item.text}</p></article>)}<a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${document.documentElement.lang === "zh-CN" ? "zh" : "en"}/the-indexing-plane.md`} target="_blank">{t.learn}</a></>}
    {activeStage === 2 && <><h3>{t.vector}</h3><code>{JSON.stringify((run as any).query_vector || [])}</code><a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${document.documentElement.lang === "zh-CN" ? "zh" : "en"}/retrieval-mechanics.md`} target="_blank">{t.learn}</a></>}
    {activeStage === 3 && <><h3>{t.evidence}</h3>{run.evidence.map((item) => <article key={item.chunk_id}><strong>#{item.rank} · {item.score.toFixed(4)} · {item.selected_for_context === false ? t.omitted : t.selected}</strong><p>{item.text}</p><code>{JSON.stringify(item.score_components)}</code></article>)}<pre>{JSON.stringify(trace.latency_by_stage, null, 2)}</pre><a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${document.documentElement.lang === "zh-CN" ? "zh" : "en"}/retrieval-mechanics.md`} target="_blank">{t.learn}</a></>}
    {activeStage === 4 && <>{trace.context_pack && <pre>{JSON.stringify(trace.context_pack, null, 2)}</pre>}{"prompt" in trace && <pre>{trace.prompt}</pre>}<a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${document.documentElement.lang === "zh-CN" ? "zh" : "en"}/context-budget-and-structured-answers.md`} target="_blank">{t.learn}</a></>}
    {activeStage === 5 && <>{"answer" in trace && <><p>{trace.answer}</p><p>{(trace.citations || []).join(", ")}</p></>}<a href={`https://github.com/jameswei/tiny-rag-lab/blob/main/learning_materials/${document.documentElement.lang === "zh-CN" ? "zh" : "en"}/retrieval-and-generation.md`} target="_blank">{t.learn}</a></>}
  </div>;
}

function FailureSideView({ title, side, t }: { title: string; side: FailureSide; t: any }) {
  return <div className="failure-side"><h4>{title}</h4><code>{JSON.stringify(side.config)}</code><p><strong>{side.trace.outcome_label}</strong></p>{side.trace.evidence.map((evidence) => <article key={`${title}-${evidence.doc_id}`}><strong>#{evidence.rank} · {evidence.score.toFixed(4)} · {evidence.doc_id}</strong><p>{evidence.text}</p></article>)}<p>{t.selected}: {side.trace.context_pack.selected.join(", ") || "—"}</p><p>{t.omitted}: {side.trace.context_pack.omitted.join(", ") || "—"}</p><p>{side.trace.answer}</p><p>{side.trace.citations.join(", ")}</p></div>;
}

createRoot(document.getElementById("root")!).render(<App />);
