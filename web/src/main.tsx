import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { copy, type Copy } from "./copy";
import { BuildInspectView } from "./components/BuildInspectView";
import { ExploreView } from "./components/ExploreView";
import { FailureLabView } from "./components/FailureLabView";
import { HomeView } from "./components/HomeView";
import { LearnView } from "./components/LearnView";
import { SettingsView } from "./components/SettingsView";
import type { Area, BackendAvailability, BuildView, CatalogQuestion, Corpus, FailureLesson, GuidedLesson, GuidedLessonSummary, IndexItem, LabRun, Lang, Stage } from "./types";
import "./styles.css";

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || body.error || response.statusText); }
  return response.json();
}

function useReducedMotion() {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(() => window.matchMedia?.(query).matches ?? false);
  useEffect(() => {
    const media = window.matchMedia?.(query); if (!media) return;
    const update = () => setReduced(media.matches); update(); media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);
  return reduced;
}

export function App() {
  const [lang, setLang] = useState<Lang>(() => (localStorage.getItem("tiny-rag-lab-lang") as Lang) || "en");
  const [area, setArea] = useState<Area>("home");
  const [buildView, setBuildView] = useState<BuildView>("build");
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [indexes, setIndexes] = useState<IndexItem[]>([]);
  const [corpusId, setCorpusId] = useState("");
  const [indexId, setIndexId] = useState("");
  const [question, setQuestion] = useState("");
  const [catalogQuestions, setCatalogQuestions] = useState<CatalogQuestion[]>([]);
  const [catalogQuestionId, setCatalogQuestionId] = useState("");
  const [retriever, setRetriever] = useState<"dense" | "bm25" | "hybrid">("dense");
  const [topK, setTopK] = useState(5);
  const [contextBudget, setContextBudget] = useState(0);
  const [backend, setBackend] = useState<"numpy" | "qdrant">("numpy");
  const [qdrantAvailable, setQdrantAvailable] = useState(false);
  const [building, setBuilding] = useState(false);
  const [providerUrl, setProviderUrl] = useState(() => localStorage.getItem("tiny-rag-lab-provider-url") || "");
  const [providerModel, setProviderModel] = useState(() => localStorage.getItem("tiny-rag-lab-provider-model") || "");
  const [providerKey, setProviderKey] = useState("");
  const [environmentProviderReady, setEnvironmentProviderReady] = useState(false);
  const [providerVerified, setProviderVerified] = useState(false);
  const [running, setRunning] = useState<"retrieve" | "ask" | null>(null);
  const [testingProvider, setTestingProvider] = useState(false);
  const [modelReady, setModelReady] = useState<boolean | null>(null);
  const [downloadingModel, setDownloadingModel] = useState(false);
  const [lessons, setLessons] = useState<FailureLesson[]>([]);
  const [guidedLessons, setGuidedLessons] = useState<GuidedLessonSummary[]>([]);
  const [guidedLesson, setGuidedLesson] = useState<GuidedLesson | null>(null);
  const [lessonProgress, setLessonProgress] = useState<Record<string, Stage>>({});
  const [lessonId, setLessonId] = useState("");
  const [activeStage, setActiveStage] = useState<Stage>(3);
  const [run, setRun] = useState<LabRun | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [message, setMessage] = useState("");
  const runRequestId = useRef(0);
  const providerTestRequestId = useRef(0);
  const runAbort = useRef<AbortController | null>(null);
  const providerTestAbort = useRef<AbortController | null>(null);
  const reducedMotion = useReducedMotion();
  const t = copy[lang] as Copy;

  const refresh = async () => {
    const [nextCorpora, nextIndexes] = await Promise.all([api<{ items: Corpus[] }>("/corpora"), api<{ items: IndexItem[] }>("/indexes")]);
    setCorpora(nextCorpora.items); setIndexes(nextIndexes.items);
  };

  useEffect(() => { void refresh().catch((error: Error) => setMessage(error.message)); }, []);
  useEffect(() => { localStorage.setItem("tiny-rag-lab-lang", lang); document.documentElement.lang = lang === "zh" ? "zh-CN" : "en"; }, [lang]);
  useEffect(() => { localStorage.setItem("tiny-rag-lab-provider-url", providerUrl); localStorage.setItem("tiny-rag-lab-provider-model", providerModel); }, [providerUrl, providerModel]);
  useEffect(() => { setMessage(""); }, [area]);
  useEffect(() => { if (indexId) void api(`/indexes/${indexId}`).then(setDetail).catch((error: Error) => setMessage(error.message)); }, [indexId]);
  useEffect(() => { const corpusId = indexes.find((item) => item.id === indexId)?.manifest.source_corpus_id; setCatalogQuestionId(""); if (typeof corpusId === "string") void api<{ items: CatalogQuestion[] }>(`/corpora/${corpusId}/questions`).then((data) => setCatalogQuestions(data.items)).catch(() => setCatalogQuestions([])); else setCatalogQuestions([]); }, [indexId, indexes]);
  useEffect(() => { void api<{ ready: boolean }>("/models/default/status").then((status) => setModelReady(status.ready)).catch((error: Error) => setMessage(error.message)); }, []);
  useEffect(() => { void api<{ items: BackendAvailability[] }>("/backends").then((status) => setQdrantAvailable(status.items.some((item) => item.id === "qdrant" && item.available))).catch(() => setQdrantAvailable(false)); }, []);
  useEffect(() => { void api<{ configured: boolean }>("/provider-status").then((status) => setEnvironmentProviderReady(status.configured)).catch(() => setEnvironmentProviderReady(false)); }, []);
  useEffect(() => { void api<{ items: FailureLesson[] }>("/failure-lessons").then((data) => { setLessons(data.items); setLessonId(data.items[0]?.id || ""); }).catch((error: Error) => setMessage(error.message)); }, []);
  const openLesson = async (id: string, navigate = true) => { try { setGuidedLesson(await api<GuidedLesson>(`/lessons/${id}`)); setActiveStage(0); if (navigate) setArea("learn"); } catch (error: any) { setMessage(error.message); } };
  useEffect(() => { void api<{ items: GuidedLessonSummary[] }>("/lessons").then((data) => { const items = data.items || []; setGuidedLessons(items); if (items[0]) void api<GuidedLesson>(`/lessons/${items[0].id}`).then(setGuidedLesson).catch((error: Error) => setMessage(error.message)); }).catch((error: Error) => setMessage(error.message)); }, []);
  useEffect(() => () => { runAbort.current?.abort(); providerTestAbort.current?.abort(); }, []);

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files; if (!files?.length) return;
    const form = new FormData(); Array.from(files).forEach((file) => form.append("files", file)); form.append("name", t.myCorpus);
    try { const result = await api<Corpus>("/corpora/upload", { method: "POST", body: form }); setCorpusId(result.id); setMessage(t.corpusAdded); await refresh(); } catch (error: any) { setMessage(error.message); }
  };
  const build = async () => {
    if (building || (backend === "qdrant" && !qdrantAvailable)) return;
    setBuilding(true);
    try {
      const job = await api<{ id: string }>("/indexes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corpus_id: corpusId, index_backend: backend }) });
      setMessage(t.indexing);
      const poll = async (): Promise<void> => {
        try {
          const state = await api<any>(`/jobs/${job.id}`);
          if (state.status === "complete") { setIndexId(state.index_id); setBuildView("inspect"); setMessage(t.indexReady); await refresh(); setBuilding(false); return; }
          if (state.status === "failed") { setMessage(state.error || "Indexing failed."); setBuilding(false); return; }
          window.setTimeout(() => { void poll(); }, 700);
        } catch (error: any) { setMessage(error.message); setBuilding(false); }
      };
      void poll();
    } catch (error: any) { setMessage(error.message); setBuilding(false); }
  };
  const execute = async (kind: "retrieve" | "ask") => {
    if (running || (kind === "ask" && (!providerVerified || testingProvider))) return;
    const requestId = ++runRequestId.current;
    const controller = new AbortController();
    runAbort.current = controller;
    setRunning(kind);
    try {
      const nextRun = await api<LabRun>(`/runs/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, body: JSON.stringify({ index_id: indexId, query: question || undefined, catalog_question_id: catalogQuestionId || undefined, retriever, top_k: topK, context_budget: contextBudget, provider: kind === "ask" ? { base_url: providerUrl || undefined, model: providerModel || undefined, api_key: providerKey || undefined } : undefined }) });
      if (controller.signal.aborted || requestId !== runRequestId.current) return;
      setRun(nextRun); setActiveStage(kind === "ask" ? 5 : 3); setArea("explore");
    } catch (error: any) {
      if (controller.signal.aborted || requestId !== runRequestId.current) return;
      setMessage(
        error.message === "Download the default embedding model before dense or hybrid retrieval"
          ? t.modelDownloadRequired
          : error.message,
      );
    } finally { if (requestId === runRequestId.current) { runAbort.current = null; setRunning(null); } }
  };
  const providerReady = environmentProviderReady || Boolean((providerUrl || providerKey) && (providerModel || "gpt-4o-mini"));
  const testProvider = async () => {
    if (!providerReady || testingProvider || running === "ask") return;
    const requestId = ++providerTestRequestId.current;
    const controller = new AbortController();
    providerTestAbort.current = controller;
    setTestingProvider(true);
    try {
      const result = await api<{ message: string }>("/provider/test", { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, body: JSON.stringify({ base_url: providerUrl || undefined, model: providerModel || undefined, api_key: providerKey || undefined }) });
      if (controller.signal.aborted || requestId !== providerTestRequestId.current) return;
      setProviderVerified(true); setMessage(result.message);
    } catch (error: any) {
      if (controller.signal.aborted || requestId !== providerTestRequestId.current) return;
      setProviderVerified(false); setMessage(error.message);
    } finally { if (requestId === providerTestRequestId.current) { providerTestAbort.current = null; setTestingProvider(false); } }
  };
  const replayStarter = () => { if (guidedLessons[0]) void openLesson(guidedLessons[0].id); else setArea("learn"); };
  const downloadModel = async () => {
    if (downloadingModel) return;
    try {
      const job = await api<{ id: string; status: string }>("/models/default/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
      if (job.status === "complete") { setModelReady(true); setMessage(t.modelReady); return; }
      setDownloadingModel(true); setMessage(t.modelDownloading);
      const poll = async (): Promise<void> => {
        try {
          const state = await api<{ status: string; error?: string }>(`/jobs/${job.id}`);
          if (state.status === "complete") { setModelReady(true); setDownloadingModel(false); setMessage(t.modelReady); return; }
          if (state.status === "failed") { setDownloadingModel(false); setMessage(state.error || "Model download failed."); return; }
          window.setTimeout(() => { void poll(); }, 700);
        } catch (error: any) { setDownloadingModel(false); setMessage(error.message); }
      };
      void poll();
    } catch (error: any) { setMessage(error.message); }
  };
  const navigation = useMemo<Array<[Area, string]>>(() => [
    ["home", t.areas.home], ["learn", t.areas.learn], ["explore", t.areas.explore], ["build", t.areas.build], ["failure", t.areas.failure], ["settings", t.areas.settings],
  ], [t]);

  return <main className="app-shell" data-motion={reducedMotion ? "reduced" : "full"}>
    <header className="app-header"><div className="brand"><h1>{t.title}</h1><p>{t.subtitle}</p></div><button className="language-toggle" type="button" onClick={() => setLang(lang === "en" ? "zh" : "en")}>{lang === "en" ? "中文" : "English"}</button></header>
    <nav aria-label="Lab areas">{navigation.map(([id, label]) => <button type="button" className={area === id ? "active" : ""} aria-current={area === id ? "page" : undefined} key={id} onClick={() => setArea(id)}>{label}</button>)}</nav>
    {message && <p className="notice status-panel status-info" role="status">{message}<button type="button" aria-label="Dismiss" onClick={() => setMessage("")}>×</button></p>}
    {area === "home" && <HomeView onReplay={replayStarter} onBuild={() => { setBuildView("build"); setArea("build"); }} t={t} />}
    {area === "learn" && <LearnView lessons={guidedLessons} lesson={guidedLesson} activeStage={activeStage} maxStage={guidedLesson ? lessonProgress[guidedLesson.lesson.id] ?? 0 : 0} onLesson={(id) => void openLesson(id)} onStage={setActiveStage} onAdvance={(stage) => { if (guidedLesson) setLessonProgress((current) => ({ ...current, [guidedLesson.lesson.id]: Math.max(current[guidedLesson.lesson.id] ?? 0, stage) as Stage })); }} lang={lang} t={t} />}
    {area === "build" && <BuildInspectView view={buildView} setView={setBuildView} corpora={corpora} indexes={indexes} corpusId={corpusId} indexId={indexId} backend={backend} qdrantAvailable={qdrantAvailable} modelReady={modelReady} building={building} detail={detail} onCorpus={setCorpusId} onIndex={setIndexId} onBackend={setBackend} onUpload={upload} onBuild={build} t={t} />}
    {area === "explore" && <ExploreView indexes={indexes} indexId={indexId} question={question} catalogQuestions={catalogQuestions} catalogQuestionId={catalogQuestionId} retriever={retriever} topK={topK} contextBudget={contextBudget} run={run} activeStage={activeStage} lang={lang} running={running} testingProvider={testingProvider} providerReady={providerVerified} onIndex={setIndexId} onQuestion={(value) => { setQuestion(value); setCatalogQuestionId(""); }} onCatalogQuestion={(id) => { setCatalogQuestionId(id); const found = catalogQuestions.find((item) => item.id === id); setQuestion(found?.question || ""); }} onRetriever={setRetriever} onTopK={setTopK} onContextBudget={setContextBudget} onStage={setActiveStage} onRun={execute} t={t} />}
    {area === "failure" && <FailureLabView lessons={lessons} lessonId={lessonId} onLesson={setLessonId} lang={lang} t={t} />}
    {area === "settings" && <SettingsView modelReady={modelReady} downloadingModel={downloadingModel} providerUrl={providerUrl} providerModel={providerModel} providerKey={providerKey} providerReady={providerReady} testing={testingProvider} generating={running === "ask"} onDownload={downloadModel} onTest={testProvider} onProviderUrl={(value) => { setProviderVerified(false); setProviderUrl(value); }} onProviderModel={(value) => { setProviderVerified(false); setProviderModel(value); }} onProviderKey={(value) => { setProviderVerified(false); setProviderKey(value); }} t={t} />}
    <footer className="app-footer"><div><strong>tiny-rag-lab</strong><span>{lang === "en" ? "Created by James Wei · Learn classic RAG, locally." : "由 James Wei 创建 · 在本地学习经典 RAG。"}</span></div><div className="footer-links"><a href="https://github.com/jameswei/tiny-rag-lab" target="_blank" rel="noreferrer">GitHub</a><a href="https://jameswei.github.io/tiny-rag-lab/" target="_blank" rel="noreferrer">{lang === "en" ? "Project site" : "项目主页"}</a></div></footer>
  </main>;
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<App />);
