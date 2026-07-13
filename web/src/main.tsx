import { type ChangeEvent, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { copy, type Copy } from "./copy";
import { BuildInspectView } from "./components/BuildInspectView";
import { ExploreView } from "./components/ExploreView";
import { FailureLabView } from "./components/FailureLabView";
import { HomeView } from "./components/HomeView";
import { SettingsView } from "./components/SettingsView";
import type { Area, BuildView, Corpus, FailureLesson, IndexItem, LabRun, Lang, Stage } from "./types";
import "./styles.css";

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || response.statusText);
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
  const [activeStage, setActiveStage] = useState<Stage>(3);
  const [run, setRun] = useState<LabRun | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [message, setMessage] = useState("");
  const reducedMotion = useReducedMotion();
  const t = copy[lang] as Copy;

  const refresh = async () => {
    const [nextCorpora, nextIndexes] = await Promise.all([api<{ items: Corpus[] }>("/corpora"), api<{ items: IndexItem[] }>("/indexes")]);
    setCorpora(nextCorpora.items); setIndexes(nextIndexes.items);
  };

  useEffect(() => { void refresh().catch((error: Error) => setMessage(error.message)); }, []);
  useEffect(() => { localStorage.setItem("tiny-rag-lab-lang", lang); document.documentElement.lang = lang === "zh" ? "zh-CN" : "en"; }, [lang]);
  useEffect(() => { localStorage.setItem("tiny-rag-lab-provider-url", providerUrl); localStorage.setItem("tiny-rag-lab-provider-model", providerModel); }, [providerUrl, providerModel]);
  useEffect(() => { if (indexId) void api(`/indexes/${indexId}`).then(setDetail).catch((error: Error) => setMessage(error.message)); }, [indexId]);
  useEffect(() => { void api<{ ready: boolean }>("/models/default/status").then((status) => setModelReady(status.ready)).catch((error: Error) => setMessage(error.message)); }, []);
  useEffect(() => { void api<{ items: FailureLesson[] }>("/failure-lessons").then((data) => { setLessons(data.items); setLessonId(data.items[0]?.id || ""); }).catch((error: Error) => setMessage(error.message)); }, []);

  const upload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files; if (!files?.length) return;
    const form = new FormData(); Array.from(files).forEach((file) => form.append("files", file)); form.append("name", t.myCorpus);
    try { const result = await api<Corpus>("/corpora/upload", { method: "POST", body: form }); setCorpusId(result.id); setMessage(t.corpusAdded); await refresh(); } catch (error: any) { setMessage(error.message); }
  };
  const build = async () => {
    try {
      const job = await api<{ id: string }>("/indexes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corpus_id: corpusId, index_backend: backend }) });
      setMessage(t.indexing);
      const timer = window.setInterval(async () => {
        const state = await api<any>(`/jobs/${job.id}`);
        if (state.status === "complete") { window.clearInterval(timer); setIndexId(state.index_id); setBuildView("inspect"); setMessage(t.indexReady); await refresh(); }
        if (state.status === "failed") { window.clearInterval(timer); setMessage(state.error); }
      }, 700);
    } catch (error: any) { setMessage(error.message); }
  };
  const execute = async (kind: "retrieve" | "ask") => {
    try {
      const nextRun = await api<LabRun>(`/runs/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ index_id: indexId, query: question, retriever, top_k: topK, context_budget: contextBudget, provider: kind === "ask" ? { base_url: providerUrl || undefined, model: providerModel || undefined, api_key: providerKey || undefined } : undefined }) });
      setRun(nextRun); setActiveStage(kind === "ask" ? 5 : 3); setArea("explore");
    } catch (error: any) { setMessage(error.message); }
  };
  const replayStarter = async () => {
    try { setRun(await api<LabRun>("/starter-run")); setActiveStage(0); setArea("explore"); } catch (error: any) { setMessage(error.message); }
  };
  const importWatson = async () => { try { const job = await api<{ status: string }>("/corpora/watsonxdocsqa/import", { method: "POST" }); setMessage(`${t.watsonStatus}: ${job.status}`); } catch (error: any) { setMessage(error.message); } };
  const downloadModel = async () => { try { const job = await api<{ status: string }>("/models/default/download", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }); setMessage(`${t.downloadModel}: ${job.status}`); } catch (error: any) { setMessage(error.message); } };
  const navigation = useMemo<Array<[Area, string]>>(() => [
    ["home", t.areas.home], ["explore", t.areas.explore], ["build", t.areas.build], ["failure", t.areas.failure], ["settings", t.areas.settings],
  ], [t]);

  return <main className="app-shell" data-motion={reducedMotion ? "reduced" : "full"}>
    <header className="app-header"><div className="brand"><h1>{t.title}</h1><p>{t.subtitle}</p></div><button className="language-toggle" type="button" onClick={() => setLang(lang === "en" ? "zh" : "en")}>{lang === "en" ? "中文" : "English"}</button></header>
    <nav aria-label="Lab areas">{navigation.map(([id, label]) => <button type="button" className={area === id ? "active" : ""} aria-current={area === id ? "page" : undefined} key={id} onClick={() => setArea(id)}>{label}</button>)}</nav>
    {message && <p className="notice" role="status">{message}</p>}
    {area === "home" && <HomeView onReplay={replayStarter} onBuild={() => { setBuildView("build"); setArea("build"); }} t={t} />}
    {area === "build" && <BuildInspectView view={buildView} setView={setBuildView} corpora={corpora} indexes={indexes} corpusId={corpusId} indexId={indexId} backend={backend} modelReady={modelReady} detail={detail} onCorpus={setCorpusId} onIndex={setIndexId} onBackend={setBackend} onUpload={upload} onImportWatson={importWatson} onBuild={build} t={t} />}
    {area === "explore" && <ExploreView indexes={indexes} indexId={indexId} question={question} retriever={retriever} topK={topK} contextBudget={contextBudget} run={run} activeStage={activeStage} lang={lang} onIndex={setIndexId} onQuestion={setQuestion} onRetriever={setRetriever} onTopK={setTopK} onContextBudget={setContextBudget} onStage={setActiveStage} onRun={execute} t={t} />}
    {area === "failure" && <FailureLabView lessons={lessons} lessonId={lessonId} onLesson={setLessonId} lang={lang} t={t} />}
    {area === "settings" && <SettingsView modelReady={modelReady} providerUrl={providerUrl} providerModel={providerModel} providerKey={providerKey} onDownload={downloadModel} onProviderUrl={setProviderUrl} onProviderModel={setProviderModel} onProviderKey={setProviderKey} t={t} />}
    <footer className="app-footer"><div><strong>tiny-rag-lab</strong><span>{lang === "en" ? "Created by James Wei · Learn classic RAG, locally." : "由 James Wei 创建 · 在本地学习经典 RAG。"}</span></div><div className="footer-links"><a href="https://github.com/jameswei/tiny-rag-lab" target="_blank" rel="noreferrer">GitHub</a><a href="https://jameswei.github.io/tiny-rag-lab/" target="_blank" rel="noreferrer">{lang === "en" ? "Project site" : "项目主页"}</a></div></footer>
  </main>;
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<App />);
