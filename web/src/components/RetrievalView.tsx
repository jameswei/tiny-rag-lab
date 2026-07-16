import { useEffect, useRef, useState } from "react";
import { learningMaterialUrl, type Copy } from "../copy";
import type { Evidence, LabRun, Lang, RetrievalMaterial, RetrievalModule } from "../types";
import { EvidenceCard } from "./EvidenceCard";
import { RawArtifact } from "./RawArtifact";

const modules: RetrievalModule[] = ["lexical", "dense", "vector-db", "hybrid", "reranking", "evaluation"];

export function RetrievalView({
  module, materials, materialId, run, running, modelReady, rerankerReady, lang, onModule, onMaterial, onRun, t,
}: {
  module: RetrievalModule;
  materials: RetrievalMaterial[];
  materialId: string;
  run: LabRun | null;
  running: boolean;
  modelReady: boolean | null;
  rerankerReady: boolean | null;
  lang: Lang;
  onModule: (module: RetrievalModule) => void;
  onMaterial: (id: string) => void;
  onRun: () => void;
  t: Copy;
}) {
  const category = ["lexical", "dense", "hybrid", "reranking"].includes(module)
    ? module as "lexical" | "dense" | "hybrid" | "reranking"
    : null;
  const choices = category ? materials.filter((item) => item.category === category) : [];
  const selected = materials.find((item) => item.question_id === materialId);
  const ready = module === "lexical"
    || ((module === "dense" || module === "hybrid") && modelReady === true)
    || (module === "reranking" && modelReady === true && rerankerReady === true);
  const guidePage = module === "reranking"
    ? "reranking.md"
    : module === "evaluation" ? "evaluating-retrieval.md" : "retrieval-mechanics.md";

  return <section className="view retrieval-view" aria-labelledby="retrieval-title">
    <span className="kicker">{t.areas.retrieval}</span>
    <h2 id="retrieval-title">{lang === "en" ? "How retrieval decides" : "检索如何作出选择"}</h2>
    <p className="lead">{t.retrievalIntro}</p>

    <ol className="retrieval-module-rail">
      {modules.map((id, index) => <li key={id}>
        <button type="button" disabled={running} className={module === id ? "active" : ""} onClick={() => onModule(id)}>
          <small>{String(index + 1).padStart(2, "0")}</small>
          <strong>{t.retrievalModules[id].title}</strong>
          <span>{t.retrievalModules[id].description}</span>
        </button>
      </li>)}
    </ol>

    <section className="retrieval-workbench">
      <header><span className="kicker">{t.retrievalModules[module].title}</span><h3>{t.retrievalModules[module].description}</h3></header>
      {module === "vector-db" ? <QdrantModule materials={materials} lang={lang} t={t} /> : module === "evaluation" ? <EvaluationModule lang={lang} t={t} /> : category ? <>
        {choices.length > 0 ? <div className="retrieval-controls">
          <label>{t.curatedQuestion}<select disabled={running} value={materialId} onChange={(event) => onMaterial(event.target.value)}>{choices.map((item) => <option value={item.question_id} key={item.question_id}>{item.question}</option>)}</select></label>
          <button className="primary-action" type="button" disabled={running || !materialId || !ready} onClick={onRun}>{running ? t.runningLesson : t.runLesson}</button>
        </div> : <p className="status-panel status-caution course-empty">{t.courseUnavailable}</p>}
        {module !== "lexical" && modelReady === false && <p className="status-panel status-info course-empty">{t.modelDownloadRequired}</p>}
        {module === "reranking" && rerankerReady === false && <p className="status-panel status-info course-empty">{t.rerankerDownloadRequired}</p>}
        {selected?.teaching_note && <blockquote className="teaching-note">{selected.teaching_note[lang]}</blockquote>}
        {selected && <div className="reviewed-sources"><strong>{t.reviewedSources}</strong><span>{selected.gold_doc_ids.join(", ")}</span></div>}
        {run && category && <RetrievalResult module={category} run={run} lang={lang} t={t} />}
      </> : <p className="status-panel status-info course-empty">{t.moduleComing}</p>}
    </section>
    <a className="learning-link" href={learningMaterialUrl(lang, guidePage)} target="_blank" rel="noreferrer">{t.learn}</a>
  </section>;
}

type QdrantStatus = {
  available: boolean;
  prepared: boolean;
  launch_command: string;
  source_fingerprint: string;
  collection: { alias: string; collection: string; point_count: number; dimension: number; reused: boolean; verified: boolean } | null;
  filters: string[];
};

type EvaluationConfig = { retriever: "bm25" | "dense" | "hybrid"; top_k: number; reranker: "none" | "cross-encoder"; rerank_top_n: number };
type EvaluationPreset = { id: string; left: EvaluationConfig; right: EvaluationConfig };
type EvaluationStatus = { ready: boolean; reason: string | null; question_count: number; source_vector_fingerprint?: string; presets: EvaluationPreset[] };
type EvaluationJob = { id: string; status: string; progress: { current: number; total: number | null; message: string }; error?: string; left?: EvaluationConfig; right?: EvaluationConfig };
type EvaluationQuestion = { question_id: string; question: string; category: string; gold_doc_ids: string[]; metrics: { hit: number; reciprocal_rank: number; context_precision: number; context_recall: number }; evidence: Evidence[] };
type EvaluationSide = { config: EvaluationConfig; metrics: { n_questions: number; hit_rate: number; mrr: number; context_precision: number; context_recall: number }; questions: EvaluationQuestion[] };
type EvaluationResult = { question_count: number; left: EvaluationSide; right: EvaluationSide; bundle: { index_id: string; source_vector_fingerprint: string } };

function EvaluationModule({ lang, t }: { lang: Lang; t: Copy }) {
  const [status, setStatus] = useState<EvaluationStatus | null>(null);
  const [presetId, setPresetId] = useState("");
  const [left, setLeft] = useState<EvaluationConfig | null>(null);
  const [right, setRight] = useState<EvaluationConfig | null>(null);
  const [job, setJob] = useState<EvaluationJob | null>(null);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [questionId, setQuestionId] = useState("");
  const [error, setError] = useState("");
  const timer = useRef<number | null>(null);
  const mounted = useRef(true);

  const request = async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`/api${path}`, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || body.error || response.statusText);
    return body;
  };
  const applyPreset = (preset: EvaluationPreset) => {
    setPresetId(preset.id); setLeft({ ...preset.left }); setRight({ ...preset.right }); setResult(null); setQuestionId(""); setError("");
  };
  const loadResult = async (jobId: string) => {
    const value = await request<EvaluationResult>(`/jobs/${jobId}/result`);
    if (!mounted.current) return;
    setResult(value); setQuestionId(value.left.questions[0]?.question_id || "");
  };
  const poll = async (jobId: string) => {
    try {
      const value = await request<EvaluationJob>(`/jobs/${jobId}`);
      if (!mounted.current) return;
      setJob(value);
      if (value.status === "complete") await loadResult(jobId);
      else if (["queued", "running", "cancel_requested", "publishing"].includes(value.status) && mounted.current) timer.current = window.setTimeout(() => void poll(jobId), 700);
    } catch (reason: any) { if (mounted.current) setError(reason.message); }
  };

  useEffect(() => {
    mounted.current = true;
    let active = true;
    const initialize = async () => {
      try {
        const value = await request<EvaluationStatus>("/evaluations/status");
        if (!active) return;
        setStatus(value);
        if (value.presets[0]) applyPreset(value.presets[0]);
        const activeJobs = await request<{ items: EvaluationJob[] }>("/jobs/active?kind=evaluation");
        if (active && activeJobs.items[0]) {
          const recovered = activeJobs.items[0];
          setJob(recovered);
          if (recovered.left && recovered.right) {
            setLeft({ ...recovered.left }); setRight({ ...recovered.right });
            const matching = value.presets.find((preset) => JSON.stringify(preset.left) === JSON.stringify(recovered.left) && JSON.stringify(preset.right) === JSON.stringify(recovered.right));
            setPresetId(matching?.id || "");
          }
          void poll(recovered.id);
        }
      } catch (reason: any) { if (active) setError(reason.message); }
    };
    void initialize();
    return () => { active = false; mounted.current = false; if (timer.current !== null) window.clearTimeout(timer.current); };
  }, []);

  const start = async () => {
    if (!left || !right) return;
    setError(""); setResult(null); setQuestionId("");
    try {
      const value = await request<{ id: string; status: string }>("/evaluations", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ left, right }),
      });
      const next = { id: value.id, status: value.status, progress: { current: 0, total: status?.question_count || 16, message: lang === "en" ? "Queued" : "已排队" } };
      setJob(next); void poll(value.id);
    } catch (reason: any) { setError(reason.message); }
  };
  const cancel = async () => {
    if (!job) return;
    try { setJob(await request<EvaluationJob>(`/jobs/${job.id}/cancel`, { method: "POST" })); }
    catch (reason: any) { setError(reason.message); }
  };
  const busy = !!job && ["queued", "running", "cancel_requested", "publishing"].includes(job.status);
  const selectedLeft = result?.left.questions.find((item) => item.question_id === questionId);
  const selectedRight = result?.right.questions.find((item) => item.question_id === questionId);

  if (!status) return <p className="status-panel status-info course-empty">{error || (lang === "en" ? "Checking the reviewed evaluation bundle…" : "正在检查评审过的评估包……")}</p>;
  if (!status.ready) return <p className="status-panel status-caution course-empty">{status.reason}</p>;
  return <section className="evaluation-module">
    <blockquote className="teaching-note">{lang === "en" ? "Compare retrieval choices over the same fixed index and 16 reviewed questions. Each metric answers a different question; there is no composite winner." : "在同一个固定索引和 16 个评审问题上比较检索选择。每项指标回答不同问题，这里不会给出综合赢家。"}</blockquote>
    <dl className="fact-grid evaluation-bundle-facts"><div><dt>{t.index}</dt><dd><code>cloudflare-state-structural-v1</code></dd></div><div><dt>{lang === "en" ? "Reviewed set" : "评审题集"}</dt><dd>{status.question_count} {lang === "en" ? "questions" : "个问题"}</dd></div><div><dt>{t.sourceIdentity}</dt><dd><code>{status.source_vector_fingerprint?.slice(0, 16)}…</code></dd></div></dl>
    <div className="evaluation-preset"><label>{lang === "en" ? "Comparison preset" : "比较预设"}<select disabled={busy} value={presetId} onChange={(event) => { const preset = status.presets.find((item) => item.id === event.target.value); if (preset) applyPreset(preset); }}>{presetId === "" && <option value="">{lang === "en" ? "Recovered custom comparison" : "已恢复的自定义比较"}</option>}{status.presets.map((preset) => <option key={preset.id} value={preset.id}>{presetLabel(preset.id, lang)}</option>)}</select></label><span>{status.question_count} {lang === "en" ? "reviewed questions" : "个评审问题"}</span></div>
    {left && right && <div className="evaluation-configs"><EvaluationConfigEditor label={lang === "en" ? "Configuration A" : "配置 A"} value={left} disabled={busy} onChange={setLeft} lang={lang} t={t} /><EvaluationConfigEditor label={lang === "en" ? "Configuration B" : "配置 B"} value={right} disabled={busy} onChange={setRight} lang={lang} t={t} /></div>}
    <div className="evaluation-actions"><button type="button" className="primary-action" disabled={busy} onClick={() => void start()}>{lang === "en" ? "Run comparison" : "运行比较"}</button>{busy && <button type="button" disabled={job?.status === "cancel_requested" || job?.status === "publishing"} onClick={() => void cancel()}>{job?.status === "cancel_requested" ? (lang === "en" ? "Stopping after the current operation…" : "将在当前操作后停止……") : (lang === "en" ? "Cancel" : "取消")}</button>}</div>
    {job && <div data-status={job.status} className={`status-panel ${job.status === "failed" ? "status-error" : job.status === "complete" ? "status-success" : "status-info"} evaluation-progress`} aria-live="polite">
      <div className="evaluation-progress-head"><div><span aria-hidden="true" /><strong>{jobStatusLabel(job.status, lang)}</strong></div>{job.progress.total && <output>{lang === "en" ? `${job.progress.current} of ${job.progress.total} questions` : `${job.progress.current} / ${job.progress.total} 个问题`}</output>}</div>
      <p>{job.error || jobProgressLabel(job, lang)}</p>
      {job.progress.total && <div className="evaluation-progress-meter" role="progressbar" aria-label={lang === "en" ? "Evaluation progress" : "评估进度"} aria-valuemin={0} aria-valuemax={job.progress.total} aria-valuenow={job.progress.current}><span style={{ width: `${Math.min(100, Math.max(0, job.progress.current / job.progress.total * 100))}%` }} /></div>}
    </div>}
    {error && <p className="status-panel status-error course-empty">{error}</p>}
    {result && <EvaluationResults result={result} questionId={questionId} onQuestion={setQuestionId} leftQuestion={selectedLeft} rightQuestion={selectedRight} lang={lang} t={t} />}
  </section>;
}

function EvaluationConfigEditor({ label, value, disabled, onChange, lang, t }: { label: string; value: EvaluationConfig; disabled: boolean; onChange: (value: EvaluationConfig) => void; lang: Lang; t: Copy }) {
  const field = <K extends keyof EvaluationConfig,>(key: K, next: EvaluationConfig[K]) => onChange({ ...value, [key]: next });
  const topK = (next: number) => onChange({ ...value, top_k: next, rerank_top_n: Math.max(next, value.rerank_top_n) });
  return <fieldset className="evaluation-config" disabled={disabled}><legend>{label}</legend><label>{t.retriever}<select value={value.retriever} onChange={(event) => field("retriever", event.target.value as EvaluationConfig["retriever"])}><option value="bm25">BM25</option><option value="dense">{lang === "en" ? "Dense" : "语义检索"}</option><option value="hybrid">{lang === "en" ? "Hybrid (RRF)" : "混合（RRF）"}</option></select></label><label>{t.topK}<input type="number" min="1" max="20" value={value.top_k} onChange={(event) => topK(Number(event.target.value))} /></label><label>{t.reranker}<select value={value.reranker} onChange={(event) => field("reranker", event.target.value as EvaluationConfig["reranker"])}><option value="none">{t.noReranker}</option><option value="cross-encoder">{t.crossEncoder}</option></select></label><label>{t.candidateDepth}<input type="number" min={value.top_k} max="50" disabled={disabled || value.reranker === "none"} value={value.rerank_top_n} onChange={(event) => field("rerank_top_n", Number(event.target.value))} /></label></fieldset>;
}

function EvaluationResults({ result, questionId, onQuestion, leftQuestion, rightQuestion, lang, t }: { result: EvaluationResult; questionId: string; onQuestion: (id: string) => void; leftQuestion?: EvaluationQuestion; rightQuestion?: EvaluationQuestion; lang: Lang; t: Copy }) {
  const metrics = [{ key: "hit_rate", label: lang === "en" ? "Hit rate" : "命中率" }, { key: "mrr", label: "MRR" }, { key: "context_precision", label: lang === "en" ? "Context precision" : "上下文精确率" }, { key: "context_recall", label: lang === "en" ? "Context recall" : "上下文召回率" }] as const;
  return <section className="evaluation-results"><h3>{lang === "en" ? "Aggregate results" : "汇总结果"}</h3><div className="metric-comparison">{metrics.map(({ key, label }) => <article key={key}><strong>{label}</strong><span>A <b>{format(result.left.metrics[key])}</b></span><span>B <b>{format(result.right.metrics[key])}</b></span><small>Δ {format(result.right.metrics[key] - result.left.metrics[key])}</small></article>)}</div><p className="hint">{lang === "en" ? "A delta is B minus A. Inspect the questions below before interpreting an aggregate change." : "差值为 B 减 A。解读汇总变化前，请先检查下方的逐题结果。"}</p><label className="evaluation-question-picker">{lang === "en" ? "Inspect one question" : "检查一道问题"}<select value={questionId} onChange={(event) => onQuestion(event.target.value)}>{result.left.questions.map((item) => <option key={item.question_id} value={item.question_id}>{item.question}</option>)}</select></label>{leftQuestion && rightQuestion && <><div className="evaluation-question-head"><span>{categoryLabel(leftQuestion.category, lang)}</span><h4>{leftQuestion.question}</h4><p>{lang === "en" ? "Gold documents" : "标准文档"}: {leftQuestion.gold_doc_ids.join(", ")}</p></div><div className="comparison evaluation-evidence"><EvaluationQuestionSide label="A" item={leftQuestion} lang={lang} t={t} /><EvaluationQuestionSide label="B" item={rightQuestion} lang={lang} t={t} /></div></>}</section>;
}

function EvaluationQuestionSide({ label, item, lang, t }: { label: string; item: EvaluationQuestion; lang: Lang; t: Copy }) {
  return <section><h4>{label}</h4><div className="question-metrics"><span>{item.metrics.hit ? (lang === "en" ? "Hit" : "命中") : (lang === "en" ? "Miss" : "未命中")}</span><span>RR {format(item.metrics.reciprocal_rank)}</span><span>P {format(item.metrics.context_precision)}</span><span>R {format(item.metrics.context_recall)}</span></div><div className="evaluation-evidence-list">{item.evidence.map((evidence) => <EvidenceCard key={evidence.chunk_id} evidence={evidence} t={t} />)}</div></section>;
}

function presetLabel(id: string, lang: Lang) {
  const labels: Record<string, [string, string]> = { "bm25-vs-dense": ["BM25 vs Dense", "BM25 对比语义检索"], "dense-vs-hybrid": ["Dense vs Hybrid", "语义检索对比混合检索"], "hybrid-vs-reranked": ["Hybrid vs Hybrid + cross-encoder", "混合检索对比混合检索 + 交叉编码器"] };
  return labels[id]?.[lang === "en" ? 0 : 1] || id;
}

function jobStatusLabel(status: string, lang: Lang) {
  const labels: Record<string, [string, string]> = { queued: ["Queued", "已排队"], running: ["Running", "运行中"], cancel_requested: ["Stopping", "正在停止"], publishing: ["Publishing", "正在发布"], complete: ["Complete", "已完成"], failed: ["Failed", "失败"], cancelled: ["Cancelled", "已取消"] };
  return labels[status]?.[lang === "en" ? 0 : 1] || status;
}

function jobProgressLabel(job: EvaluationJob, lang: Lang) {
  if (lang === "en") return job.progress.message;
  if (job.status === "queued") return "等待本地评估开始";
  if (job.status === "cancel_requested") return "将在当前模型或检索操作返回后停止";
  if (job.status === "publishing") return "正在原子发布比较结果";
  if (job.status === "complete") return "比较已完成";
  if (job.status === "cancelled") return "比较已取消，未发布结果";
  return `已比较 ${job.progress.current} / ${job.progress.total || 16} 个问题`;
}

function categoryLabel(category: string, lang: Lang) {
  if (lang === "en") return category;
  return ({ lexical: "词法检索", dense: "语义检索", hybrid: "混合检索", reranking: "重排" } as Record<string, string>)[category] || category;
}
type VectorDbHit = Evidence & { payload?: Record<string, unknown> | null };
type QdrantComparison = {
  question: string;
  numpy: VectorDbHit[];
  qdrant: VectorDbHit[];
  parity: { equivalent: boolean; score_tolerance: number; items: Array<{ chunk_id: string; numpy_rank: number | null; qdrant_rank: number | null; equivalent: boolean }> };
  source_group: string | null;
  filtered_qdrant: VectorDbHit[];
};

function QdrantModule({ materials, lang, t }: { materials: RetrievalMaterial[]; lang: Lang; t: Copy }) {
  const questions = materials.filter((item) => item.category === "dense");
  const [status, setStatus] = useState<QdrantStatus | null>(null);
  const [materialId, setMaterialId] = useState(questions[0]?.question_id || "");
  const [sourceGroup, setSourceGroup] = useState("");
  const [comparison, setComparison] = useState<QdrantComparison | null>(null);
  const [busy, setBusy] = useState<"prepare" | "compare" | null>(null);
  const [error, setError] = useState("");

  const request = async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`/api${path}`, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || body.error || response.statusText);
    return body;
  };
  const refresh = async () => setStatus(await request<QdrantStatus>("/retrieval/qdrant/status"));
  useEffect(() => { void refresh().catch((reason: Error) => setError(reason.message)); }, []);
  useEffect(() => { if (!materialId && questions[0]) setMaterialId(questions[0].question_id); }, [materialId, questions]);

  const prepare = async () => {
    setBusy("prepare"); setError("");
    try { await request("/retrieval/qdrant/prepare", { method: "POST" }); await refresh(); }
    catch (reason: any) { setError(reason.message); }
    finally { setBusy(null); }
  };
  const compare = async () => {
    setBusy("compare"); setError("");
    try {
      setComparison(await request<QdrantComparison>("/retrieval/qdrant/compare", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ retrieval_material_id: materialId, top_k: 5, source_group: sourceGroup || null }),
      }));
    } catch (reason: any) { setError(reason.message); }
    finally { setBusy(null); }
  };

  if (!status) return <p className="status-panel status-info course-empty">{error || t.vectorDbChecking}</p>;
  return <section className="vector-db-module">
    <div className="concept-shift"><article><span>01</span><strong>NumPy</strong><p>{t.numpyRole}</p></article><span aria-hidden="true">→</span><article><span>02</span><strong>Qdrant</strong><p>{t.qdrantRole}</p></article></div>
    <p className="backend-continuity"><strong>{t.sameRetrievalFlowTitle}</strong>{t.sameRetrievalFlow}</p>
    {!status.available ? <div className="status-panel status-caution vector-db-state"><h4>{t.qdrantNotRunning}</h4><p>{t.qdrantOptional}</p><code>{status.launch_command}</code></div> : !status.prepared ? <div className="status-panel status-info vector-db-state"><h4>{t.qdrantReadyToPrepare}</h4><p>{t.qdrantCopyExact}</p><button type="button" className="primary-action" disabled={busy !== null} onClick={() => void prepare()}>{busy === "prepare" ? t.qdrantPreparing : t.qdrantPrepare}</button></div> : <>
      <dl className="fact-grid vector-db-facts"><div><dt>{t.vectorPointCount}</dt><dd>{status.collection?.point_count ?? "—"} {t.vectorPoints}</dd></div><div><dt>{t.vectorDimension}</dt><dd>{status.collection?.dimension ?? "—"}D</dd></div><div><dt>{t.backend}</dt><dd>Qdrant · exact</dd></div><div><dt>{t.sourceIdentity}</dt><dd><code>{status.source_fingerprint.slice(0, 16)}…</code></dd></div></dl>
      <div className="vector-db-maintenance"><p>{t.qdrantPreparedState}</p><button type="button" disabled={busy !== null} onClick={() => void prepare()}>{busy === "prepare" ? t.qdrantPreparing : t.qdrantVerifyAgain}</button></div>
      <div className="retrieval-controls vector-db-controls"><label>{t.curatedQuestion}<select disabled={busy !== null} value={materialId} onChange={(event) => { setMaterialId(event.target.value); setComparison(null); }}>{questions.map((item) => <option key={item.question_id} value={item.question_id}>{item.question}</option>)}</select></label><label>{t.payloadFilter}<select disabled={busy !== null} value={sourceGroup} onChange={(event) => setSourceGroup(event.target.value)}><option value="">{t.noFilter}</option>{status.filters.map((filter) => <option key={filter} value={filter}>{filter}</option>)}</select></label><button type="button" className="primary-action" disabled={busy !== null || !materialId} onClick={() => void compare()}>{busy === "compare" ? t.comparing : t.compareBackends}</button></div>
      {comparison && <VectorDbComparison comparison={comparison} t={t} />}
    </>}
    {error && <p className="status-panel status-error course-empty">{error}</p>}
  </section>;
}

function VectorDbComparison({ comparison, t }: { comparison: QdrantComparison; t: Copy }) {
  return <section className="vector-db-comparison">
    <div className={`status-panel ${comparison.parity.equivalent ? "status-success" : "status-error"} parity-result`}><strong>{comparison.parity.equivalent ? t.parityMatch : t.parityMismatch}</strong><span>{t.parityTolerance} {comparison.parity.score_tolerance}</span></div>
    <div className="comparison"><VectorResultList title="NumPy" description={t.numpyMetadataSource} items={comparison.numpy} t={t} /><VectorResultList title="Qdrant · exact" description={t.qdrantPayloadSource} items={comparison.qdrant} t={t} showPayload /></div>
    {comparison.source_group && <section className="filtered-results"><h4>{t.filteredResults}: {comparison.source_group}</h4><p className="hint">{t.filterSeparate}</p><VectorResultList title="Qdrant · filtered" items={comparison.filtered_qdrant} t={t} showPayload /></section>}
  </section>;
}

function VectorResultList({ title, description, items, t, showPayload = false }: { title: string; description?: string; items: VectorDbHit[]; t: Copy; showPayload?: boolean }) {
  return <section className="vector-result-list"><h4>{title}</h4>{description && <p>{description}</p>}{items.map((item) => <article key={item.chunk_id} className="vector-result"><span className="evidence-rank">#{item.rank}</span><div><strong>{item.title || item.doc_id}</strong><small>{item.path}</small><code>{item.chunk_id}</code></div><b>{format(item.score)}</b>{showPayload && item.payload && <RawArtifact label={t.storedPayload} value={item.payload} />}</article>)}</section>;
}

function RetrievalResult({ module, run, lang, t }: { module: "lexical" | "dense" | "hybrid" | "reranking"; run: LabRun; lang: Lang; t: Copy }) {
  const explanation = run.explanations || {};
  return <section className="retrieval-result" aria-live="polite">
    <p className="stage-focus">{t.exactArtifacts}</p>
    {module === "lexical" ? <LexicalResult run={run} explanation={explanation.bm25} t={t} />
      : module === "dense" ? <DenseResult run={run} explanation={explanation.dense} t={t} />
      : module === "hybrid" ? <HybridResult run={run} explanation={explanation.hybrid} t={t} />
      : <RerankingResult run={run} lang={lang} t={t} />}
    <RawArtifact label={t.raw} value={{ explanations: run.explanations, trace: run.trace }} />
  </section>;
}

function HybridResult({ run, explanation, t }: { run: LabRun; explanation: any; t: Copy }) {
  if (!explanation) return null;
  const evidence = new Map(run.evidence.map((item) => [item.chunk_id, item]));
  const sourceList = (title: string, items: any[]) => <section className="source-ranking"><h4>{title}</h4><ol>{items.map((item) => <li key={item.chunk_id}><span>#{item.rank}</span><code>{item.chunk_id}</code><b>{format(item.score)}</b></li>)}</ol></section>;
  return <>
    <section className="rrf-intro status-panel status-info"><strong>RRF</strong><span>{t.rrfFormula.replace("{k}", String(explanation.rrf_k))} {t.rrfMissingSource}</span></section>
    <div className="comparison hybrid-source-lists">{sourceList(t.denseRanking, explanation.dense)}{sourceList(t.lexicalRanking, explanation.bm25)}</div>
    <section className="rrf-results"><h3>{t.fusedRanking}</h3>{explanation.candidates.map((candidate: any) => {
      const sourceByName = new Map(candidate.sources.map((source: any) => [source.source, source]));
      return <article className="rrf-candidate" key={candidate.chunk_id}><header><span className="evidence-rank">#{candidate.rank}</span><div><strong>{evidence.get(candidate.chunk_id)?.title || evidence.get(candidate.chunk_id)?.doc_id || candidate.chunk_id}</strong><code>{candidate.chunk_id}</code></div><b>{format(candidate.score)}</b></header><div className="rrf-contributions">{(["dense", "bm25"] as const).map((sourceName) => {
        const source: any = sourceByName.get(sourceName);
        return source ? <span key={sourceName}><strong>{sourceName === "dense" ? t.denseRanking : t.lexicalRanking}</strong><small>#{source.rank} · 1 / ({explanation.rrf_k} + {source.rank})</small><b>+{format(source.contribution)}</b></span> : <span className="missing" key={sourceName}><strong>{sourceName === "dense" ? t.denseRanking : t.lexicalRanking}</strong><small>{t.notInSourceRanking}</small><b>+0</b></span>;
      })}</div></article>;
    })}</section>
  </>;
}

export function RerankingResult({ run, lang, t, compact = false }: { run: LabRun; lang: Lang; t: Copy; compact?: boolean }) {
  const explanation = run.explanations?.reranking;
  if (!explanation) return null;
  const candidates = new Map((run.candidates || []).map((item) => [item.chunk_id, item]));
  return <section className={`rerank-audit${compact ? " compact" : ""}`}>
    <div className="rerank-flow"><strong>{explanation.candidate_count} {t.firstStageCandidates}</strong><span aria-hidden="true">→</span><strong>{explanation.final_top_k} {t.finalEvidence}</strong></div>
    <div className="rerank-table-wrap"><table className="rerank-table"><caption>{t.candidateToFinal}</caption><thead><tr><th>{t.candidate}</th><th>{t.firstRank}</th><th>{t.firstScore}</th><th>{t.rerankerScore}</th><th>{t.finalRank}</th><th>{t.movement}</th></tr></thead><tbody>{explanation.candidates.map((item: any) => { const candidate = candidates.get(item.chunk_id); return <tr key={item.chunk_id} className={item.outcome === "dropped" ? "dropped" : ""}><th scope="row"><strong>{candidate?.title || candidate?.doc_id || item.chunk_id}</strong><code>{item.chunk_id}</code></th><td>#{item.pre_rank}</td><td>{format(item.pre_score)}</td><td>{format(item.reranker_score)}</td><td>{item.final_rank ? `#${item.final_rank}` : "—"}</td><td><span className={`movement ${item.outcome}`}>{t.rerankOutcomes[item.outcome as keyof typeof t.rerankOutcomes]}{item.rank_delta ? ` · ${item.rank_delta > 0 ? "+" : ""}${item.rank_delta}` : ""}</span></td></tr>; })}</tbody></table></div>
    {!compact && <p className="hint">{lang === "en" ? "Dropped candidates remain visible here for audit, but they do not enter the final context." : "被丢弃的候选仍保留在这里供检查，但不会进入最终上下文。"}</p>}
  </section>;
}

function LexicalResult({ run, explanation, t }: { run: LabRun; explanation: any; t: Copy }) {
  if (!explanation) return null;
  return <>
    <section className="token-panel"><h3>{t.questionTokens}</h3><div className="token-list">{explanation.query_tokens.map((token: string, index: number) => <code key={`${token}-${index}`}>{token}</code>)}</div><p><strong>BM25</strong> · k1={format(explanation.k1)} · b={format(explanation.b)} · N={explanation.corpus_size}</p><p>{t.bm25Meaning}</p><code className="bm25-formula">{t.bm25Formula}</code></section>
    <div className="retrieval-candidates">{run.evidence.map((evidence) => {
      const candidate = explanation.candidates.find((item: any) => item.chunk_id === evidence.chunk_id);
      return <article className="explained-candidate" key={evidence.chunk_id}>
        <EvidenceCard evidence={evidence} t={t} />
        {candidate && <div className="calculation-panel"><h4>{t.termContributions}</h4><div className="calculation-facts"><span>{t.bm25Columns.documentLength}<strong>{candidate.document_length}</strong></span><span>{t.bm25Columns.averageDocumentLength}<strong>{format(candidate.average_document_length)}</strong></span><span>{t.score}<strong>{format(candidate.score)}</strong></span></div><div className="term-table-wrap"><table className="term-table"><caption>{t.termContributions}</caption><thead><tr><th>{t.bm25Columns.term}</th><th>{t.bm25Columns.queryCount}</th><th>{t.bm25Columns.termFrequency}</th><th>{t.bm25Columns.documentFrequency}</th><th>{t.bm25Columns.inverseDocumentFrequency}</th><th>{t.bm25Columns.contribution}</th></tr></thead><tbody>{candidate.terms.map((term: any) => <tr key={term.term}><th scope="row"><code>{term.term}</code></th><td>{term.query_frequency}</td><td>{term.term_frequency}</td><td>{term.document_frequency}</td><td>{format(term.inverse_document_frequency)}</td><td><strong>{format(term.contribution)}</strong></td></tr>)}</tbody></table></div></div>}
      </article>;
    })}</div>
  </>;
}

function DenseResult({ run, explanation, t }: { run: LabRun; explanation: any; t: Copy }) {
  if (!explanation) return null;
  return <div className="retrieval-candidates">{run.evidence.map((evidence) => {
    const candidate = explanation.candidates.find((item: any) => item.chunk_id === evidence.chunk_id);
    return <article className="explained-candidate" key={evidence.chunk_id}>
      <EvidenceCard evidence={evidence} t={t} />
      {candidate && <div className="calculation-panel dense-calculation"><h4>{t.vectorMath}</h4><div className="calculation-facts"><span>{t.queryNorm}<strong>{format(candidate.query_norm)}</strong></span><span>{t.chunkNorm}<strong>{format(candidate.chunk_norm)}</strong></span><span>{t.dotProduct}<strong>{format(candidate.dot_product)}</strong></span><span>{t.cosine}<strong>{format(candidate.cosine_similarity)}</strong></span></div><p className="vector-reading-guide">{t.vectorReadingGuide}</p><div className="vector-pair"><SignedVector values={candidate.query_vector_preview} label={t.vector} /><SignedVector values={candidate.chunk_vector_preview} label={t.candidateVector} /></div></div>}
    </article>;
  })}</div>;
}

function SignedVector({ values, label }: { values: number[]; label: string }) {
  const maximum = Math.max(...values.map((value) => Math.abs(value)), 0.000001);
  const chartStyle = { "--component-count": Math.max(values.length, 1) } as React.CSSProperties;
  return <div className="signed-vector"><strong>{label}</strong><div className="signed-vector-chart" role="img" aria-label={label} style={chartStyle}><span className="vector-zero-axis" aria-hidden="true"><b>0</b></span>{values.map((value, index) => <span className={`vector-component ${value < 0 ? "negative" : "positive"}`} key={index} style={{ "--height": `${Math.max(4, Math.abs(value) / maximum * 36)}%` } as React.CSSProperties}><i /><small>{formatSigned(value)}</small></span>)}</div></div>;
}

function format(value: number) {
  return Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function formatSigned(value: number) {
  return value > 0 ? `+${format(value)}` : format(value);
}
