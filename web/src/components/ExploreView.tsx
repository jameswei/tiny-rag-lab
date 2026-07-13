import { learningMaterialUrl, type Copy } from "../copy";
import type { Evidence, IndexItem, LabRun, Stage } from "../types";
import { EvidenceCard } from "./EvidenceCard";
import { Pipeline } from "./Pipeline";
import { RawArtifact } from "./RawArtifact";
import { VectorPreview } from "./BuildInspectView";

export function ExploreView({
  indexes, indexId, question, retriever, topK, contextBudget, run, activeStage, lang, onIndex, onQuestion, onRetriever, onTopK, onContextBudget, onStage, onRun, t,
}: {
  indexes: IndexItem[]; indexId: string; question: string; retriever: "dense" | "bm25" | "hybrid"; topK: number; contextBudget: number; run: LabRun | null; activeStage: Stage; lang: "en" | "zh";
  onIndex: (value: string) => void; onQuestion: (value: string) => void; onRetriever: (value: "dense" | "bm25" | "hybrid") => void; onTopK: (value: number) => void; onContextBudget: (value: number) => void; onStage: (value: Stage) => void; onRun: (kind: "retrieve" | "ask") => void; t: Copy;
}) {
  return <section className="view explore-view" aria-labelledby="explore-title">
    <span className="kicker">{t.areas.explore}</span><h2 id="explore-title">{t.areas.explore}</h2><p className="lead">{t.exploreIntro}</p>
    <section className="run-setup" aria-label={t.runConfiguration}><div className="control-row">
      <label>{t.index}<select value={indexId} onChange={(event) => onIndex(event.target.value)}><option value="">—</option>{indexes.map((index) => <option key={index.id} value={index.id}>{index.id}</option>)}</select></label>
      <label>{t.retriever}<select value={retriever} onChange={(event) => onRetriever(event.target.value as "dense" | "bm25" | "hybrid")}><option value="dense">{t.retrieverOptions.dense}</option><option value="bm25">{t.retrieverOptions.bm25}</option><option value="hybrid">{t.retrieverOptions.hybrid}</option></select></label>
      <label>{t.topK}<input type="number" min="1" max="50" value={topK} onChange={(event) => onTopK(Number(event.target.value))} /></label>
      <label>{t.contextBudget}<input type="number" min="0" value={contextBudget} onChange={(event) => onContextBudget(Number(event.target.value))} /></label>
    </div>
    <label className="question-field"><span>{t.question}</span><input aria-label={t.question} placeholder={t.question} value={question} onChange={(event) => onQuestion(event.target.value)} /></label>
    <div className="action-row run-actions"><button type="button" disabled={!indexId || !question} onClick={() => onRun("retrieve")}>{t.retrieve}</button><button type="button" className="primary-action" disabled={!indexId || !question} onClick={() => onRun("ask")}>{t.ask}</button></div>
    <p className="hint">{t.provider}</p></section>
    {run ? <div className="run-artifact"><Pipeline activeStage={activeStage} onSelect={onStage} t={t} /><div className="stage-intro"><span>{t.stages[activeStage]}</span><p>{t.stageBriefs[activeStage]}</p></div><StageView run={run} activeStage={activeStage} lang={lang} t={t} /></div> : <div className="empty-run"><p>{t.noIndex}</p></div>}
  </section>;
}

function StageView({ run, activeStage, lang, t }: { run: LabRun; activeStage: Stage; lang: "en" | "zh"; t: Copy }) {
  const trace = run.trace;
  const selected = run.evidence.filter((item) => item.selected_for_context !== false);
  const material = activeStage < 2 ? "the-indexing-plane.md" : activeStage < 4 ? "retrieval-mechanics.md" : activeStage === 4 ? "context-budget-and-structured-answers.md" : "retrieval-and-generation.md";
  const learningLink = <a className="learning-link" href={learningMaterialUrl(lang, material)} target="_blank" rel="noreferrer">{t.learn}</a>;
  if (activeStage === 0) return <><dl className="fact-grid"><div><dt>{t.documents}</dt><dd>{run.index.document_count ?? "—"}</dd></div><div><dt>{t.chunks}</dt><dd>{run.index.chunk_count ?? run.index.manifest?.chunk_count ?? "—"}</dd></div><div><dt>{t.backend}</dt><dd>{String(run.index.manifest?.index_backend || "numpy")}</dd></div></dl>{learningLink}<RawArtifact label={t.raw} value={run.index.manifest} /></>;
  if (activeStage === 1) return <><EvidenceList evidence={run.evidence} t={t} compact />{learningLink}</>;
  if (activeStage === 2) return <section className="vector-stage"><h3>{t.vector}</h3><VectorPreview vector={run.query_vector || []} t={t} />{learningLink}<RawArtifact label={t.raw} value={run.query_vector || []} /></section>;
  if (activeStage === 3) return <><EvidenceList evidence={run.evidence} t={t} />{learningLink}<RawArtifact label={t.timings} value={trace.latency_by_stage || {}} /></>;
  if (activeStage === 4) return <><EvidenceList evidence={selected} t={t} />{learningLink}<RawArtifact label={t.prompt} value={trace.prompt || trace.context_pack || {}} /></>;
  return <section className="answer-stage">{trace.answer && <article className="answer-card"><h3>{t.answer}</h3><p>{trace.answer}</p></article>}{Array.isArray(trace.citations) && <section className="citation-list"><h3>{t.citations}</h3><ul>{trace.citations.map((citation: string) => <li key={citation}>{citation}</li>)}</ul></section>}<EvidenceList evidence={selected} t={t} compact />{learningLink}<RawArtifact label={t.raw} value={trace} /></section>;
}

function EvidenceList({ evidence, t, compact = false }: { evidence: Evidence[]; t: Copy; compact?: boolean }) {
  return <div className="evidence-list">{evidence.map((item, index) => <EvidenceCard key={item.chunk_id || `${item.doc_id}-${index}`} evidence={item} t={t} compact={compact} />)}</div>;
}
