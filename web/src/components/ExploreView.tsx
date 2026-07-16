import { learningMaterialUrl, type Copy } from "../copy";
import type { CatalogQuestion, Evidence, IndexItem, LabRun, RetrievalMaterial, Stage } from "../types";
import { EvidenceCard } from "./EvidenceCard";
import { CitationList } from "./CitationList";
import { Pipeline } from "./Pipeline";
import { RawArtifact } from "./RawArtifact";
import { RerankingResult } from "./RetrievalView";
import { VectorPreview } from "./BuildInspectView";

export function ExploreView({
  indexes, indexId, question, catalogQuestions, catalogQuestionId, rerankingMaterials, retriever, topK, reranker, rerankTopN, rerankerReady, contextBudget, run, activeStage, lang, running, testingProvider, providerReady, onIndex, onQuestion, onCatalogQuestion, onRetriever, onTopK, onReranker, onRerankTopN, onContextBudget, onStage, onRun, t,
}: {
  indexes: IndexItem[]; indexId: string; question: string; catalogQuestions: CatalogQuestion[]; catalogQuestionId: string; rerankingMaterials: RetrievalMaterial[]; retriever: "dense" | "bm25" | "hybrid"; topK: number; reranker: "none" | "cross-encoder"; rerankTopN: number; rerankerReady: boolean | null; contextBudget: number; run: LabRun | null; activeStage: Stage; lang: "en" | "zh"; running: "retrieve" | "ask" | null; testingProvider: boolean; providerReady: boolean;
  onIndex: (value: string) => void; onQuestion: (value: string) => void; onCatalogQuestion: (value: string) => void; onRetriever: (value: "dense" | "bm25" | "hybrid") => void; onTopK: (value: number) => void; onReranker: (value: "none" | "cross-encoder") => void; onRerankTopN: (value: number) => void; onContextBudget: (value: number) => void; onStage: (value: Stage) => void; onRun: (kind: "retrieve" | "ask") => void; t: Copy;
}) {
  const rerankBlocked = reranker === "cross-encoder" && (rerankerReady !== true || rerankTopN < topK);
  return <section className="view explore-view" aria-labelledby="explore-title">
    <span className="kicker">{t.areas.explore}</span><h2 id="explore-title">{t.areas.explore}</h2><p className="lead">{t.exploreIntro}</p>
    <section className="run-setup" aria-label={t.runConfiguration}><div className="control-row">
      <label>{t.index}<select disabled={!!running} value={indexId} onChange={(event) => onIndex(event.target.value)}><option value="">—</option>{indexes.map((index) => <option key={index.id} value={index.id}>{index.id}</option>)}</select></label>
      <label>{t.retriever}<select disabled={!!running} value={retriever} onChange={(event) => onRetriever(event.target.value as "dense" | "bm25" | "hybrid")}><option value="dense">{t.retrieverOptions.dense}</option><option value="bm25">{t.retrieverOptions.bm25}</option><option value="hybrid">{t.retrieverOptions.hybrid}</option></select></label>
      <label>{t.topK}<input disabled={!!running} type="number" min="1" max="50" value={topK} onChange={(event) => onTopK(Number(event.target.value))} /></label>
      <label>{t.reranker}<select disabled={!!running} value={reranker} onChange={(event) => onReranker(event.target.value as "none" | "cross-encoder")}><option value="none">{t.noReranker}</option><option value="cross-encoder" disabled={rerankerReady !== true}>{t.crossEncoder}</option></select></label>
      {reranker !== "none" && <label>{t.candidateDepth}<input disabled={!!running} type="number" min={topK} max="50" value={rerankTopN} onChange={(event) => onRerankTopN(Number(event.target.value))} /></label>}
      <label>{t.contextBudget}<input disabled={!!running} type="number" min="0" value={contextBudget} onChange={(event) => onContextBudget(Number(event.target.value))} /></label>
    </div>
    {reranker !== "cross-encoder" && catalogQuestions.length > 0 && <label className="question-field"><span>{t.catalogQuestion}</span><select disabled={!!running} value={catalogQuestionId} onChange={(event) => onCatalogQuestion(event.target.value)}><option value="">{t.freeQuestion}</option>{catalogQuestions.map((item) => <option key={item.id} value={item.id}>{item.featured ? "★ " : ""}{item.question}</option>)}</select></label>}
    <label className="question-field"><span>{t.question}</span><input disabled={!!running || (reranker !== "cross-encoder" && !!catalogQuestionId)} aria-label={t.question} list={reranker === "cross-encoder" && rerankingMaterials.length > 0 ? "reviewed-reranking-questions" : undefined} placeholder={t.question} value={question} onChange={(event) => onQuestion(event.target.value)} />{reranker === "cross-encoder" && rerankingMaterials.length > 0 && <><datalist id="reviewed-reranking-questions">{rerankingMaterials.map((item) => <option key={item.question_id} value={item.question} />)}</datalist><small className="question-suggestion-hint">{t.rerankingQuestionHint}</small></>}</label>
    {rerankerReady === false && <p className="hint">{t.rerankerDownloadRequired}</p>}
    <div className="action-row run-actions"><button type="button" disabled={!!running || rerankBlocked || !indexId || (!question && !catalogQuestionId)} onClick={() => onRun("retrieve")}>{running === "retrieve" ? t.retrieving : t.retrieve}</button><button type="button" className="primary-action" disabled={!!running || rerankBlocked || testingProvider || !providerReady || !indexId || (!question && !catalogQuestionId)} onClick={() => onRun("ask")}>{running === "ask" ? t.generating : t.ask}</button></div>
    <p className="hint">{providerReady ? t.providerUnlocked : t.provider}</p></section>
    {run ? <div className="run-artifact"><Pipeline activeStage={activeStage} onSelect={onStage} t={t} /><StageView run={run} activeStage={activeStage} lang={lang} t={t} />{run.catalog_check && activeStage >= 3 && <section className={`catalog-check status-panel ${run.catalog_check.hit ? "status-success" : "status-caution"}`}><strong>{t.goldCheck}:</strong> {run.catalog_check.hit ? t.hit : t.miss}<dl><div><dt>{t.expectedSources}</dt><dd>{run.catalog_check.expected_document_ids.join(", ") || "—"}</dd></div><div><dt>{t.retrievedSources}</dt><dd>{run.catalog_check.retrieved_document_ids.join(", ") || "—"}</dd></div></dl></section>}</div> : <div className="empty-run status-panel status-info"><p>{t.noIndex}</p></div>}
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
  if (activeStage === 3) return <>{run.explanations?.reranking && <RerankingResult run={run} lang={lang} t={t} compact />}<EvidenceList evidence={run.evidence} t={t} />{learningLink}<RawArtifact label={t.timings} value={trace.latency_by_stage || {}} /></>;
  if (activeStage === 4) return <><EvidenceList evidence={selected} t={t} />{learningLink}<RawArtifact label={t.prompt} value={trace.prompt || trace.context_pack || {}} /></>;
  const abstained = typeof trace.answer === "string" && trace.answer.toLowerCase().includes("does not contain enough information");
  return <section className="answer-stage">{run.error ? <article className="answer-card failure status-panel status-error"><h3>{t.generationFailed}</h3><p>{run.error}</p></article> : <>{trace.answer && (abstained ? <article className="answer-card abstention status-panel status-caution"><span className="kicker">{t.abstention}</span><h3>{t.noSupportedAnswer}</h3><p>{t.abstained}</p></article> : <article className="answer-card answer-result"><h3>{t.answer}</h3><p>{trace.answer}</p></article>)}{Array.isArray(trace.citations) && trace.citations.length > 0 && <CitationList citations={trace.citations} evidence={selected} t={t} />}</>}{run.explanations?.reranking && <section className="generation-retrieval-audit"><h3>{t.generationRetrievalAudit}</h3><p className="hint">{t.generationRetrievalAuditHint}</p><RerankingResult run={run} lang={lang} t={t} /></section>}<h3>{t.selectedEvidence}</h3><EvidenceList evidence={selected} t={t} />{learningLink}<RawArtifact label={t.raw} value={trace} /></section>;
}

function EvidenceList({ evidence, t, compact = false }: { evidence: Evidence[]; t: Copy; compact?: boolean }) {
  return <div className="evidence-list">{evidence.map((item, index) => <EvidenceCard key={item.chunk_id || `${item.doc_id}-${index}`} evidence={item} t={t} compact={compact} />)}</div>;
}
