import { learningMaterialUrl, type Copy } from "../copy";
import type { GuidedLesson, GuidedLessonSummary, Stage } from "../types";
import { EvidenceCard } from "./EvidenceCard";
import { CitationList } from "./CitationList";
import { Pipeline } from "./Pipeline";
import { RawArtifact } from "./RawArtifact";
import { VectorPreview } from "./BuildInspectView";

export function LearnView({ lessons, lesson, activeStage, maxStage, onLesson, onStage, onAdvance, lang, t }: {
  lessons: GuidedLessonSummary[]; lesson: GuidedLesson | null; activeStage: Stage;
  maxStage: Stage; onLesson: (id: string) => void; onStage: (stage: Stage) => void; onAdvance: (stage: Stage) => void; lang: "en" | "zh"; t: Copy;
}) {
  const run = lesson?.run; const trace = run?.trace || {};
  const selected = run?.evidence.filter((item) => item.selected_for_context !== false) || [];
  return <section className="view learn-view" aria-labelledby="learn-title">
    <span className="kicker">{t.areas.learn}</span><h2 id="learn-title">{t.lesson}</h2><p className="lead">{t.learnIntro}</p>
    <div className="lesson-rail" aria-label={t.lesson}>{lessons.map((item) => <button key={item.id} type="button" className={lesson?.lesson.id === item.id ? "active" : ""} onClick={() => onLesson(item.id)}><small>{String(item.order).padStart(2, "0")}</small>{item.title}</button>)}</div>
    {!run ? <div className="empty-run"><p>{t.loadingLesson}</p></div> : <>
      <section className="lesson-intro status-panel status-info"><span className="kicker">{t.recorded}</span><h3>{lesson?.lesson.question}</h3><p>{lesson?.lesson.focus}</p><dl className="config-list"><div><dt>{t.index}</dt><dd>{run.index.index_id}</dd></div><div><dt>{t.retriever}</dt><dd>dense · top-k 5</dd></div></dl></section>
      <Pipeline activeStage={activeStage} maxStage={maxStage} onSelect={onStage} t={t} /><aside className="stage-focus"><span>{t.whatToNotice}</span><p>{t.learnStageFocus[activeStage]}</p></aside>
      {activeStage === 0 && <><dl className="fact-grid"><div><dt>{t.documents}</dt><dd>{run.index.document_count}</dd></div><div><dt>{t.chunks}</dt><dd>{run.index.chunk_count}</dd></div><div><dt>{t.backend}</dt><dd>{String(run.index.manifest.index_backend)}</dd></div><div><dt>{t.sourceIdentity}</dt><dd>{run.source_snapshot?.source_revision?.slice(0, 8)}</dd></div></dl><RawArtifact label={t.raw} value={run.source_snapshot || run.index.manifest} /></>}
      {activeStage === 1 && <EvidenceList evidence={run.evidence} t={t} compact />}
      {activeStage === 2 && <><VectorPreview vector={run.query_vector || []} t={t} /><RawArtifact label={t.raw} value={run.query_vector || []} /></>}
      {activeStage === 3 && <EvidenceList evidence={run.evidence} t={t} />}
      {activeStage === 4 && <><EvidenceList evidence={selected} t={t} /><RawArtifact label={t.prompt} value={trace.prompt || {}} /></>}
      {activeStage === 5 && <section className="answer-stage"><article className="answer-card answer-result"><span className="kicker">{t.recorded}</span><h3>{t.answer}</h3><p>{trace.answer}</p></article><CitationList citations={trace.citations || []} evidence={run.evidence} t={t} /><a className="learning-link" href={learningMaterialUrl(lang, "retrieval-and-generation.md")} target="_blank" rel="noreferrer">{t.learn}</a><RawArtifact label={t.raw} value={trace} /></section>}
      {activeStage < 5 && <div className="action-row"><button className="primary-action" type="button" onClick={() => { const next = (activeStage + 1) as Stage; onAdvance(next); onStage(next); }}>{t.continueLesson}</button></div>}
    </>}
  </section>;
}

function EvidenceList({ evidence, t, compact = false }: { evidence: NonNullable<GuidedLesson["run"]>["evidence"]; t: Copy; compact?: boolean }) {
  return <div className="evidence-list">{evidence.map((item, index) => <EvidenceCard key={item.chunk_id || index} evidence={item} t={t} compact={compact} />)}</div>;
}
