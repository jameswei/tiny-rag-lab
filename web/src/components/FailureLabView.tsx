import { learningMaterialUrl, type Copy } from "../copy";
import type { FailureLesson, FailureSide } from "../types";
import { EvidenceCard } from "./EvidenceCard";
import { RawArtifact } from "./RawArtifact";

export function FailureLabView({ lessons, lessonId, onLesson, lang, t }: { lessons: FailureLesson[]; lessonId: string; onLesson: (value: string) => void; lang: "en" | "zh"; t: Copy }) {
  const lesson = lessons.find((item) => item.id === lessonId);
  return <section className="view" aria-labelledby="failure-title"><span className="kicker">{t.areas.failure}</span><h2 id="failure-title">{t.areas.failure}</h2><p className="lead">{t.failureIntro}</p>
    <label className="lesson-picker">{t.areas.failure}<select value={lessonId} onChange={(event) => onLesson(event.target.value)}>{lessons.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
    {lesson && <article className="lesson-intro"><h3>{lesson.label}</h3><p>{lesson.question}</p><p>{lesson.explanation[lang]}</p></article>}
    {lesson && <div className="comparison"><FailureSideView title={t.baseline} side={lesson.baseline} t={t} /><FailureSideView title={t.intervention} side={lesson.intervention} t={t} /></div>}
    <a className="learning-link" href={learningMaterialUrl(lang, "rag-failure-lab.md")} target="_blank" rel="noreferrer">{t.learn}</a>
  </section>;
}

function FailureSideView({ title, side, t }: { title: string; side: FailureSide; t: Copy }) {
  return <section className="failure-side"><header><h3>{title}</h3><span>{side.trace.outcome_label}</span></header><dl className="config-list">{Object.entries(side.config).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{String(value)}</dd></div>)}</dl>
    <div className="evidence-list">{side.trace.evidence.map((item, index) => <EvidenceCard key={`${title}-${item.doc_id}-${index}`} evidence={{ ...item, selected_for_context: isSelected(item.doc_id, side.trace.context_pack.selected) }} t={t} compact />)}</div>
    <article className="answer-card"><h4>{t.answer}</h4><p>{side.trace.answer}</p></article>{side.trace.citations.length > 0 && <section className="citation-list"><h4>{t.citations}</h4><ul>{side.trace.citations.map((citation) => <li key={citation}>{citation}</li>)}</ul></section>}<RawArtifact label={t.raw} value={side.trace} />
  </section>;
}

function isSelected(docId: string, selectedIds: string[]) {
  const normalize = (value: string) => value.toLowerCase().split("/").at(-1)!.replace(/\.md$/u, "").replace(/[\s_]+/gu, "-").replace(/-+/gu, "-");
  return selectedIds.some((id) => normalize(id) === normalize(docId));
}
