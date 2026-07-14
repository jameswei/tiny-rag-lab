import type { Copy } from "../copy";
import type { Evidence } from "../types";
import { ExpandableText } from "./ExpandableText";

export function EvidenceCard({ evidence, t, compact = false }: { evidence: Evidence; t: Copy; compact?: boolean }) {
  const selected = evidence.selected_for_context;
  return <article className={`evidence-card${selected === true ? " evidence-selected" : ""}${selected === false ? " evidence-omitted" : ""}`}>
    <header>
      <span className="evidence-rank">#{evidence.rank}</span>
      <div><strong>{evidence.title || evidence.doc_id}</strong><span className="evidence-source">{evidence.path || evidence.doc_id}</span>{evidence.chunk_id && <code className="evidence-chunk-id">{t.chunkReference}: {evidence.chunk_id}</code>}</div>
      {selected !== null && selected !== undefined && <span className="evidence-state">{selected ? t.selected : t.omitted}</span>}
    </header>
    <ExpandableText text={evidence.text} t={t} />
    {!compact && <footer>
      <span><b>{t.score}:</b> {evidence.score.toFixed(4)}</span>
      {evidence.score_semantics && <span className="score-semantics">{evidence.score_semantics}</span>}
      {!!Object.keys(evidence.score_components || {}).length && <details className="score-details"><summary>{t.scoreDetails}</summary><dl>{Object.entries(evidence.score_components || {}).map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value.toFixed(4)}</dd></div>)}</dl></details>}
    </footer>}
  </article>;
}
