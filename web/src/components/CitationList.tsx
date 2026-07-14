import type { Copy } from "../copy";

type CitationEvidence = { chunk_id?: string; doc_id: string; title?: string; path?: string };

/** Render citations as quiet, readable source references rather than raw IDs. */
export function CitationList({ citations, evidence, t, level = 3 }: { citations: string[]; evidence: CitationEvidence[]; t: Copy; level?: 3 | 4 }) {
  const Heading = level === 4 ? "h4" : "h3";
  return <section className="citation-list"><Heading>{t.citations}</Heading><ul>{citations.map((citation) => {
    const source = evidence.find((item) => item.chunk_id === citation || item.doc_id === citation || item.path === citation);
    const label = source?.title || source?.doc_id || citation;
    const path = source?.path && source.path !== label ? source.path : undefined;
    const chunkId = source?.chunk_id;
    return <li key={citation}><strong>{label}</strong>{path && <span>{path}</span>}<code>{chunkId ? `${t.chunkReference}: ${chunkId}` : `${t.citationReference}: ${citation}`}</code></li>;
  })}</ul></section>;
}
