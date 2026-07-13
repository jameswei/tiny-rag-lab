import type { ChangeEvent, CSSProperties } from "react";
import type { Copy } from "../copy";
import type { BuildView, Corpus, IndexItem } from "../types";
import { RawArtifact } from "./RawArtifact";

export function BuildInspectView({
  view, setView, corpora, indexes, corpusId, indexId, backend, modelReady, detail, onCorpus, onIndex, onBackend, onUpload, onImportWatson, onBuild, t,
}: {
  view: BuildView; setView: (view: BuildView) => void; corpora: Corpus[]; indexes: IndexItem[]; corpusId: string; indexId: string; backend: "numpy" | "qdrant"; modelReady: boolean | null; detail: any;
  onCorpus: (value: string) => void; onIndex: (value: string) => void; onBackend: (value: "numpy" | "qdrant") => void; onUpload: (event: ChangeEvent<HTMLInputElement>) => void; onImportWatson: () => void; onBuild: () => void; t: Copy;
}) {
  return <section className="view" aria-labelledby="build-title">
    <span className="kicker">{t.areas.build}</span><h2 id="build-title">{view === "build" ? t.buildViews.build : t.buildViews.inspect}</h2>
    <p className="lead">{view === "build" ? t.buildIntro : t.inspectIntro}</p>
    <div className="subnav" role="tablist" aria-label={t.areas.build}>
      <button role="tab" aria-selected={view === "build"} className={view === "build" ? "active" : ""} onClick={() => setView("build")}>{t.buildViews.build}</button>
      <button role="tab" aria-selected={view === "inspect"} className={view === "inspect" ? "active" : ""} onClick={() => setView("inspect")}>{t.buildViews.inspect}</button>
    </div>
    {view === "build" ? <div className="build-layout">
      <section className="section-block" aria-labelledby="corpus-title"><h3 id="corpus-title">{t.stages[0]}</h3>
        <div className="action-row"><label className="upload">{t.upload}<input type="file" multiple accept=".md,.txt,text/plain,text/markdown" onChange={onUpload} /></label><button type="button" onClick={onImportWatson}>{t.watson}</button></div>
        <label>{t.stages[0]}<select value={corpusId} onChange={(event) => onCorpus(event.target.value)}><option value="">—</option>{corpora.map((corpus) => <option key={corpus.id} value={corpus.id}>{corpus.name} ({corpus.file_count})</option>)}</select></label>
      </section>
      <section className="section-block" aria-labelledby="index-title"><h3 id="index-title">{t.buildViews.build}</h3>
        <label>{t.backend}<select value={backend} onChange={(event) => onBackend(event.target.value as "numpy" | "qdrant")}><option value="numpy">NumPy</option><option value="qdrant">{t.qdrantOptional}</option></select></label>
        <p className="hint">{modelReady ? t.modelReady : t.modelMissing}</p><button type="button" className="primary-action" disabled={!corpusId || !modelReady} onClick={onBuild}>{t.build}</button>
      </section>
    </div> : <div className="inspect-layout">
      <label>{t.buildViews.inspect}<select value={indexId} onChange={(event) => onIndex(event.target.value)}><option value="">—</option>{indexes.map((index) => <option key={index.id} value={index.id}>{index.id}</option>)}</select></label>
      {detail && <IndexInspection detail={detail} t={t} />}
    </div>}
  </section>;
}

function IndexInspection({ detail, t }: { detail: any; t: Copy }) {
  const manifest = detail.manifest || {};
  return <div className="index-inspection">
    <h3>{t.indexFacts}</h3>
    <dl className="fact-grid">
      <div><dt>{t.backend}</dt><dd>{String(manifest.index_backend || "numpy")}</dd></div>
      <div><dt>{t.documents}</dt><dd>{detail.document_count ?? "—"}</dd></div>
      <div><dt>{t.chunks}</dt><dd>{manifest.chunk_count ?? detail.chunks?.length ?? "—"}</dd></div>
      <div><dt>{t.embeddingModel}</dt><dd>{String(manifest.embedding_model || "—")}</dd></div>
    </dl>
    <h3>{t.chunks}</h3>
    <div className="chunk-grid">{(detail.chunks || []).slice(0, 8).map((chunk: any, index: number) => <article className="chunk-card" key={chunk.chunk_id}><span>#{String(index + 1).padStart(2, "0")}</span><strong>{chunk.doc_id}</strong><p>{chunk.text}</p><VectorPreview vector={chunk.vector || []} t={t} /></article>)}</div>
    <RawArtifact label={t.raw} value={manifest} />
  </div>;
}

export function VectorPreview({ vector, t }: { vector: number[]; t: Copy }) {
  const preview = vector.slice(0, 8);
  const maximum = Math.max(0.001, ...preview.map((value) => Math.abs(value)));
  return <div className="vector-preview"><span>{t.vectors}</span><div aria-label={t.rawVector}>{preview.map((value, index) => <i key={index} style={{ "--magnitude": `${Math.max(12, Math.round((Math.abs(value) / maximum) * 100))}%` } as CSSProperties} title={`Dimension ${index + 1}: ${value.toFixed(3)}`}><b>{value.toFixed(2)}</b></i>)}</div><small>{t.rawVector}</small></div>;
}
