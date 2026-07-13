import type { Copy } from "../copy";
import { Pipeline } from "./Pipeline";

export function HomeView({ onReplay, onBuild, t }: { onReplay: () => void; onBuild: () => void; t: Copy }) {
  return <section className="view home-view" aria-labelledby="home-title">
    <div className="home-copy">
      <span className="kicker">{t.areas.home}</span>
      <h2 id="home-title">{t.title}</h2>
      <p className="lead">{t.quick}</p>
      <div className="home-actions">
        <div className="path-action"><button className="primary-action" type="button" onClick={onReplay}>{t.replay}</button><span>{t.replayDetail}</span></div>
        <div className="path-action"><button type="button" onClick={onBuild}>{t.buildPath}</button><span>{t.buildPathDetail}</span></div>
      </div>
    </div>
    <div className="home-pipeline"><Pipeline t={t} /></div>
  </section>;
}
