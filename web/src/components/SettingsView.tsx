import type { Copy } from "../copy";

export function SettingsView({ modelReady, providerUrl, providerModel, providerKey, onDownload, onProviderUrl, onProviderModel, onProviderKey, t }: {
  modelReady: boolean | null; providerUrl: string; providerModel: string; providerKey: string; onDownload: () => void; onProviderUrl: (value: string) => void; onProviderModel: (value: string) => void; onProviderKey: (value: string) => void; t: Copy;
}) {
  return <section className="view" aria-labelledby="settings-title"><span className="kicker">{t.areas.settings}</span><h2 id="settings-title">{t.areas.settings}</h2><p className="lead">{t.settingsIntro}</p>
    <section className="section-block"><h3>{t.embeddingSettings}</h3><p className="hint">{modelReady ? t.modelReady : t.modelMissing}</p>{!modelReady && <button type="button" className="primary-action" onClick={onDownload}>{t.downloadModel}</button>}</section>
    <section className="section-block"><h3>{t.providerSettings}</h3><div className="settings-fields"><label>{t.providerUrl}<input value={providerUrl} onChange={(event) => onProviderUrl(event.target.value)} placeholder="http://127.0.0.1:11434/v1" /></label><label>{t.providerModel}<input value={providerModel} onChange={(event) => onProviderModel(event.target.value)} placeholder="model-name" /></label><label>{t.providerKey}<input type="password" value={providerKey} onChange={(event) => onProviderKey(event.target.value)} /></label></div><p className="hint">{t.provider}</p></section>
  </section>;
}
