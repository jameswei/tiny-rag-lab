import type { Copy } from "../copy";
import type { Stage } from "../types";

export function Pipeline({ activeStage, onSelect, t }: { activeStage?: Stage; onSelect?: (stage: Stage) => void; t: Copy }) {
  return <ol className="pipeline" aria-label="RAG pipeline">
    {t.stages.map((label, index) => {
      const stage = index as Stage;
      const active = activeStage === stage;
      return <li key={label} className="pipeline-item">
        <button type="button" className={`pipeline-stage${active ? " stage-active" : ""}`} aria-pressed={active} disabled={!onSelect} onClick={() => onSelect?.(stage)}>
          <span className="stage-number">{String(index + 1).padStart(2, "0")}</span>
          <span>{label}</span>
        </button>
        {index < t.stages.length - 1 && <span className="pipeline-arrow" aria-hidden="true">→</span>}
      </li>;
    })}
  </ol>;
}
