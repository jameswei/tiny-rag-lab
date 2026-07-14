import type { Copy } from "../copy";

/** Keeps source-text cards compact while preserving the complete text. */
export function ExpandableText({ text, t, limit = 320 }: { text: string; t: Copy; limit?: number }) {
  if (text.length <= limit) return <p className="context-preview">{text}</p>;
  const preview = `${text.slice(0, limit).trimEnd()}…`;
  return <div className="context-preview"><p>{preview}</p><details><summary>{t.showFull}</summary><p>{text}</p></details></div>;
}
