export function RawArtifact({ label, value }: { label: string; value: unknown }) {
  const content = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return <details className="raw-artifact"><summary>{label}</summary><pre>{content}</pre></details>;
}
