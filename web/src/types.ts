export type Lang = "en" | "zh";
export type Area = "home" | "learn" | "build" | "explore" | "failure" | "settings";
export type BuildView = "build" | "inspect";
export type Stage = 0 | 1 | 2 | 3 | 4 | 5;

export type Corpus = { id: string; name: string; kind: string; file_count: number };
export type BackendAvailability = { id: "numpy" | "qdrant"; available: boolean };
export type IndexItem = { id: string; manifest: Record<string, unknown> };
export type Evidence = {
  chunk_id?: string;
  doc_id: string;
  title?: string;
  path?: string;
  text: string;
  rank: number;
  score: number;
  score_semantics?: string;
  score_components?: Record<string, number>;
  selected_for_context?: boolean | null;
};
export type LabRun = {
  trace: Record<string, any>;
  evidence: Evidence[];
  index: { index_id?: string; manifest: Record<string, any>; document_count?: number; chunk_count?: number };
  query_vector?: number[] | null;
  run_id: string;
  error?: string | null;
  mode?: "live" | "replay" | "saved_lesson";
  config?: Record<string, any>;
  catalog_check?: { question_id: string; expected_document_ids: string[]; retrieved_document_ids: string[]; hit: boolean } | null;
  source_snapshot?: Record<string, string> | null;
};
export type GuidedLessonSummary = { id: string; package_id: string; order: number; title: string; question: string; focus: string };
export type GuidedLesson = { lesson: GuidedLessonSummary & { answer_provenance: string; source_snapshot: Record<string, string> }; run: LabRun };
export type CatalogQuestion = { id: string; question: string; featured: boolean };
export type FailureEvidence = { rank: number; score: number; doc_id: string; text: string };
export type FailureSide = {
  config: Record<string, string | number>;
  trace: {
    evidence: FailureEvidence[];
    context_pack: { selected: string[]; omitted: string[] };
    answer: string;
    citations: string[];
    outcome_label: string;
  };
};
export type FailureLesson = {
  id: string;
  label: string;
  question: string;
  explanation: { en: string; zh: string };
  baseline: FailureSide;
  intervention: FailureSide;
};
