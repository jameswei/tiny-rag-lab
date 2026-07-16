import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { learningMaterialUrl } from "./copy";
import { App } from "./main";

const starterRun = {
  run_id: "starter", index: { manifest: { index_backend: "numpy", chunk_count: 1 }, document_count: 1, chunk_count: 1 },
  trace: { query: "What is alpha?", latency_by_stage: { retrieval: 1 }, context_pack: { selected: ["alpha.md"], omitted: [] }, answer: "Alpha is evidence.", citations: ["alpha.md"] },
  evidence: [{ chunk_id: "alpha", doc_id: "alpha.md", title: "Alpha", path: "alpha.md", text: "alpha evidence", rank: 1, score: 0.98, score_semantics: "cosine_similarity", score_components: { dense: 0.98 }, selected_for_context: true }],
  query_vector: [0.2, 0.4, 0.8], mode: "replay",
};
const guidedLesson = { lesson: { id: "cloudflare-do-coordinator-v1", package_id: "cloudflare-state-coordination-v1", order: 1, title: "Route one entity", question: "How does a Worker route one entity?", focus: "Real evidence.", answer_provenance: "recorded_lesson_result", source_snapshot: { source_revision: "3dcb728" } }, run: { ...starterRun, mode: "saved_lesson", config: { answer_provenance: "recorded_lesson_result" }, source_snapshot: { source_revision: "3dcb728" } } };
const citationMismatchLesson = {
  id: "citation-mismatch", label: "citation_mismatch", question: "Where does the nested document live?", explanation: { en: "A citation must point to the supporting source.", zh: "引用必须指向支持该断言的来源。" },
  baseline: { config: { retriever: "dense", top_k: 3 }, trace: { evidence: [{ rank: 1, score: 0.77, doc_id: "subdir/nested.md", text: "Nested document is in a subdirectory." }], context_pack: { selected: ["nested"], omitted: [] }, answer: "The nested document lives in the root directory.", citations: ["with_h1.md"], outcome_label: "citation_mismatch" } },
  intervention: { config: { retriever: "dense", top_k: 3 }, trace: { evidence: [{ rank: 1, score: 0.77, doc_id: "subdir/nested.md", text: "Nested document is in a subdirectory." }], context_pack: { selected: ["nested"], omitted: [] }, answer: "The nested document lives in a subdirectory.", citations: ["subdir/nested.md"], outcome_label: "no_failure" } },
};
const retrievalMaterials = [
  { question_id: "lex-1", category: "lexical", question: "What does max_retries control?", gold_doc_ids: ["queues.md"], teaching_note: { en: "Watch the exact term.", zh: "观察精确词项。" } },
  { question_id: "dense-1", category: "dense", question: "Why can an update arrive later elsewhere?", gold_doc_ids: ["kv.md"], teaching_note: { en: "Compare vector direction.", zh: "比较向量方向。" } },
  { question_id: "hybrid-1", category: "hybrid", question: "How does a queue retry become delayed?", gold_doc_ids: ["queues.md"], teaching_note: { en: "Follow both ranks.", zh: "观察两种排名。" } },
  { question_id: "rerank-1", category: "reranking", question: "What does getByName return?", gold_doc_ids: ["durable.md"], teaching_note: { en: "Watch candidates move.", zh: "观察候选移动。" } },
];
const lexicalRun = {
  ...starterRun,
  explanations: { kind: "bm25", bm25: { query_tokens: ["what", "max_retries"], corpus_size: 40, k1: 1.5, b: .75, candidates: [{ chunk_id: "alpha", rank: 1, score: .98, document_length: 12, average_document_length: 20, terms: [{ term: "max_retries", query_frequency: 1, term_frequency: 2, document_frequency: 1, inverse_document_frequency: 2.3, contribution: .98 }] }] } },
};
const denseRun = {
  ...starterRun,
  explanations: { kind: "dense", dense: { candidates: [{ chunk_id: "alpha", rank: 1, score: .7057, query_norm: 1, chunk_norm: 1, dot_product: .7057, cosine_similarity: .7057, query_vector_preview: [-.0351, -.0221, -.0106, .0356, -.0287, -.0344, -.0643, -.0263, .0525, .0315, .0629, .0537], chunk_vector_preview: [.0076, -.0087, -.0522, .0459, -.0343, -.0343, -.0444, .0389, .0941, -.0022, .0181, .0905] }] } },
};
const hybridRun = {
  ...starterRun,
  explanations: { kind: "hybrid", hybrid: { rrf_k: 60, dense: [{ chunk_id: "alpha", rank: 2, score: .7 }], bm25: [{ chunk_id: "alpha", rank: 1, score: 3.2 }, { chunk_id: "beta", rank: 3, score: 2.1 }], candidates: [{ chunk_id: "alpha", rank: 1, score: .0325, sources: [{ source: "dense", rank: 2, score: .7, contribution: .0161 }, { source: "bm25", rank: 1, score: 3.2, contribution: .0164 }] }, { chunk_id: "beta", rank: 2, score: .0159, sources: [{ source: "bm25", rank: 3, score: 2.1, contribution: .0159 }] }] } },
};
const rerankRun = {
  ...hybridRun,
  evidence: [{ ...starterRun.evidence[0], rank: 1, score: 2.4 }],
  candidates: [{ ...starterRun.evidence[0], chunk_id: "alpha", rank: 1, score: .0325 }, { ...starterRun.evidence[0], chunk_id: "beta", doc_id: "beta.md", title: "Beta", rank: 2, score: .03 }],
  explanations: { ...hybridRun.explanations, kind: "reranking", reranking: { candidate_count: 20, final_top_k: 5, candidates: [{ chunk_id: "alpha", pre_rank: 2, final_rank: 1, pre_score: .0325, reranker_score: 2.4, rank_delta: 1, outcome: "moved_up" }, { chunk_id: "beta", pre_rank: 1, final_rank: null, pre_score: .03, reranker_score: -.2, rank_delta: null, outcome: "dropped" }] } },
};
const qdrantReady = {
  available: true, prepared: true, launch_command: "docker compose --profile qdrant up -d",
  source_fingerprint: "1234567890abcdef1234567890abcdef", filters: ["r2", "queues"],
  collection: { alias: "teaching", collection: "teaching__1234", point_count: 522, dimension: 384, reused: true, verified: true },
};
const qdrantComparison = {
  question: retrievalMaterials[1].question,
  numpy: [{ ...starterRun.evidence[0], payload: null }],
  qdrant: [{ ...starterRun.evidence[0], payload: { chunk_id: "alpha", source_group: "r2", source_fingerprint: qdrantReady.source_fingerprint } }],
  parity: { equivalent: true, score_tolerance: .00001, items: [{ chunk_id: "alpha", numpy_rank: 1, qdrant_rank: 1, equivalent: true }] },
  source_group: "r2",
  filtered_qdrant: [{ ...starterRun.evidence[0], payload: { chunk_id: "alpha", source_group: "r2" } }],
};
const evaluationConfigA = { retriever: "bm25", top_k: 5, reranker: "none", rerank_top_n: 20 };
const evaluationConfigB = { retriever: "dense", top_k: 5, reranker: "none", rerank_top_n: 20 };
const evaluationQuestion = { question_id: "eval-1", question: "How are queue messages delivered?", category: "lexical", gold_doc_ids: ["queues.md"], metrics: { hit: 1, reciprocal_rank: 1, context_precision: .2, context_recall: 1 }, evidence: starterRun.evidence };
const evaluationResult = {
  question_count: 16,
  bundle: { index_id: "cloudflare-state-structural-v1", source_vector_fingerprint: "1234" },
  left: { config: evaluationConfigA, metrics: { n_questions: 16, hit_rate: .75, mrr: .6, context_precision: .2, context_recall: .75 }, questions: [evaluationQuestion] },
  right: { config: evaluationConfigB, metrics: { n_questions: 16, hit_rate: .875, mrr: .72, context_precision: .24, context_recall: .875 }, questions: [evaluationQuestion] },
};

function response(body: unknown) { return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } })); }

function mockApi({ corpora = [], indexes = [], catalog = [], lessons = [], guided = [guidedLesson.lesson], qdrant = qdrantReady, indexDetail = {}, providerConfigured = false, evaluationActive = [], evaluationState, modelReady = true, rerankerReady = true, activeJob = null, modelJobState }: { corpora?: unknown[]; indexes?: unknown[]; catalog?: unknown[]; lessons?: unknown[]; guided?: unknown[]; qdrant?: unknown; indexDetail?: unknown; providerConfigured?: boolean; evaluationActive?: any[]; evaluationState?: any; modelReady?: boolean; rerankerReady?: boolean; activeJob?: any; modelJobState?: any } = {}) {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, options?: RequestInit) => {
    const path = String(input);
    if (path.endsWith("/corpora")) return response({ items: corpora });
    if (/\/corpora\/[^/]+\/questions$/.test(path)) return response({ items: catalog });
    if (path.endsWith("/indexes")) return response({ items: indexes });
    if (path.includes("/indexes/")) return response(indexDetail);
    if (path.endsWith("/backends")) return response({ items: [{ id: "numpy", available: true }, { id: "qdrant", available: true }] });
    if (path.endsWith("/models/default/status")) return response({ ready: modelReady });
    if (path.endsWith("/models/reranker/status")) return response({ ready: rerankerReady });
    if (path.endsWith("/models/default/download")) return response({ id: "embedding-model-ready", status: "complete" });
    if (path.endsWith("/models/reranker/download")) return response({ id: "reranker-model-ready", status: "complete" });
    if (path.endsWith("/jobs/active")) return response({ items: activeJob ? [activeJob] : [] });
    if (activeJob && path.endsWith(`/jobs/${activeJob.id}`)) return response(modelJobState || activeJob);
    if (path.endsWith("/provider-status")) return response({ configured: providerConfigured });
    if (path.endsWith("/provider/test")) return response({ message: "Provider connection verified" });
    if (path.endsWith("/failure-lessons")) return response({ items: lessons });
    if (path.endsWith("/lessons")) return response({ items: guided });
    if (path.endsWith("/retrieval/materials")) return response({ index_id: "cloudflare-state-structural-v1", items: retrievalMaterials });
    if (path.endsWith("/retrieval/qdrant/status")) return response(qdrant);
    if (path.endsWith("/retrieval/qdrant/prepare")) return response(qdrantReady.collection);
    if (path.endsWith("/retrieval/qdrant/compare")) return response(qdrantComparison);
    if (path.endsWith("/evaluations/status")) return response({ ready: true, reason: null, question_count: 16, presets: [{ id: "bm25-vs-dense", left: evaluationConfigA, right: evaluationConfigB }, { id: "dense-vs-hybrid", left: evaluationConfigB, right: { ...evaluationConfigB, retriever: "hybrid" } }] });
    if (path.endsWith("/jobs/active?kind=evaluation")) return response({ items: evaluationActive });
    if (path.endsWith("/evaluations")) return response({ id: "evaluation-1", status: "queued" });
    if (path.endsWith("/jobs/evaluation-1/result")) return response(evaluationResult);
    if (path.endsWith("/jobs/evaluation-1")) return response(evaluationState || { id: "evaluation-1", status: "complete", progress: { current: 16, total: 16, message: "Complete" } });
    if (evaluationActive.some((item) => path.endsWith(`/jobs/${item.id}`))) return response(evaluationState || evaluationActive.find((item) => path.endsWith(`/jobs/${item.id}`)));
    if (path.includes("/lessons/")) return response(guidedLesson);
    if (path.endsWith("/starter-run")) return response(starterRun);
    if (path.endsWith("/runs/retrieve") || path.endsWith("/runs/ask")) {
      const body = JSON.parse(String(options?.body || "{}"));
      if (body.reranker === "cross-encoder") return response(rerankRun);
      if (body.retriever === "hybrid") return response(hybridRun);
      if (body.retriever === "dense") return response(denseRun);
      return response(lexicalRun);
    }
    return response({});
  }));
}

describe("field-guide visual foundation", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); localStorage.clear(); });

  it("routes both learning-guide languages to packaged static HTML", () => {
    expect(learningMaterialUrl("en", "retrieval-mechanics.md")).toBe("/docs/en/retrieval-mechanics.html");
    expect(learningMaterialUrl("zh", "rag-failure-lab.md")).toBe("/docs/zh/rag-failure-lab.html");
  });

  it("groups Build and Inspect while preserving both internal views", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Build & Inspect" }));
    expect(await screen.findByRole("heading", { name: "Build index", level: 2 })).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: "Inspect index" }));
    expect(screen.getByRole("heading", { name: "Inspect index" })).toBeInTheDocument();
  });

  it("makes bundled and uploaded corpus sources explicit and shows ready Qdrant", async () => {
    mockApi({ corpora: [{ id: "cloudflare-state-v1", name: "Cloudflare State", kind: "catalog", file_count: 40 }] }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Build & Inspect" }));
    expect(screen.getByRole("heading", { name: "Use a bundled corpus", level: 4 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Bring a small corpus", level: 4 })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Qdrant — ready" })).toBeInTheDocument();
  });

  it("offers newcomers a replay path and a build-your-own path from Home", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    expect(Array.from(screen.getByRole("navigation", { name: "Lab areas" }).querySelectorAll("button")).map((button) => button.textContent)).toEqual(["Home", "Learn", "Retrieval", "Explore", "Build & Inspect", "Failure Lab", "Settings"]);
    await user.click(screen.getByRole("button", { name: "Build a small index" }));
    expect(await screen.findByRole("heading", { name: "Build index", level: 2 })).toBeInTheDocument();
  });

  it("switches the UI language without translating corpus data", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "中文" }));
    expect(await screen.findByRole("button", { name: "构建与检查" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始引导课程" })).toBeInTheDocument();
  });

  it("shows two independently pinned model lifecycles in Settings", async () => {
    mockApi({ modelReady: false, rerankerReady: false }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Settings" }));
    expect(await screen.findByText("sentence-transformers/all-MiniLM-L6-v2")).toBeInTheDocument();
    expect(screen.getByText("cross-encoder/ms-marco-MiniLM-L-6-v2")).toBeInTheDocument();
    expect(screen.getAllByText(/Pinned revision/)).toHaveLength(2);
    expect(screen.getByText(/Required for Dense and Hybrid retrieval/)).toBeInTheDocument();
    expect(screen.getByText(/Additionally required only when cross-encoder reranking/)).toBeInTheDocument();
    expect(screen.getByText(/No GPU, CUDA, or NVIDIA runtime is required/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download embedding model" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Download reranker model" }));
    expect(await screen.findAllByText("Ready locally · cross-encoder reranking is available")).toHaveLength(2);
    expect(vi.mocked(fetch).mock.calls.some(([path]) => String(path).endsWith("/models/reranker/download"))).toBe(true);
    await user.click(screen.getByRole("button", { name: "中文" }));
    expect(screen.getByText("固定版本的重排模型")).toBeInTheDocument();
    expect(screen.getByText("已在本地就绪 · 可以使用交叉编码器重排")).toBeInTheDocument();
  });

  it("recovers an active reranker download after refresh", async () => {
    mockApi({ rerankerReady: false, activeJob: { id: "reranker-model-1", kind: "reranker-model" }, modelJobState: { id: "reranker-model-1", kind: "reranker-model", status: "complete" } });
    const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Settings" }));
    expect(await screen.findByText("Ready locally · cross-encoder reranking is available")).toBeInTheDocument();
  });

  it("opens a complete recorded lesson with inspectable evidence", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Start guided lesson" }));
    expect(await screen.findByRole("heading", { name: "Guided lesson" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Corpus$/ })).toHaveAttribute("aria-pressed", "true");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "04Retrieve" }));
    expect(screen.getByText("alpha evidence")).toBeInTheDocument();
    expect(screen.getByText("Selected for context")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByText("Inspect raw artifact"));
    expect(screen.getByText(/context_pack/)).toBeInTheDocument();
    const guideLink = screen.getByRole("link", { name: "Read the learning guide" });
    expect(guideLink).toHaveAttribute("href", "/docs/en/retrieval-and-generation.html");
    expect(guideLink).toHaveAttribute("target", "_blank");
  });

  it("keeps the Failure Lab context selection and citations visible", async () => {
    mockApi({ lessons: [citationMismatchLesson] }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Failure Lab" }));
    expect((await screen.findAllByText("Nested document is in a subdirectory.")).length).toBe(2);
    expect(screen.getAllByText("Selected for context")).toHaveLength(2);
    expect(screen.getAllByText("Sources cited in the answer")).toHaveLength(2);
    expect(screen.getByText("with_h1.md")).toBeInTheDocument();
    expect(screen.getAllByText("subdir/nested.md").some((element) => element.closest("li"))).toBe(true);
    expect(screen.getByRole("link", { name: "Read the learning guide" })).toHaveAttribute("href", "/docs/en/rag-failure-lab.html");
  });

  it("can select an existing index and run retrieval", async () => {
    mockApi({ indexes: [{ id: "existing-index", manifest: {} }] }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Explore" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "Index" }), "existing-index");
    await user.type(screen.getByRole("textbox", { name: "Ask a question" }), "What is alpha?");
    await user.click(screen.getByRole("button", { name: "Retrieve" }));
    expect(await screen.findByText("alpha evidence")).toBeInTheDocument();
    expect(vi.mocked(fetch).mock.calls.some(([path]) => String(path).endsWith("/runs/retrieve"))).toBe(true);
  });

  it("preserves catalog and free-question behavior when reranking is not selected", async () => {
    const indexes = [{ id: "catalog-index", manifest: { source_corpus_id: "catalog-corpus" } }];
    const catalog = [{ id: "catalog-1", question: "What is the catalog answer?", featured: true }];
    mockApi({ indexes, catalog }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Explore" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "Index" }), "catalog-index");
    const catalogPicker = await screen.findByRole("combobox", { name: "Catalog question" });
    expect(screen.getByRole("combobox", { name: "Reranker" })).toHaveValue("none");
    expect(screen.getByLabelText("Ask a question")).not.toHaveAttribute("list");
    await user.selectOptions(catalogPicker, "catalog-1");
    expect(screen.getByLabelText("Ask a question")).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Retrieve" }));
    const request = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith("/runs/retrieve"));
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ catalog_question_id: "catalog-1", reranker: "none" });
  });

  it("runs a reviewed lexical lesson and renders engine-owned BM25 math", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    expect(await screen.findByRole("heading", { name: "How retrieval decides" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: retrievalMaterials[0].question })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Run live retrieval" }));
    expect(await screen.findByRole("heading", { name: "BM25 term contributions", level: 4 })).toBeInTheDocument();
    expect(screen.getAllByText("max_retries").length).toBeGreaterThan(0);
    expect(screen.getByRole("columnheader", { name: "Frequency in chunk" })).toBeInTheDocument();
    expect(screen.getByText(/score = Σ query count/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "中文" }));
    expect(screen.getByRole("columnheader", { name: "分块内出现次数" })).toBeInTheDocument();
    expect(screen.getByText(/BM25 会把每个查询词项的贡献相加/)).toBeInTheDocument();
    const request = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith("/runs/retrieve"));
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ retrieval_material_id: "lex-1", retriever: "bm25", explain: true });
  });

  it("anchors positive and negative vector components to opposite sides of one baseline", async () => {
    mockApi(); const user = userEvent.setup(); const { container } = render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /02Dense/ }));
    await user.click(screen.getByRole("button", { name: "Run live retrieval" }));
    expect(await screen.findByRole("img", { name: "Query vector" })).toBeInTheDocument();
    const charts = container.querySelectorAll<HTMLElement>(".signed-vector-chart");
    expect(charts).toHaveLength(2);
    expect(charts[0].style.getPropertyValue("--component-count")).toBe("12");
    expect(charts[0].querySelectorAll(".vector-component")).toHaveLength(12);
    expect(container.querySelectorAll(".signed-vector-chart .vector-zero-axis")).toHaveLength(2);
    expect(container.querySelectorAll(".signed-vector-chart .vector-component.negative").length).toBeGreaterThan(0);
    expect(container.querySelectorAll(".signed-vector-chart .vector-component.positive").length).toBeGreaterThan(0);
    expect(screen.getByText(/blue positive values extend upward/)).toBeInTheDocument();
    expect(screen.getAllByText("+0.0459").length).toBeGreaterThan(0);
    expect(screen.getByText("Reviewed source documents").nextSibling).toHaveTextContent("kv.md");
  });

  it("shows both source rankings and engine-owned RRF contributions", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /04Hybrid/ }));
    await user.click(screen.getByRole("button", { name: "Run live retrieval" }));
    expect(await screen.findByRole("heading", { name: "Dense ranking", level: 4 })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "BM25 ranking", level: 4 })).toBeInTheDocument();
    expect(screen.getByText(/1 \/ \(60 \+ 2\)/)).toBeInTheDocument();
    expect(screen.getByText("+0.0161")).toBeInTheDocument();
    expect(screen.getByText(/outside one source’s candidate list/)).toBeInTheDocument();
    expect(screen.getByText("Outside this source’s candidate list")).toBeInTheDocument();
    expect(screen.getByText("+0")).toBeInTheDocument();
    const request = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith("/runs/retrieve"));
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ retrieval_material_id: "hybrid-1", retriever: "hybrid", reranker: "none", explain: true });
  });

  it("keeps the complete candidate-to-final reranking audit visible", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /05Reranking/ }));
    expect(screen.getByRole("link", { name: "Read the learning guide" })).toHaveAttribute("href", "/docs/en/reranking.html");
    await user.click(screen.getByRole("button", { name: "Run live retrieval" }));
    expect(await screen.findByText("20 first-stage candidates")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Cross-encoder score" })).toBeInTheDocument();
    expect(screen.getByText("Moved up · +1")).toBeInTheDocument();
    expect(screen.getByText("Dropped")).toBeInTheDocument();
    const request = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith("/runs/retrieve"));
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ retrieval_material_id: "rerank-1", retriever: "hybrid", top_k: 5, reranker: "cross-encoder", rerank_top_n: 20, explain: true });
  });

  it("runs the fixed evaluation comparison and exposes four metrics plus per-question evidence", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /06Evaluation/ }));
    expect(screen.getByRole("link", { name: "Read the learning guide" })).toHaveAttribute("href", "/docs/en/evaluating-retrieval.html");
    expect(await screen.findByText("16 reviewed questions")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Run comparison" }));
    expect(await screen.findByRole("heading", { name: "Aggregate results" })).toBeInTheDocument();
    expect(screen.getByText("Hit rate")).toBeInTheDocument();
    expect(screen.getByText("Context precision")).toBeInTheDocument();
    expect(screen.getAllByText("How are queue messages delivered?")).toHaveLength(2);
    expect(screen.getAllByText("alpha evidence")).toHaveLength(2);
    const request = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith("/evaluations"));
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toEqual({ left: evaluationConfigA, right: evaluationConfigB });
    await user.click(screen.getByRole("button", { name: "中文" }));
    expect(screen.getByText("已完成")).toBeInTheDocument();
    expect(screen.getAllByText("词法检索")).toHaveLength(2);
    expect(screen.getAllByText("命中")).toHaveLength(2);
  });

  it("restores an active custom evaluation configuration after navigation", async () => {
    const recovered = { id: "evaluation-recovered", status: "running", progress: { current: 7, total: 16, message: "Compared 7 of 16 questions" }, left: { ...evaluationConfigB, retriever: "hybrid", top_k: 7 }, right: { retriever: "hybrid", top_k: 4, reranker: "cross-encoder", rerank_top_n: 24 } };
    mockApi({ evaluationActive: [recovered] }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /06Evaluation/ }));
    expect(await screen.findByRole("option", { name: "Recovered custom comparison" })).toBeInTheDocument();
    const retrievers = screen.getAllByRole("combobox", { name: "Retriever" });
    expect(retrievers[0]).toHaveValue("hybrid");
    expect(retrievers[1]).toHaveValue("hybrid");
    expect(screen.getByText("Compared 7 of 16 questions")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Evaluation progress" })).toHaveAttribute("aria-valuenow", "7");
    expect(screen.getByRole("progressbar", { name: "Evaluation progress" })).toHaveAttribute("aria-valuemax", "16");
    expect(screen.getByText("7 of 16 questions")).toBeInTheDocument();
  });

  it("presents a terminal evaluation failure without publishing a result", async () => {
    mockApi({ evaluationState: { id: "evaluation-1", status: "failed", error: "Evaluation failed safely.", progress: { current: 3, total: 16, message: "Stopped" } } });
    const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /06Evaluation/ }));
    await user.click(await screen.findByRole("button", { name: "Run comparison" }));
    expect(await screen.findByText("Evaluation failed safely.")).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Aggregate results" })).not.toBeInTheDocument();
  });

  it("uses one reranker configuration for Explore Retrieve and Live ask", async () => {
    mockApi({ indexes: [{ id: "course", manifest: {} }], providerConfigured: true });
    const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Test connection" }));
    await user.click(screen.getByRole("button", { name: "Explore" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "Index" }), "course");
    await user.selectOptions(screen.getByRole("combobox", { name: "Reranker" }), "cross-encoder");
    const questionInput = screen.getByLabelText("Ask a question");
    expect(questionInput).toHaveAttribute("list", "reviewed-reranking-questions");
    expect(document.querySelector('datalist option[value="What does getByName return?"]')).toBeInTheDocument();
    expect(screen.getByText(/Choose a reviewed reranking example from the suggestions/)).toBeInTheDocument();
    await user.type(questionInput, "What does getByName return?");
    await user.clear(screen.getByRole("spinbutton", { name: "Candidate depth" }));
    await user.type(screen.getByRole("spinbutton", { name: "Candidate depth" }), "20");
    await user.click(screen.getByRole("button", { name: "Retrieve" }));
    await user.click(screen.getByRole("button", { name: "Live ask" }));
    expect(await screen.findByRole("heading", { name: "Retrieval path used for generation" })).toBeInTheDocument();
    expect(screen.getByText(/Only the selected evidence is packed into the generation prompt/)).toBeInTheDocument();
    const requests = vi.mocked(fetch).mock.calls.filter(([path]) => /\/runs\/(retrieve|ask)$/.test(String(path)));
    expect(requests).toHaveLength(2);
    for (const request of requests) expect(JSON.parse(String((request[1] as RequestInit).body))).toMatchObject({ reranker: "cross-encoder", rerank_top_n: 20, explain: true });
    await user.click(screen.getByRole("button", { name: "04Retrieve" }));
    expect(await screen.findByText("20 first-stage candidates")).toBeInTheDocument();
  });

  it("locks Retrieval module identity while a live run is pending", async () => {
    mockApi();
    const normalFetch = vi.mocked(fetch).getMockImplementation()!;
    let finishRun!: (value: Response) => void;
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, options?: RequestInit) => {
      if (String(input).endsWith("/runs/retrieve")) return new Promise<Response>((resolve) => { finishRun = resolve; });
      return normalFetch(input, options);
    });
    const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: "Run live retrieval" }));
    const dense = screen.getByRole("button", { name: /02Dense/ });
    expect(dense).toBeDisabled();
    await user.click(dense);
    finishRun(new Response(JSON.stringify(lexicalRun), { status: 200, headers: { "Content-Type": "application/json" } }));
    expect(await screen.findByRole("heading", { name: "BM25 term contributions", level: 4 })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /01Lexical/ })).toHaveClass("active");
  });

  it("keeps Qdrant optional and shows the exact local launch command", async () => {
    mockApi({ qdrant: { ...qdrantReady, available: false, prepared: false, collection: null } });
    const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /03Vector database/ }));
    expect(await screen.findByRole("heading", { name: "Qdrant is not running", level: 4 })).toBeInTheDocument();
    expect(screen.getByText("docker compose --profile qdrant up -d")).toBeInTheDocument();
    expect(screen.getByText(/rest of the Retrieval course still works/)).toBeInTheDocument();
  });

  it("compares NumPy and Qdrant exact search while separating payload filtering", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Retrieval" }));
    await user.click(screen.getByRole("button", { name: /03Vector database/ }));
    expect(await screen.findByText("522 points")).toBeInTheDocument();
    expect(screen.getByText(/retrieval flow does not change/)).toBeInTheDocument();
    expect(screen.getByText("Vector points")).toBeInTheDocument();
    expect(screen.getByText("Vector dimension")).toBeInTheDocument();
    await user.selectOptions(screen.getByRole("combobox", { name: "Payload filter" }), "r2");
    await user.click(screen.getByRole("button", { name: "Compare exact search" }));
    expect(await screen.findByText("Exact-search parity verified")).toBeInTheDocument();
    expect(screen.getByText(/NumPy has no database payload object/)).toBeInTheDocument();
    expect(screen.getByText(/Qdrant stores the same learner-facing metadata/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Separate filter demonstration: r2", level: 4 })).toBeInTheDocument();
    expect(screen.getByText(/Filtering changes the candidate set/)).toBeInTheDocument();
    expect(screen.getAllByText("alpha").length).toBeGreaterThan(1);
    await user.click(screen.getByRole("button", { name: "Verify prepared collection again" }));
    expect(vi.mocked(fetch).mock.calls.filter(([path]) => String(path).endsWith("/retrieval/qdrant/prepare"))).toHaveLength(1);
    const request = vi.mocked(fetch).mock.calls.find(([path]) => String(path).endsWith("/retrieval/qdrant/compare"));
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ retrieval_material_id: "dense-1", top_k: 5, source_group: "r2" });
  });

  it("keeps a legacy Qdrant index searchable while visibly disabling payload filters", async () => {
    const indexes = [{ id: "legacy-qdrant", manifest: { index_backend: "qdrant" } }];
    mockApi({ indexes, indexDetail: { id: "legacy-qdrant", manifest: indexes[0].manifest, document_count: 1, chunk_count: 1, chunks: [], capabilities: { payload_filters: false } } });
    const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Build & Inspect" }));
    await user.click(screen.getByRole("tab", { name: "Inspect index" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "Inspect index" }), "legacy-qdrant");
    expect(await screen.findByText("Payload filters")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.getByText(/older Qdrant index remains searchable/)).toBeInTheDocument();
  });

  it("uses the static motion path when the user prefers reduced motion", async () => {
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }));
    mockApi(); render(<App />);
    expect(document.querySelector("main")).toHaveAttribute("data-motion", "reduced");
  });
});
