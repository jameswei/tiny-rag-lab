import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

function response(body: unknown) { return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } })); }

function mockApi({ corpora = [], indexes = [], lessons = [], guided = [guidedLesson.lesson] }: { corpora?: unknown[]; indexes?: unknown[]; lessons?: unknown[]; guided?: unknown[] } = {}) {
  vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
    const path = String(input);
    if (path.endsWith("/corpora")) return response({ items: corpora });
    if (path.endsWith("/indexes")) return response({ items: indexes });
    if (path.endsWith("/backends")) return response({ items: [{ id: "numpy", available: true }, { id: "qdrant", available: true }] });
    if (path.endsWith("/models/default/status")) return response({ ready: true });
    if (path.endsWith("/failure-lessons")) return response({ items: lessons });
    if (path.endsWith("/lessons")) return response({ items: guided });
    if (path.includes("/lessons/")) return response(guidedLesson);
    if (path.endsWith("/starter-run")) return response(starterRun);
    if (path.endsWith("/runs/retrieve")) return response(starterRun);
    return response({});
  }));
}

describe("field-guide visual foundation", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); localStorage.clear(); });

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
    expect(Array.from(screen.getByRole("navigation", { name: "Lab areas" }).querySelectorAll("button")).map((button) => button.textContent)).toEqual(["Home", "Learn", "Explore", "Build & Inspect", "Failure Lab", "Settings"]);
    await user.click(screen.getByRole("button", { name: "Build a small index" }));
    expect(await screen.findByRole("heading", { name: "Build index", level: 2 })).toBeInTheDocument();
  });

  it("switches the UI language without translating corpus data", async () => {
    mockApi(); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "中文" }));
    expect(await screen.findByRole("button", { name: "构建与检查" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始引导课程" })).toBeInTheDocument();
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
    expect(screen.getByRole("link", { name: "Read the learning material" })).toHaveAttribute("href", expect.stringContaining("retrieval-and-generation.md"));
  });

  it("keeps the Failure Lab context selection and citations visible", async () => {
    mockApi({ lessons: [citationMismatchLesson] }); const user = userEvent.setup(); render(<App />);
    await user.click(screen.getByRole("button", { name: "Failure Lab" }));
    expect((await screen.findAllByText("Nested document is in a subdirectory.")).length).toBe(2);
    expect(screen.getAllByText("Selected for context")).toHaveLength(2);
    expect(screen.getAllByText("Sources cited in the answer")).toHaveLength(2);
    expect(screen.getByText("with_h1.md")).toBeInTheDocument();
    expect(screen.getAllByText("subdir/nested.md").some((element) => element.closest("li"))).toBe(true);
    expect(screen.getByRole("link", { name: "Read the learning material" })).toHaveAttribute("href", expect.stringContaining("rag-failure-lab.md"));
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

  it("uses the static motion path when the user prefers reduced motion", async () => {
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() }));
    mockApi(); render(<App />);
    expect(document.querySelector("main")).toHaveAttribute("data-motion", "reduced");
  });
});
