# Phase 3.3: Local Learning Guides Docsite

**Status:** Complete — owner preview accepted and independent implementation
sign-off recorded 2026-07-15 in the [taskboard](phase-3.3-taskboard.md).

## Goal

Turn the existing English and Simplified-Chinese learning materials into a
readable, searchable static docsite that ships with the local Studio. Links
from the visual lab should open the corresponding local guide rather than a
GitHub source page.

The docsite is a supporting learning surface. It does not add a RAG plane,
retrieval behavior, generation behavior, API, or persistence contract.

## Approved Product Decisions

- Use **Learning Guides** / **学习指南** as the user-facing label.
- Build the site with VitePress from the existing `learning_materials/en` and
  `learning_materials/zh` Markdown files; those files remain the only content
  source.
- Serve the generated site from the existing Studio container and port under
  `/docs/`; do not add a Compose service or runtime Node process.
- Keep the standard VitePress reading experience with light visual alignment
  to the visual lab. Do not create a custom shared design system.
- Use local offline search and local assets only. Do not add hosted search,
  analytics, comments, accounts, or CDN dependencies.
- Preserve article detail and phase history. Make only paired EN/ZH factual
  corrections for obsolete future tense, architecture drift, stale counts,
  and broken or misleading references.
- Release the completed phase as semantic version `0.4.0`. Update the English
  and Chinese READMEs, but leave the public landing page unchanged.

## In Scope

- A locked, self-contained VitePress build rooted in `learning_materials/`.
- Explicit English and Chinese guide routes below `/docs/`, a local search
  index, roadmap-ordered sidebars, tables of contents, and equivalent-page
  language switching.
- A `/docs/` entry that prefers the saved lab language, otherwise maps any
  browser `zh-*` locale to Chinese, and otherwise uses English. Direct language
  choices remain available, including without JavaScript.
- Same-origin local links from Learn, Explore, and Failure Lab. Links continue
  to open in a new tab so current lab state is preserved.
- Source-development support for running the React client and VitePress site
  together.
- Docker packaging shared by full and slim Studio images.
- Focused EN/ZH publication-accuracy fixes, README usage documentation,
  version alignment, automated tests, packaged smoke verification, and owner
  preview.

## Out Of Scope

- Any change to indexing, chunking, embeddings, retrieval, reranking, context
  packing, generation, judging, evaluation, traces, seed assets, or provider
  behavior.
- New or changed HTTP API endpoints or response schemas.
- New learning subjects, broad editorial restructuring, generated API docs,
  documentation versioning, a CMS, public doc hosting, or landing-page work.
- A separate docsite container, Docker registry publishing, or non-local
  multi-user deployment.

## Route And Language Contract

- `/docs/` is the entry route.
- `/docs/en/<guide>.html` and `/docs/zh/<guide>.html` are the stable packaged
  guide routes. Explicit `.html` output matches the existing FastAPI static
  server without adding rewrite behavior.
- The root language precedence is a valid `tiny-rag-lab-lang` local-storage
  value, then browser language detection, then English.
- Every page exposes an English/中文 chooser. Switching languages opens the
  same guide when the paired file exists and records the explicit preference.
- With JavaScript unavailable, the root presents direct English and Chinese
  roadmap links.
- The guide header or footer includes Back to lab, GitHub, and James Wei
  credit. Back to lab returns to `/` on the same origin.

## Build And Packaging Contract

- VitePress uses `/docs/` as its base and local search as its only search
  provider. Dead internal links fail the production build.
- The VitePress production output is copied into `/app/web-dist/docs` after
  the React build. The existing `StaticFiles` mount serves both surfaces.
- The source React development server proxies `/docs` to a fixed loopback
  VitePress development port. Packaged URLs do not depend on that port.
- Full and slim images contain the same generated docsite. The image variant
  affects only the existing embedding-model behavior.
- VitePress `1.6.4` is pinned with a Vite `6.4.3` override. The latest stable
  VitePress release still declares Vite 5 compatibility, while the patched Vite
  line resolves known dependency advisories; recheck and remove the override
  when upgrading VitePress.
- No external network request is required to render navigation, search, fonts,
  styles, code blocks, or diagrams. Generated same-origin assets remain normal
  runtime dependencies.

## Publication Accuracy Contract

The accuracy pass must retain the technical body and learning order while:

- changing statements that describe already-completed phases as future work;
- describing the roadmap as a study expansion of the current four-plane
  architecture rather than a conflicting architecture;
- removing stale exact test totals or environment claims that no longer
  describe the scoped path accurately;
- keeping English and Chinese corrections paired; and
- preserving all useful phase context, examples, commands, and explanations.

## Required Verification

- VitePress production build, dead-link validation, paired-locale coverage,
  and generated local-asset inspection.
- Web tests for local EN/ZH guide URLs, `.md` to `.html` conversion, new-tab
  behavior, and absence of GitHub blob destinations.
- React production build and full Python regression suite.
- Isolated slim-image Compose smoke covering the lab, `/docs/`, representative
  English and Chinese guides, search assets, and teardown without stale
  containers or volumes.
- Owner preview of language detection and switching, search, responsive
  reading, guide rendering, links from Learn/Explore/Failure Lab, state
  preservation, Back to lab, footer credit, and offline behavior.

## Acceptance Criteria

1. A normal Compose launch serves the visual lab and bilingual Learning Guides
   on the same loopback origin without another mandatory service.
2. Every lab learning link opens the correct local-language guide and no
   longer depends on GitHub for reading.
3. All existing learning articles remain available, searchable, readable, and
   paired across English and Chinese.
4. The docsite remains functional without internet access after the image is
   built, in both full and slim image variants.
5. No RAG-engine, API, artifact, provider, or Compose-service contract changes.
6. README usage, package versions, review records, owner preview, and release
   evidence are complete for `v0.4.0`.

## Sign-off Record

The owner accepted the decision-complete Phase 3.3 plan on 2026-07-15.
`/root/phase33_scope_review` independently reviewed the candidate spec and
taskboard on 2026-07-15 and signed off with no blocking findings. The phase was
activated in the phase index on the same date.

The owner then verified language selection, search, responsive reading,
equivalent-page switching, local lab links, return navigation, footer credit,
and the final matched header styling. `/root/phase33_t05_review` independently
signed off the full implementation and each owner-preview refinement with no
remaining findings. The isolated slim Compose preview, its volume/network, and
temporary verification images were removed after acceptance.
