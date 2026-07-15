# Current Task

**Phase:** No active phase.

**Status:** Phase 3.3 — Local Learning Guides Docsite is complete. The owner
accepted the bilingual docsite, search and reading experience, local Studio
links, return navigation, language-specific footer, and matched header styling.
`/root/phase33_t05_review` independently signed off the implementation and
final refinements on 2026-07-15.

## Phase 3.3 Closeout

- `UV_CACHE_DIR=/tmp/tiny-rag-uv-cache uv run pytest --tb=short -q` —
  788 passed, 8 skipped
- `npm --prefix web test -- --run` — 9 passed
- React and VitePress production builds — passed
- `npm --prefix learning_materials audit --audit-level=moderate` —
  0 vulnerabilities
- EN/ZH basename parity, `uv lock --check`, and `git diff --check` — passed
- isolated slim Compose smoke on `127.0.0.1:8013` — lab, bilingual guides,
  and packaged runtime asset passed
- source-development `/docs` proxy and hosted-asset scan — passed; both
  temporary development servers stopped
- isolated preview container, volume, network, and temporary images — removed

## Next Work

Do not start implementation until a new phase proposal is reviewed, signed off,
and activated in `docs/phases/README.md`.
