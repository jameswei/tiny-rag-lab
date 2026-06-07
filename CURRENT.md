# Current Task

Task:         P1-T01
Phase:        Phase 1
Spec:         docs/phases/phase-1-naive-classic-rag.md
Taskboard:    docs/phases/phase-1-taskboard.md
Owner:        claude
Status:       review
Review Result: signed_off
Reviewer:     codex
Last Updated: 2026-06-07
Updated By:   codex

## Findings From Last Review

- none

## Tests Reviewed

- `uv run rag --help`: pass, printed top-level command help
- `uv run rag index --help`: pass, printed index options
- `uv run rag retrieve --help`: pass, printed retrieve options
- `uv run rag ask --help`: pass, printed ask options
- `uv run pytest --tb=short -q`: pass, 8 passed in 0.01s
- `uv run python -c "import tiny_rag_lab.cli as c; print(c.build_parser().prog)"`: pass, printed `rag`
- `uv run rag retrieve "what is watsonx?"`: expected stub failure, raised `NotImplementedError: rag retrieve: not yet implemented (P1-T14)`

## Blocker

- none

## Notes

## Handoff

### Task Summary

Added the `rag` CLI entry point with three argparse subcommands (index,
retrieve, ask). All argument flags match the spec. Subcommand handlers are
stubs that raise NotImplementedError until their respective tasks are
implemented.

### Files Changed

- `tiny_rag_lab/cli.py`: argparse parser, three subcommand stubs, `main()` entry point
- `pyproject.toml`: added `[project.scripts]` entry `rag = "tiny_rag_lab.cli:main"`
- `tests/test_cli.py`: 7 tests covering --help exits, arg parsing defaults and overrides for all three subcommands

### Design Decisions

- `build_parser()` is a separate function from `main()` so tests can call it directly without subprocess, keeping tests fast and dependency-free.
- Chunking flags (`--chunk-size`, `--chunk-overlap`) are on `index` only, matching the spec's rule that these are index-time settings.
- `--index-dir` defaults to `.tiny-rag/index` on all commands that read or write the index.

### Tests Run

- `uv run rag --help`: exit 0
- `uv run rag index --help`: exit 0
- `uv run rag retrieve --help`: exit 0
- `uv run rag ask --help`: exit 0
- `uv run pytest --tb=short -q`: 8 passed

### Known Gaps

- Subcommand handlers raise NotImplementedError; real implementations come in T13, T14, T18.

### Questions For Next Agent

- None. T03 (corpus gitignore) and T04 (data contracts) can both proceed independently.
