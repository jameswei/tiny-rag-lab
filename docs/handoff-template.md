# Handoff Template

Use this template when an implementation owner moves a task to `review`.

```markdown
## Handoff

### Task Summary

Briefly describe what changed and why.

### Files Changed

- `path/to/file.py`: short purpose of change

### Design Decisions

- Decision made and reason

### Tests Run

- `command`: pass/fail/skip

### Known Gaps

- Any limitation, skipped test, missing artifact, or incomplete follow-up

### Learning Notes

- Concepts or implementation areas that deserve careful line-by-line reading

### Questions For Next Agent

- Open questions, if any
```

The reviewer should write findings in `CURRENT.md`, not in this handoff note.
When the reviewer signs off, they should update `CURRENT.md` and the taskboard
`Notes` before marking the task `done`.

