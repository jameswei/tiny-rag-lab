# Phase 2.1 Record: Context Budgets and Structured Answers

**Status:** Complete
**Completion:** Reviewed and closed in the original phase; detailed execution
evidence was consolidated after Phase 3.0.

## Delivered

- Optional context-budget packing for ask, answer evaluation, and answer-side
  diagnosis.
- Inspectable selected/omitted evidence and estimated token use in ask traces.
- Optional JSON answer output while preserving readable plain-text output.

## Durable Decisions

- A context budget is disabled by default (`0`); enabling it is an explicit
  caller choice.
- Packing measures the exact formatted context blocks used by prompting, not an
  unrelated approximation of raw chunk text.
- Token counting is selected by available local capability, with a deterministic
  fake counter for tests and an optional real tokenizer path.

## Follow-on Context

The project does not infer a budget from a model’s advertised context window or
stream responses as part of this feature.
