# P03 Financial 10-K Analyst: Progress Log

Read this first when resuming, then the tail of `logbook.md` for the latest atomic changes. The full design is in `PLAN.md`.

Every change to the project gets a timestamped line in `logbook.md` as soon as it is made.

## Status
| Phase | Work | Status |
|---|---|---|
| Plan | Research, design, plan approved | done (2026-10-28) |
| P0 | Environment: venv, pinned requirements, Ollama update, model pulls, smoke tests | not started |
| P1 | Notebook §0–3: data, tagging, derived calculation set, splits | not started |
| P1b | Teammates: verify derived calculation items, hand-label 100 type tags | not started |
| P2 | §5a retrieval + Recall/MRR grid | not started |
| P3 | §5b tools + unit tests | not started |
| P4 | §5b agent graph + guardrails | not started |
| P5 | Evaluation harness + baselines/ablations → results table | not started |
| P6 | §7–10: error analysis, demo, ethics, references | not started |
| P7 | Stretch: self-consistency, B3, extra ablations | not started |
| P8 | Freeze: timed Restart & Run All, outputs saved, slides | not started |

## Decisions
- **Agent LLM:** `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507) via Ollama; no bake-off.
- **Judge:** `gemma3:4b`, a different family from the agent, checked against human labels.
- **Calculation questions:** all real ones (about 1–2% of the data), plus a hand-verified derived set, labelled as derived everywhere.
- **Hardware:** target laptop is an RTX 4070 8GB with 32GB RAM; code must be portable (any laptop, Colab T4).
- **Git:** commits authored by Vyom Nikhra, no co-author trailers; ask before pushing.

## Open items
- [ ] Timeline: fit the phases to the deadline.
- [ ] Which machine runs P0 (installs, Ollama update, ~6 GB of model pulls).
- [ ] Instructor: do cached predictions count as "saved artefacts" for the 60-minute rule?
- [ ] Vyom: wording of the AI-assistance disclosure in §10 (required by the course Integrity rule).
- [ ] Check in P0: Ollama tags `qwen3:4b-instruct` and `gemma3:4b` exist; whether `ChatOllama(reasoning=...)` is accepted for a model without thinking mode.

## Environment notes
- Planning laptop: RTX 3050 Laptop 4GB, 14GB RAM, Python 3.13.5, Ollama 0.14.3 (outdated), git 2.49.
