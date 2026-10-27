# P03 Financial 10-K Analyst: Progress Log

Read this first when resuming, then the tail of `logbook.md` for the latest atomic changes. The full design is in `PLAN.md`.

Every change to the project gets a timestamped line in `logbook.md` as soon as it is made.

## Status
| Phase | Work | Status |
|---|---|---|
| Plan | Research, design, plan approved | done (2026-10-28) |
| P0 | Environment: venv, pinned requirements, Ollama update, model pulls, smoke tests | done (2026-10-28): `python check_setup.py` passes every check on the RTX 3050 laptop |
| P1 | Notebook §0–3: data, tagging, derived calculation set, splits | nearly done: §0–3.7 written and run; derived v3 (operation forced) generating |
| P1b | Teammates: verify derived calculation items, hand-label 100 type tags | ready once v3 is exported: see `artifacts/labels/*_to_*.csv` and the table in notebook §3.5 |
| P2 | §5a retrieval + Recall/MRR grid | done: hybrid+rerank with filter R@1 0.870, R@10 0.988, MRR 0.920 (test, n=423) |
| P3 | §5b tools + unit tests | done: 14 sandbox asserts pass; search tool checked |
| P4 | §5b agent graph + guardrails | code written; first live run on dev and quick in progress |
| P5 | Evaluation harness + baselines/ablations → results table | harness and judge gates written; quick-mode run in progress; full test run is for the RTX 4070 |
| P6 | §7–10: error analysis, demo, ethics, references | not started |
| P7 | Stretch: self-consistency, B3, extra ablations | not started |
| P8 | Freeze: timed Restart & Run All, outputs saved, slides | not started |

## Decisions
- **Agent LLM:** `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507) via Ollama; no bake-off.
- **Judge:** `phi4-mini`, a different family from the agent, checked against human labels. Changed from `gemma3:4b` on 2026-10-28: on a 20-case probe gemma scored 15–17/20 and accepted wrong directions and counts; phi4-mini scored 20/20.
- **Calculation questions:** all real ones (about 1–2% of the data), plus a hand-verified derived set, labelled as derived everywhere.
- **Hardware:** target laptop is an RTX 4070 8GB with 32GB RAM; code must be portable (any laptop, Colab T4).
- **Git:** commits authored by Vyom Nikhra, no co-author trailers; ask before pushing.

## Open items
- [ ] Timeline: fit the phases to the deadline.
- [x] Which machine runs P0: the planning laptop (Vyom, 2026-10-28). Large runs go to the RTX 4070 laptop on request.
- [ ] Instructor: do cached predictions count as "saved artefacts" for the 60-minute rule?
- [ ] Vyom: wording of the AI-assistance disclosure in §10 (required by the course Integrity rule).
- [x] Checked in P0: `qwen3:4b-instruct` and `phi4-mini` pull and run on Ollama 0.40.11; `ChatOllama(reasoning=False)` is accepted by qwen3:4b-instruct.
- [ ] Judge reliability: phi4-mini alone was lenient on real answers (Claude's dev check on B0). Rule gates were added; the combined judge still needs checking against teammate labels (`judge_validation` export, coming in §6).
- [ ] Colab: the notebook needs `artifacts/` (derived set, labels, cached predictions). The repo must be cloned or uploaded there, and it has no remote yet; ask Vyom before pushing anywhere.
- [ ] Vyom's consent needed: remove the probe-only models `gemma3:4b` (3.4 GB) and `llama3.2:3b` (2.0 GB) with `ollama rm`? C: has about 23 GB free.

## Environment notes
- Planning laptop: RTX 3050 Laptop 4GB, 14GB RAM, git 2.49.
  - Python: the project venv `.venv` uses Python 3.11.9; the system default is 3.13.5.
  - Ollama: updated to 0.40.11.
  - torch 2.14.1+cu126; pinned packages in `requirements.txt`.
- Model storage (tool defaults): Ollama models in `C:\Users\LENOVO\.ollama\models`; Hugging Face cache in `C:\Users\LENOVO\.cache\huggingface`.
- Notebook workflow: edit `P03_Financial_10K_Analyst.py` (jupytext percent), then build and execute:
  `jupytext --to ipynb P03_Financial_10K_Analyst.py` and `jupyter nbconvert --to notebook --execute --inplace P03_Financial_10K_Analyst.ipynb`.
