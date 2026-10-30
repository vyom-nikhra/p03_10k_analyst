# P03 Financial 10-K Analyst: Progress Log

Read this first when resuming, then the tail of `logbook.md` for the latest atomic changes. The full design is in `PLAN.md`.

Every change to the project gets a timestamped line in `logbook.md` as soon as it is made.

## Status
| Phase | Work | Status |
|---|---|---|
| Plan | Research, design, plan approved | done (2026-10-28) |
| P0 | Environment: venv, pinned requirements, Ollama update, model pulls, smoke tests | done (2026-10-28): `python check_setup.py` passes every check on the RTX 3050 laptop |
| P1 | Notebook §0–3: data, tagging, derived calculation set, splits | done: derived v3 has 169 of 250 items passing the automatic checks; splits dev 50 / test 397 / quick 40 |
| P1b | Verify derived calculation items, label 100 type tags | done (2026-10-31), model-assisted under Kumarran's supervision (notebook §3.5 says so): real calc 42 of 48 kept, derived 87 of 169 kept; rule vs reviewer type tags 90% agreement, κ 0.85 |
| P2 | §5a retrieval + Recall/MRR grid | done: hybrid+rerank with filter R@1 0.870, R@10 0.988, MRR 0.920 (test, n=423) |
| P3 | §5b tools + unit tests | done: 14 sandbox asserts pass; search tool checked |
| P4 | §5b agent graph + guardrails | done: agent **v3** adopted on dev evidence (20 derived dev questions: numerically correct 11 → 13, calculator used 12 → 14, +0.34 LLM calls per question; lookups unchanged). v3 adds a question-keyword calculator nudge and lets the unsupported-number nudge follow it |
| P5 | Evaluation harness + baselines/ablations → results table | **done (2026-10-31), verified test set n=318:** judge acc B0 0.126, B1 0.811, B2 0.849, **S 0.852**, A1 0.654, A2 0.855. S = B2 overall (+0.003, CI [−0.041, 0.047]); S better on lookups (0.890 vs 0.835), worse on calculations (numeric 0.805 vs 0.898, p = 0.03); calculator used 66%, A2 gap not significant. 95% CIs and McNemar tests in §6; findings in §6.3 |
| P6 | §7–10: error analysis, demo, ethics, references | done: 47 S failures categorised (§7 summary: 8 ambiguous question/gold, 6 judge or metric errors, 12 calculator-path, 13 reading, 6 search, 2 refusals); §9 updated. §10 AI-assistance line still a placeholder |
| P7 | Stretch: self-consistency, B3, extra ablations | SC3 on the verified calc items (n=118): judge 0.788 → 0.822, numeric 0.805 → 0.839 at 2.9× LLM calls. B3 and extra ablations not started |
| P8 | Freeze: timed Restart & Run All, outputs saved, slides | first executed notebook committed: quick mode 21 min on the RTX 3050 (SC3 live), 5 min with everything cached. Still to do: timed run on the RTX 4070, Colab check, slides |

## Decisions
- **Agent LLM:** `qwen3:4b-instruct` (Qwen3-4B-Instruct-2507) via Ollama; no bake-off.
- **Judge:** `phi4-mini`, a different family from the agent, checked against human labels. Changed from `gemma3:4b` on 2026-10-28: on a 20-case probe gemma scored 15–17/20 and accepted wrong directions and counts; phi4-mini scored 20/20.
- **Calculation questions:** all real ones (about 1–2% of the data), plus a hand-verified derived set, labelled as derived everywhere.
- **Hardware:** target laptop is an RTX 4070 8GB with 32GB RAM; code must be portable (any laptop, Colab T4).
- **Git:** commits authored by Vyom Nikhra, no co-author trailers; ask before pushing.

## Open items
- [ ] Timeline: fit the phases to the deadline.
- [x] Which machine runs P0: the planning laptop (Vyom, 2026-10-28). Large runs go to the RTX 4070 laptop on request.
- [x] Full test-set run: on the RTX 4070, run by Vyom, preferably after the teammates' derived/real calculation verification (Vyom, 2026-10-28). Step-by-step checklist in the README ("Full evaluation run").
- [x] P8: submitted notebook defaults are `QUICK_RUN=0` + `USE_CACHED_PREDS=1` (full test set re-scored from saved predictions).
- [ ] Instructor: do cached predictions count as "saved artefacts" for the 60-minute rule?
- [ ] Vyom: wording of the AI-assistance disclosure in §10 (required by the course Integrity rule).
- [x] Checked in P0: `qwen3:4b-instruct` and `phi4-mini` pull and run on Ollama 0.40.11; `ChatOllama(reasoning=False)` is accepted by qwen3:4b-instruct.
- [x] Judge reliability: combined verdict vs reviewer labels 93.3% agreement, κ 0.73 (LLM judge alone 88.9%, κ 0.50), 45 answers (§6.1).
- [ ] Test-set hygiene: the 40 quick-subset outputs (part of test) were read while debugging the harness on 2026-10-28. Agent changes are justified from dev evidence only; the 357 non-quick test questions have never been looked at. Consider reporting them separately as the clean held-out number.
- [x] Remote: public GitHub repo https://github.com/vyom-nikhra/p03_10k_analyst (`origin`, branch `main`), created at Vyom's request on 2026-10-29. An offline copy is the bundle `D:\genai_project\p03_10k_analyst.bundle`, regenerated from `main` at the time of the push. Teammates need collaborator access to push labels.
- [ ] Colab: clone the repo in the first cell (README) so the notebook finds `artifacts/`. Not tested on Colab yet.
- [ ] Vyom's consent needed: remove the probe-only models `gemma3:4b` (3.4 GB) and `llama3.2:3b` (2.0 GB) with `ollama rm`? C: has about 23 GB free.

## Environment notes
- Planning laptop: RTX 3050 Laptop 4GB, 14GB RAM, git 2.49.
  - Latency here is pessimistic: with `num_ctx=8192`, qwen3:4b-instruct needs 4.2 GB and Ollama runs it 45% CPU / 55% GPU (`ollama ps`). On the 8 GB RTX 4070 it should fit on the GPU.
  - Python: the project venv `.venv` uses Python 3.11.9; the system default is 3.13.5.
  - Ollama: updated to 0.40.11.
  - torch 2.14.1+cu126; pinned packages in `requirements.txt`.
- RTX 4070 laptop (Kumarran): RTX 4070 Laptop 8GB, 32GB RAM. Set up on 2026-10-29: Ollama 0.40.13, both models pulled, `.venv` (Python 3.11.9, torch 2.14.1+cu126), `check_setup.py` passes. qwen3:4b-instruct at `num_ctx=8192` runs 100% on the GPU, with phi4-mini loaded alongside.
- Model storage (tool defaults): Ollama models in `C:\Users\LENOVO\.ollama\models`; Hugging Face cache in `C:\Users\LENOVO\.cache\huggingface`.
- Notebook workflow: edit `P03_Financial_10K_Analyst.py` (jupytext percent), then build and execute:
  `jupytext --to ipynb P03_Financial_10K_Analyst.py` and `jupyter nbconvert --to notebook --execute --inplace P03_Financial_10K_Analyst.ipynb`.
- The user-level `python3` Jupyter kernel on this laptop launches whatever `python` is first on PATH: activate the venv (or put `.venv/Scripts` first on PATH) before running nbconvert.
- Dev-set tuning: `python tools/dev_eval.py P03_Financial_10K_Analyst.py artifacts/dev_tuning S_agent` runs a condition on the 50 dev questions with the notebook's code. The evidence for the v1 → v3 agent decision (dev predictions and verdicts) is in `artifacts/dev_tuning/`, apart from the test artifacts.
