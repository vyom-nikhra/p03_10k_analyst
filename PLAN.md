# P03 Financial 10-K Analyst: Execution Plan

## Context
This is the course group project for UG3/UG4 GenAI, Modules 1–4. The goal is a **single ReAct agent** that answers analyst questions over 10-K passages from `virattt/financial-qa-10K`. It has two tools:
- `search_filings(query, ticker)`: hybrid RAG with a ticker filter.
- `run_python(code)`: a sandboxed program-of-thought calculator, so numbers are computed rather than guessed.

**Deliverable:** one Jupyter notebook built on `GenAI/Group Projects/Project_Notebook_Template.ipynb` (sections 0–10). It must:
- be submitted with all outputs saved;
- finish Restart & Run All in under 60 minutes;
- have a `QUICK_RUN` flag;
- support a 10-minute talk plus 5 minutes of Q&A.

**Decisions so far (Vyom):**
- **Target machine:** RTX 4070 8GB, 32GB RAM. It is not the laptop we are planning on, so everything must move cleanly to other laptops and to Colab T4.
- **LLM:** the best allowed model according to the research, with no bake-off → `qwen3:4b-instruct` (see D7).
- **Calculation questions:** all real ones, plus a derived set that teammates hand-verify (see §3).
- **Timeline:** not fixed yet. The plan is phased, with the minimum viable version first, so it can be fitted to the deadline later.
- **Team:** Vyom builds; teammates present.

**Marks that shape the design:**
- **Implementation (22):** each component must be built, **measured on its own and inside the whole system**.
- **Evaluation (12):** one table with a baseline, the full system and the ablations, all on the same fixed subset.
- **Design depth (8):** each choice justified against a simpler alternative, going beyond the lab code.
- **Other notebook marks:** Problem & data 8; Error analysis & ethics 5; Reproducibility 5.
- **Talk:** Presentation 25 (the live demo must show one success and one honest failure); Q&A 15. Every member must answer questions on their own part.
- **Bonus +5:** the self-consistency stretch goal.

**Penalties:** dataset not loaded from Hugging Face inside the notebook (−10); no outputs saved or notebook fails to run (−5 to −10).

**Working rules (Vyom):**
- **No deleting without consent.** Never delete a file without Vyom's explicit consent. Editing files, and removing scratch files Claude created itself, are allowed.
- **Git:**
  - Commit as `Vyom Nikhra`: repo-local `git config user.name "Vyom Nikhra"` and `user.email vyomnikhra@gmail.com`.
  - **No `Co-Authored-By` or Claude/Anthropic trailers** anywhere: commits, code, notebook, README.
  - Small, focused commits; **ask before any push**.
- **Code style:** read code before changing it; raise errors explicitly, with no silent recovery or unrequested fallbacks.
  - Applied here: tool errors become agent observations only because the brief requires "tool-error handling", and they are logged and counted.
- **Comments and explanations:** sparse comments that explain non-obvious reasons, matching the surrounding style; no tutorial comments or extra abstractions. Explain a concept briefly when it first comes up.
- **Long jobs:** evaluation runs are resumable (JSONL checkpoints + `PROGRESS.md`).
- **Reporting:** at the end of each phase, say what changed, what was verified and what is still uncertain.
- ⚠ **AI-use disclosure.** The course Integrity rule requires "Mark AI-assisted code". That is a disclosure, not a commit trailer, so §10 gets a plain one-line statement worded by Vyom. Vyom decides the exact wording; dropping it risks the integrity rule.

---

## 1. Research: what exists and what we adopt
| Work | Finding | What we adopt |
|---|---|---|
| **FinQA** (Chen 2021) | FinQANet scores 61.2% execution accuracy; human experts score 91.2. Giving the model the gold evidence lifts it to 70.0, so about 9 points are lost to retrieval. | Retrieve → generate a program → execute it. |
| **TAT-QA** (Zhu 2021) | Answers carry a scale (thousand / million / percent), and getting the scale wrong is a major error source. | A numeric parser that normalises units and scale. |
| **Program-of-Thoughts** (Chen 2022) | Few-shot Codex: CoT → PoT goes 40.4 → 64.5 on FinQA, 45.6 → 64.6 on ConvFinQA and 61.4 → 69.0 on TAT-QA. Adding self-consistency adds another 1–4 points. On TAT-QA, **47% of the remaining errors were picking the wrong values**, not doing the arithmetic wrong. | The `run_python` tool. "Wrong value extracted" becomes an error-analysis category. |
| **PAL** (Gao 2022) | Getting the interpreter to do the arithmetic gives +15 points on GSM8K over chain-of-thought with PaLM-540B. | Same idea as PoT. |
| **Self-consistency** (Wang 2022) | +17.9 on GSM8K with 40 samples. Gains level off by about 5–10 samples, and cost grows linearly with the number of samples. | Stretch goal: 3 runs on calculation questions only, with the gain reported against the cost. |
| **ReAct** (Yao 2022) | Prompted ReAct *alone* scored below CoT on HotpotQA, and its failures were 47% reasoning loops and 23% unhelpful searches. | A step limit, a repeated-search check, and the honest comparison against RAG + CoT. |
| **Phogat 2024** | Phi-3-mini (3.8B) scores 57.6 on FinQA zero-shot but 42.5 few-shot: few-shot examples can *hurt* small models. | Keep the worked example in the prompt short, and check it on the dev set. |
| Sub-2B tool-calling study (arXiv 2609.07370) | Raw JSON from small models rarely parses; malformed tool calls are normal. | Native Ollama tool calling, plus nudge-and-retry and schema validation. |
| **FinanceBench** (Islam 2023) | GPT-4-Turbo with one vector store shared across filings gets **19%**; with one store per filing it gets **50%**. | Strong evidence for the ticker filter and the filter-on/off ablation. |
| Hybrid retrieval: BM25 vs DPR (arXiv 2109.08535); Anthropic contextual retrieval; SMARTFinRAG | BM25 beats dense on rare entities (72.0 vs 49.7 top-20). BM25 + dense + re-ranker cut the failure rate from 5.7% → 1.9%. But on 10-Ks, hybrid was *not* always better (SMARTFinRAG). | Hybrid + RRF + re-ranking, ablated rather than assumed. |
| **PoT `finqa_equal`** | Relative tolerance 1e-3; accepts the reference ×100 or ÷100; rounds to the shorter precision before comparing. | The template for our numeric-match metric. |
| **LLM-as-judge** (Zheng 2023) | Over 80% agreement with humans for GPT-4, with position, verbosity and self-preference biases. Small-model judges are unverified. | A judge from a different model family, checked against human labels; numbers checked by rule first. |
| Prior retrieval results on this dataset | bge-small (base) Recall@5 is about 0.85 without a filter on about 4k passages (self-reported, vivekkopthsd). bge-large Recall@10 is 0.823 across all 7k passages (BGE tutorial). | Our expected baseline. The ticker filter should beat it. |

**Dataset facts** (from the dataset card and the stats API; re-check locally in P1):
- **Size:** 7,000 rows; **69 tickers** with 100 rows each, except JNJ with 200, possibly a duplicated batch.
- **`filing` column:** the value is always `2023_10K`, so it is useless as a filter.
- **Contexts:** median 177 characters, maximum 5,585. Some are empty.
- **Answers:** full sentences, median 115 characters, so exact match won't work as a metric.
- **Question types:** mostly **text lookup** or **numeric lookup** (the figure is stated in the passage). Only about **1–2% need real arithmetic**.
- **Wording:** many questions say "the Company" without naming it.
- **Duplicates:** some exact or near-duplicate contexts.
- **Noisy gold:** a few gold answers contradict their passage.

---

## 2. Design decisions (each one justified against a simpler alternative)
| # | Decision | Our choice | Compared with | Evidence in the notebook |
|---|---|---|---|---|
| D1 | Knowledge-base unit | One **deduplicated `context`** = one document, with `doc_id` and `ticker` metadata. Drop empty contexts. Only split a context if it exceeds 512 tokens. | The lab's `RecursiveCharacterTextSplitter` (300 characters) | Length histogram. Splitting a passage would also break the gold match that Recall needs. |
| D2 | Retriever | BM25 + dense, fused with RRF (k=60), then a cross-encoder re-ranks the top 20 down to k | Dense only; BM25 only | Recall@{1,3,5,10} and MRR grid, with and without the re-ranker. A component stays only if it helps. |
| D3 | Embedder | `BAAI/bge-small-en-v1.5`, using its query instruction prefix | `all-MiniLM-L6-v2` (the lab default) | Recall@5 for both |
| D4 | Ticker filter | Exact pre-filter: FAISS `IDSelectorBatch` on the dense side, a score mask on BM25 | No filter | Recall, plus the end-to-end ablation A1 |
| D5 | Calculator sandbox | Separate process (`python -I`) with a 5 s timeout and an **AST allow-list**. `print`, `round` and `math` are allowed, and stdout is captured. | The Module 4 in-process `exec` with a list of banned substrings | Sandbox unit tests. The lab version hangs on `while True`, rejects the name `evaluation`, and turns `1,234` into a tuple. |
| D6 | Control flow | ReAct agent in LangGraph | Fixed pipelines: RAG → answer; RAG + CoT; RAG → single-shot program of thought | Results table by question type |
| D7 | Agent LLM | **`qwen3:4b-instruct` (Qwen3-4B-Instruct-2507)** | Justified in writing; no bake-off (Vyom's call) | Reasons listed under the table |
| D8 | Judge | **`gemma3:4b`**: a different family, so less self-preference. Binary rubric; numbers checked by rule first. | Exact match / last-number regex (the Module 2 style) | Agreement % and Cohen's κ (agreement corrected for chance) against about 60 human-labelled answers |

**D7, why `qwen3:4b-instruct`:**
- **Size and family:** the exact family and size the brief names (4.0B), so no exception is needed.
- **No thinking mode:** faster loops, and it avoids the warning in Qwen's docs about stopwords leaking out of thinking text.
- **Tool use:** the 2507 update explicitly improved it; the third-party BFCL v4 score is 35.7, against 28.4 for Qwen3-1.7B (unverified).
- **Hardware:** fits the 8GB target and the 4GB laptop.
- **Rejected alternatives:**
  - `qwen3.5:4b`: 4.21B parameters and multimodal, with no verified tool-calling advantage.
  - `phi4-mini`: different family, and its tool format in Ollama is reported to be fragile.

---

## 3. Data and the derived calculation set (notebook §3)
- **Loading and cleaning:**
  - `load_dataset("virattt/financial-qa-10K")["train"]` is the first data cell.
  - Drop rows with an empty context or answer and report how many were dropped.
  - Investigate the JNJ 200-row anomaly.
- **Exploration:** rows, unique contexts, tickers, questions per ticker and per context, length histograms, `filing` values, 3 example rows.
- **Deduplication:** normalise each context (lowercase, collapse whitespace) and assign a `doc_id`.
  - Near-duplicates (one context contains the other, or they are equal after normalisation) form a **relevant set** for each question, so duplicates don't make Recall look worse than it is.
  - Report how many near-duplicates there are.
- **Question type**, with 3 classes: `text_lookup`, `numeric_lookup` and `calculation`:
  1. **Rule A:** the gold answer contains a number that is not in the context after normalisation, so it must have been calculated → `calculation`.
  2. **Rule B:** the answer contains numbers, and all of them appear in the context → `numeric_lookup`.
  3. **Rule C:** a keyword regex on the question (percentage / ratio / change / difference / total…) flags conflicts, which an LLM tagger resolves.
  4. A teammate hand-labels 100 rows; report how well the rules agree with them.
  - The tags use the answer, so they are **for evaluation only**; the agent never sees them.
- **Derived calculation set ("10K-Calc", about 150 candidates, keeping the hand-verified ones):**
  1. A regex finds contexts with 2 or more comparable figures (the same unit, or year pairs).
  2. `qwen3:4b-instruct` with **structured output** (a Pydantic schema; Module 2 structured-generation lab) writes a question, picks an operation (difference, percent change, ratio, share of total, sum) and extracts the operands **verbatim**, labelling which one is newer and which older.
  3. An automatic check rejects any item whose operand strings do not appear in the context.
  4. **Python computes the gold answer.** The LLM never computes it.
  5. Teammate A verifies each item by hand: a `verified` column in `artifacts/derived_calc_labels.csv`.
  - The gold context is the source passage, so Recall works the same way as for real questions.
  - The set is generated by code in the notebook, then cached to `artifacts/derived_calc.jsonl`. `REBUILD_DERIVED=False` loads the cache.
  - It is clearly labelled as derived in every table.
- **Splits** (seed 42; sizes stated in the notebook):
  - **Test (real):** 100 `text_lookup` + 100 `numeric_lookup` + **all** real `calculation` questions (expected to be roughly 50–150).
  - **Test (derived):** the verified 10K-Calc items, minus 20 held back for dev.
  - **Dev:** 15 `text_lookup` + 15 `numeric_lookup` + 20 derived calculation questions. Used only for prompt tuning, so every real calculation question stays in test.
  - **QUICK_RUN:** 10 questions per type, plus 10 derived.
  - **Knowledge base:** all unique contexts from all rows. Questions are never indexed.
- **How the ticker reaches the agent (assumption):** the analyst picks a company, so its ticker is passed with the question. Ablation A1 removes both the ticker and the filter.

## 4. System (notebook §5), adapting the labs and citing each one
**5a. Retrieval** (from the Module 3 `1_RAG_Pipeline_Lab.ipynb`: `bm25_search`, `rrf_fuse`, `hybrid_retrieve`, `rerank`, `recall_at_k`, `mrr`; and from `Agentic_app/relay_ops/rag.py`)
- **Changes to the lab code:**
  - Key everything on `doc_id`; the lab keys on the parent `id`, which collides across chunks.
  - Use a BM25 tokenizer that keeps `1.5`, `2023`, `$` amounts and tickers intact; the lab's `[a-z0-9]+` breaks them.
  - FAISS `IndexFlatIP` over normalised embeddings.
  - Re-ranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - CUDA with fp16 when available, otherwise CPU.
- **Standalone evaluation:**
  - The grid is {dense, BM25, hybrid, hybrid + re-rank} × {filter on, off} × k ∈ {3, 5, 10}.
  - Report Recall and MRR, using the question text as the query.
  - Show the results as a chart.

**5b. Tools** (from `Agentic_app/relay_ops/tools.py` `run_python` and `search_knowledge_base`, and the Module 2 PAL cell)
- **`search_filings(query: str, ticker: str = "") -> str`**
  - Returns the top-k passages as `[doc_id | TICKER] text`, so they can be cited.
  - An unknown ticker returns an error string listing the 69 valid tickers.
  - Docstrings are plain text with no backticks; the lab found that backticks made the small model emit malformed calls.
- **`run_python(code: str) -> str`**
  - Keep from the lab: stripping code fences, and turning the last expression into `answer`.
  - Add: the AST allow-list, a subprocess with a timeout, removal of thousands separators inside numeric literals, rounding to 6 significant figures, and errors returned as observations.
- **Unit-test cell (asserts):**
  - The sandbox blocks `import os`, `open`, `__class__`, `while True` and huge powers.
  - It computes a percent change correctly.
  - It handles `1,234.5`.
  - The numeric parser handles `$`, `%`, `(x)` negatives and million/billion scale.

**5b (continued). Agent** (from `Agentic_app/relay_ops/graph.py` `build_react_agent` and `llm.py` `ainvoke_requiring_tool`; Module 4 Lab 1)
- **State:** `messages`, `steps` and `ticker`.
- **Nodes:**
  - `agent`: `ChatOllama(model, temperature=0, num_predict=512, seed=42).bind_tools([...])`
  - `tools`: `ToolNode(..., handle_tool_errors=True)`
  - `finalize`: the model with no tools, forced to answer at the step limit
- **Routing:**
  - Tool calls requested and steps < `MAX_STEPS` (6): go to `tools`.
  - Tool calls requested and the step limit reached: go to `finalize`.
  - No tool calls: go to the guard, then `END`.
  - `recursion_limit` is a backstop.
- **Guardrails:**
  1. The agent must search before answering (nudge and retry; Ollama ignores `tool_choice`).
  2. **Calculator guard:** if the draft answer has a number that is in no retrieved passage and `run_python` was never called, nudge once. This fixes the calculator-skip failure Module 4 Lab 1 recorded, and it uses only information available at inference time.
  3. **Repeated-search guard:** stop an identical search query from running twice (ReAct's looping failure).
- **Output contract:** `Answer: … / Source: [doc_id]`. The cited ID is parsed for the citation metric.
- **Prompt:** a system prompt stating the protocol, plus one short worked example.
- **Graph drawing:** `app.get_graph().draw_mermaid()`.

**5c. Fine-tuning:** not required by the brief; the notebook says so in one line.

## 5. Conditions, metrics, results table (notebook §4 and §6)
Every condition runs on the same fixed test sets. Results are reported **by type: text lookup, numeric lookup, real calculation and derived calculation**, plus overall.

| Condition | Description |
|---|---|
| B0 | Closed-book LLM (notebook §4) |
| B1 | Plain RAG: hybrid retrieval with filter, k=5, using the Module 3 `RAG_PROMPT` |
| B2 | RAG + CoT (ending with "Final answer:") |
| B3 (optional) | RAG + single-shot program of thought: shows whether the gain comes from the calculator or from the agent loop |
| **S** | **Full agent:** search + calculator, filter on, k=5 |
| A1 | Ticker filter off |
| A2 | Calculator off |
| A3 | k ∈ {3, 10}: Recall over all test questions; end-to-end on the agent if time allows |
| SC (stretch, bonus) | S × 3 at T=0.7 on calculation questions, majority vote on the normalised number; accuracy gain vs ×LLM calls and latency |
| Optional | Calculator-guard on/off; `qwen3:1.7b` size ablation |

**Metrics**
- **Retrieval on its own:** Recall@k and MRR.
- **In-system recall:** the gold document appeared in any `search_filings` result during the agent's run.
- **Judge accuracy:** from `gemma3:4b`, checked against human labels.
- **Numeric match (numeric and calculation questions):** adapted from PoT `finqa_equal`, with 1% relative tolerance, ×100/÷100 accepted, and scale normalisation.
- **Calculator-call rate:** for calculation questions, read from `AIMessage.tool_calls`. Also report unnecessary calls on lookup questions.
- **Citation accuracy:** the cited document is in the gold relevant set.
- **Cost:** seconds per question, LLM calls and tokens (Ollama `usage_metadata`).
- **Failure flags:** step limit hit, tool errors, malformed calls.

**One results table:** one row per condition, with these columns, each split by question type where it applies:
- Recall@5
- judge accuracy
- numeric accuracy
- calculator-call %
- citation accuracy
- seconds per question
- LLM calls per question

**Charts:**
- accuracy by condition × question type
- Recall@k curves
- self-consistency gain vs cost

## 6. Evaluation harness: runtime and resumability
- **`run_condition(name, fn, questions)`** appends one JSONL record per question to `artifacts/preds/{condition}.jsonl`.
  - Records hold the answer, trace, tool calls, latency and tokens.
  - It **skips IDs already done**, so a crashed run resumes where it stopped.
- **Judging is a separate pass** after generation, so the agent model and the judge model are not swapped in and out of VRAM on every question. Verdicts are cached in `artifacts/judge/`.
- **Flags:**
  - `QUICK_RUN=True`: everything runs live on about 40 questions.
  - `USE_CACHED_PREDS=True`: loads the full-run predictions and **re-scores them live**, which keeps Restart & Run All under 60 minutes.
  - The notebook states which numbers come from cached runs.
  - ⚠ Confirm with the instructor that cached predictions count as "saved artefacts".
- **Budget:** measure seconds per question on the dev set first. Rough estimate: about 400 test questions × 6 conditions × 3–6 s ≈ 2–4 hours for a full run, done once on the RTX 4070 and cached.

## 7. Portability: runs on any laptop or on Colab
- **Repo:** `D:\genai_project\p03_10k_analyst\` is its own git repo; the course material in `GenAI/` stays out of it. It contains:
  - `P03_Financial_10K_Analyst.ipynb`
  - `requirements.txt` (pinned once the first install works)
  - `README.md` with setup for Windows, Linux and Colab
  - `PROGRESS.md` (breadcrumbs: phase checklist, decisions, current numbers)
  - `artifacts/`
- **Setup (notebook §2):**
  - pinned `%pip install`
  - `OLLAMA_HOST` read from the environment, defaulting to `http://localhost:11434`
  - an environment check cell: Ollama reachable and its version, the required models pulled, GPU, Python version
  - on Colab only (`IN_COLAB`): install Ollama with curl, start `ollama serve` in the background, then `ollama pull qwen3:4b-instruct gemma3:4b`
- **No machine-specific code:** `pathlib` paths relative to the notebook; the device is detected automatically; no shell-specific cell magics outside the Colab branch.
- **Versions:**
  - Python 3.11 or 3.12, matching Colab. This machine has 3.13.5, so use a 3.12 venv if any wheel fails.
  - **Update Ollama:** this machine has 0.14.3.
- **Model sizes:** `qwen3:4b-instruct` (about 2.5GB) and `gemma3:4b` (about 3.3GB) fit together in 8GB. On the 4GB laptop they swap, which is fine because judging is a separate pass.

## 8. Notebook map (template order; each section cites the lab it reuses)
| § | Contents |
|---|---|
| 0 | Title, team and an honest contribution table |
| 1 | Problem, research question and success targets |
| 2 | Setup |
| 3 | Data, tagging, the derived set and splits |
| 4 | B0 closed-book |
| 5 | 5a retrieval, 5b tools + agent (graph drawn), 5c fine-tuning N/A |
| 6 | Evaluation: all conditions, the one table, charts |
| 7 | Error analysis: 20 or more failures in categories |
| 8 | Demo |
| 9 | Limitations & ethics |
| 10 | References, including a note on AI-assisted code |

- **Research question for §1:** "Does a ≤4B model that delegates retrieval (hybrid RAG + ticker filter) and arithmetic (a sandboxed Python tool) through a ReAct loop beat plain RAG and RAG + CoT on *calculation* questions without losing ground on *lookup* questions, and at what cost?"
- **Error categories for §7:**
  - retrieval miss
  - wrong value extracted (unit, scale or year)
  - wrong formula
  - calculator not called
  - malformed tool call or tool error
  - step limit or loop
  - gold answer wrong
  - judge error
- **Demo (§8):** `ask(question, ticker=None)` prints each tool call with its arguments and observation, the answer and the cited passage. One success and one failure are chosen in advance.
- **§9 Limitations & ethics:**
  - not financial advice
  - synthetic Q&A, and noisy gold answers
  - mostly one fiscal year, mostly large US companies
  - risks from the sandbox and from prompt injection through retrieved passages
  - over-trust in confident numbers

## 9. Phases (minimum viable version first; fit to the deadline later)
| Phase | Work | MVP or stretch |
|---|---|---|
| P0 | Repo (`git init`, local author config `Vyom Nikhra`), venv, pinned requirements, Ollama update and model pulls, a tool-calling smoke test of `qwen3:4b-instruct`, a judge smoke test of `gemma3:4b` | MVP |
| P1 | §0–3: load, explore, clean, deduplicate, tag 3 types, build the derived 10K-Calc candidates, splits | MVP |
| P1b | *Teammates, in parallel with P2–P4:* verify the derived items, hand-label 100 type tags | MVP |
| P2 | §5a retrieval + standalone Recall/MRR grid | MVP |
| P3 | §5b tools + sandbox and parser unit tests | MVP |
| P4 | §5b agent graph + guardrails; prompt iteration on the dev set only | MVP |
| P5 | Harness + B0, B1, B2, S, A1, A2 on the test sets → results table | MVP (brief minimum: 1 baseline, S, 1 ablation) |
| P6 | §7 error analysis, §8 demo, §9, §10; teammates label about 60 answers for judge validation | MVP |
| P7 | Self-consistency (bonus), B3, k sweep end-to-end, guard and model-size ablations | Stretch |
| P8 | Freeze: Restart & Run All timed on the target laptop, plus one Colab T4 run; outputs saved; slides; rehearsal | MVP |

## 10. Team (Vyom builds), with pushback
Q&A gives 7 marks for individual contribution, plus a ±10% adjustment per student, judged against the contribution table. Presenting work you didn't touch will lose those marks, and the integrity rule forbids inflating the table. So each teammate owns a real, checkable piece of work:

| Teammate | Owns | Presents |
|---|---|---|
| A | Verifies the derived calculation set; hand-labels 100 type tags | Data and design slides |
| B | Labels about 60 answers for judge validation; reads the results table | Results and ablation slides |
| C | Categorises the failures; writes §9 | Error analysis and limitations slides |

Vyom covers retrieval, tools, the agent and the demo. Each teammate also runs `QUICK_RUN` on their own laptop, which doubles as the portability test.

## 11. Risks
| Risk | Mitigation |
|---|---|
| Few real calculation questions | Report the count honestly; the derived set covers the gap. |
| Noisy gold answers | A "gold answer wrong" error category; spot-checks. |
| The model skips the calculator | The calculator guard; measured through the calculator-call rate. |
| Ollama quirks: `tool_choice` ignored, malformed calls dropped | Nudge and retry; plain-text docstrings. |
| Small-model judge is unreliable | Validation against human labels, κ reported; numbers checked by rule first. |
| Runtime over 60 minutes | `QUICK_RUN` + cached predictions. |
| Library version drift | Pinned requirements + the environment check cell. |

**To check in P0, not assumed:**
- whether `ChatOllama` needs or rejects `reasoning=False` for a model with no thinking mode
- that the Ollama tags `qwen3:4b-instruct` and `gemma3:4b` exist on the updated Ollama

## 12. Verification (before anything is called done)
- The tool and parser unit-test cell passes.
- The Recall/MRR grid is produced.
- An agent smoke test on 5 dev questions shows `run_python` in the traces for calculation questions.
- Restart & Run All with `QUICK_RUN` is timed on this laptop (RTX 3050 4GB), on the target laptop (RTX 4070) and once on Colab T4.
- The full evaluation runs with caching and the table is filled in.
- The rubric checklist is ticked against the notebook.
- `PROGRESS.md` is updated and committed after each phase; nothing is pushed without asking.
