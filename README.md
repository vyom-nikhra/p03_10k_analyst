# P03 Financial 10-K Analyst

A ReAct agent (LangGraph) that answers questions about company 10-K filings. It uses two tools:
- `search_filings`: hybrid BM25 + dense retrieval with a ticker filter, over `virattt/financial-qa-10K`.
- `run_python`: a sandboxed calculator, so numbers are computed rather than guessed.

The graded deliverable is the notebook `P03_Financial_10K_Analyst.ipynb`.

| File | Contents |
|---|---|
| `PLAN.md` | Design |
| `PROGRESS.md` | Current status |
| `logbook.md` | Timestamped history of every change |

## Setup (any laptop)

**Get the code** (public repo; `artifacts/` with the derived set, label files and saved predictions comes with it):
```
git clone https://github.com/vyom-nikhra/p03_10k_analyst.git
cd p03_10k_analyst
```
Without internet access, use the bundle file instead: `git clone p03_10k_analyst.bundle p03_10k_analyst`. To connect that copy to GitHub afterwards, run `git remote set-url origin https://github.com/vyom-nikhra/p03_10k_analyst.git`.

**Requirements**
- Python 3.10–3.12 (developed on 3.11)
- [Ollama](https://ollama.com) (a recent version)
- About 8 GB of disk space for models

An NVIDIA GPU is optional but recommended.

**1. Pull the models** (about 5 GB):
```
ollama pull qwen3:4b-instruct
ollama pull phi4-mini
```

**2. Create a virtual environment** in the repo folder:

Windows (PowerShell):
```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux / macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install torch first**, then the pinned requirements. Installing torch first stops the other packages from pulling in a CPU-only build.

With an NVIDIA GPU, use the CUDA build. Pick the index that matches your driver on [pytorch.org](https://pytorch.org/get-started/locally/); cu126 is the one tested here:
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

CPU only:
```
python -m pip install torch
```

Then:
```
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```
`requirements-dev.txt` holds the Jupyter kernel and nbconvert, and is only needed outside Colab.

**4. Check the machine:**
```
python check_setup.py
```
Every line should show `[ OK ]`. The first run downloads the dataset and two small Hugging Face models.

**Ollama on another host or port.** Set these environment variables before starting Jupyter:
- `OLLAMA_HOST`, e.g. `http://127.0.0.1:11434`
- `AGENT_MODEL` / `JUDGE_MODEL`, to change the models

## Colab (T4)
The notebook reads `artifacts/` from the repo, so clone it in the first Colab cell:
```
!git clone https://github.com/vyom-nikhra/p03_10k_analyst.git
%cd p03_10k_analyst
```
Then open `P03_Financial_10K_Analyst.ipynb` from the Colab file browser and run it top to bottom. The setup section detects Colab, installs Ollama with the official script, starts `ollama serve` in the background and pulls both models. Torch is already installed on Colab. This path has not been tested on Colab yet (P8).

## Running the notebook
The notebook source is `P03_Financial_10K_Analyst.py`, in jupytext "percent" format. Build the `.ipynb` from it, then execute it:
```
python -m jupytext --to ipynb P03_Financial_10K_Analyst.py
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 P03_Financial_10K_Analyst.ipynb
```

Flags are set at the top of the notebook, or with environment variables of the same name:

| Flag | Default | Effect |
|---|---|---|
| `QUICK_RUN` | `0` | `1` evaluates the 40-question quick subset live. |
| `USE_CACHED_PREDS` | `1` | With `QUICK_RUN=0`, scores the full test set from saved predictions in `artifacts/preds/`, with no live LLM runs. |
| `GENERATE_DERIVED` | `0` | Lets the LLM generate missing derived calculation questions. |

**Full evaluation run** (meant for the RTX 4070 laptop). It resumes where it stopped if interrupted.

Expected time, estimated from quick-run timings: about 9 hours on the 4 GB planning laptop, where the model runs half on the CPU. On the RTX 4070 it should be roughly 3–4 hours.

1. **Get the repo** on the RTX 4070 laptop with `git clone` or the bundle (see "Get the code" above). `artifacts/cache/` is not in the repo and is rebuilt on first use.
2. **Set up** with steps 1–4 above. `python check_setup.py` must pass.
3. **Add the verification files first** (recommended). Put the teammates' `derived_calc_verified.csv` and `real_calc_verified.csv` in `artifacts/labels/`. Rejected items are then dropped before the run, so no time is spent on them. Running without the files also works: predictions for items rejected later are simply ignored.
4. **Run it with the venv activated.** The `python3` Jupyter kernel starts whatever `python` is first on PATH.

   PowerShell:
   ```
   $env:QUICK_RUN="0"; $env:USE_CACHED_PREDS="0"
   python -m jupytext --to ipynb P03_Financial_10K_Analyst.py
   python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 P03_Financial_10K_Analyst.ipynb
   ```
5. **Commit and push the results** (`git push`; a bundle copy needs the `set-url` step above first):
   - `artifacts/preds/`
   - `artifacts/judge/`
   - `artifacts/labels/failures_to_categorize__test.csv`
   - the executed notebook

   After that, `QUICK_RUN=0` with `USE_CACHED_PREDS=1` re-scores the full test set from these files with no live model runs.

## Human labelling (teammates)
The notebook exports the files below to `artifacts/labels/`. Fill one in and save it under its "completed" name; the notebook picks it up on the next run. Exports are never overwritten.

| Export | Fill in | Save as |
|---|---|---|
| `qtype_sample_to_label.csv` | `human_label` (text_lookup / numeric_lookup / calculation / unexplained) | `qtype_human_labels.csv` |
| `real_calc_to_verify.csv` | `verified` (yes/no), `note` | `real_calc_verified.csv` |
| `derived_calc_to_verify.csv` | `verified` (yes/no), `note` | `derived_calc_verified.csv` |
| `judge_validation_to_label.csv` | `human_correct` (yes/no) | `judge_validation_labels.csv` |
| `failures_to_categorize__test.csv` | `human_category`, `what_to_change` | `failures_categorized__test.csv` |

The failure list depends on the evaluation set: `__quick` files come from a `QUICK_RUN` (only a handful of failures), `__test` files from the full test run. Categorise the `__test` file for the report.

Two exports carry an `answer_key` column. If the system changes after you label them, the notebook stops with an error instead of comparing your labels with answers you never saw.
