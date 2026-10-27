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
Open the notebook in Colab and run it top to bottom. The setup section detects Colab, installs Ollama with the official script, starts `ollama serve` in the background and pulls both models. Torch is already installed on Colab.

## Running the notebook
- `QUICK_RUN = True` (the default) evaluates a small fixed sample, so the whole notebook runs in well under an hour.
- `USE_CACHED_PREDS = True` loads the saved full-run predictions from `artifacts/preds/` and re-scores them live. These come from full evaluation runs done once and saved.
