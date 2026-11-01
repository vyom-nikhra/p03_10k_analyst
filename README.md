# P03 Financial 10-K Analyst

A ReAct agent that answers questions about company 10-K filings. It searches the filings (hybrid BM25 + dense retrieval with a ticker filter) and does arithmetic in a sandboxed Python tool.

The deliverable is `P03_Financial_10K_Analyst.ipynb`. Slides are in `slides/`.

## Setup

Needs Python 3.10–3.12, [Ollama](https://ollama.com) and about 8 GB of disk. An NVIDIA GPU helps but is not required.

```
git clone https://github.com/vyom-nikhra/p03_10k_analyst.git
cd p03_10k_analyst
ollama pull qwen3:4b-instruct
ollama pull phi4-mini
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
```

Install torch first so the other packages don't pull a CPU-only build. With an NVIDIA GPU:
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Without one: `python -m pip install torch`. Then:
```
python -m pip install -r requirements.txt -r requirements-dev.txt
python check_setup.py
```

## Running

The `.py` file is the source (jupytext percent format). Build and run the notebook:
```
python -m jupytext --to ipynb P03_Financial_10K_Analyst.py
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 P03_Financial_10K_Analyst.ipynb
```

By default it re-scores the full test set from the saved predictions in `artifacts/` (a few minutes; Ollama must be running for the demo). Flags, set at the top of the notebook or as environment variables:

| Flag | Default | Effect |
|---|---|---|
| `QUICK_RUN` | `0` | `1` runs a 40-question subset live |
| `USE_CACHED_PREDS` | `1` | `0` runs missing predictions live (about 5 hours on an RTX 4070) |
| `GENERATE_DERIVED` | `0` | `1` lets the LLM generate missing derived questions |

On an 8 GB GPU, start Ollama with `OLLAMA_MAX_LOADED_MODELS=1` for live runs, so the agent and judge models are not both held in GPU memory.

On Colab, clone the repo in the first cell and open the notebook; the setup cell installs Ollama and pulls the models.

## Labels

`artifacts/labels/` holds the label exports the notebook writes and the completed files it reads (question types, calculation checks, judge validation, failure categories). The completed files were filled in with AI assistance under the team's supervision.
