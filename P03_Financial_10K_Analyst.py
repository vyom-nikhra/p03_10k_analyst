# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project P03: Financial 10-K Analyst
#
# Grounded answers with exact calculations.
#
# | Member | Contribution |
# |---|---|
# | Kumarran Mahesh | Problem statement, research question and targets (1); setup (2) |
# | Vyom Nikhra | Data pipeline and question set (3); retrieval (5.1–5.3) |
# | Kommana Srinivasa Rao | Calculator, search tool and agent (5.4–5.12); demo (8) |
# | Vennela Reddy | Evaluation harness and baseline (4); evaluation and results (6) |
# | Adarsh Reddy | Error analysis (7); limitations and ethics (9); references (10) |

# %% [markdown]
# ## 1. Problem Statement
#
# Analysts ask two kinds of questions about 10-K filings: lookups ("what was services revenue?") and arithmetic ("what share of revenue came from services?"). Small models misread figures in long passages and slip on arithmetic. We build one ReAct agent that finds the right passage, pulls out the numbers and computes results in a sandboxed Python tool.
#
# **Research question.** Does a 4B model that hands retrieval and arithmetic to tools beat plain RAG and RAG + chain-of-thought on calculation questions, without losing accuracy on lookups? At what cost?
#
# **Targets**, fixed before any test result:
# - retrieval Recall@5 ≥ 0.90 with the ticker filter
# - numeric accuracy on calculation questions at least 15 points above RAG + CoT
# - judge accuracy on lookups within 3 points of plain RAG
# - the calculator called on ≥ 90% of calculation questions

# %% [markdown]
# ## 2. Setup
#
# ### 2.1 Packages
#
# Only needed on Colab. On a laptop the same pinned versions come from `requirements.txt`.

# %%
# %pip install -q datasets==5.1.0 sentence-transformers==6.2.0 transformers==5.21.0 faiss-cpu==1.15.1 rank-bm25==0.2.2 scikit-learn==1.9.1 langgraph==1.2.18 langchain-core==1.7.0 langchain-ollama==1.1.0 pydantic==2.14.1 pandas==3.1.0 numpy==2.4.6 matplotlib==3.11.2


# %% [markdown]
# ### 2.2 Imports and configuration
#
# Seeds are fixed so every sample in the notebook is reproducible. Three flags control how much runs live:
# - `QUICK_RUN = 1` evaluates a 40-question subset live.
# - `USE_CACHED_PREDS = 1` (default) re-scores the full test set from saved predictions in `artifacts/`.
# - `GENERATE_DERIVED = 1` lets the LLM write missing derived questions (3.7).
#
# `CONFIG` holds the dataset revision, the models and every number the pipeline uses. Environment variables override the models and the Ollama address.

# %%
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

QUICK_RUN = os.environ.get("QUICK_RUN", "0") == "1"
USE_CACHED_PREDS = os.environ.get("USE_CACHED_PREDS", "1") == "1"
GENERATE_DERIVED = os.environ.get("GENERATE_DERIVED", "0") == "1"
IN_COLAB = "google.colab" in sys.modules

CONFIG = {
    "dataset": "virattt/financial-qa-10K",
    "dataset_revision": "30e48b49ef40389cce6488b0bddef7d6f00f4c7a",
    "agent_model": os.environ.get("AGENT_MODEL", "qwen3:4b-instruct"),
    "judge_model": os.environ.get("JUDGE_MODEL", "phi4-mini"),
    "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    "embedder": "BAAI/bge-small-en-v1.5",
    "embedder_baseline": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "top_k": 5,
    "rrf_k": 60,
    "rerank_pool": 20,
    "max_steps": 6,
    "agent_num_ctx": 8192,
    "n_test_lookup_per_type": 100,
    "n_dev_lookup_per_type": 15,
    "n_dev_derived": 20,
    "n_quick_per_type": 10,
    "n_derived_candidates": 250,
}
if "://" not in CONFIG["ollama_host"]:
    CONFIG["ollama_host"] = "http://" + CONFIG["ollama_host"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ART = Path("artifacts")
for sub in ("cache", "labels", "derived", "preds", "judge"):
    (ART / sub).mkdir(parents=True, exist_ok=True)

print(f"python {sys.version.split()[0]} | torch {torch.__version__} | device {DEVICE} | colab {IN_COLAB}")
print(json.dumps(CONFIG, indent=1))


# %% [markdown]
# ### 2.3 Ollama
#
# The agent and judge run locally through Ollama. On Colab this cell installs Ollama, starts the server and pulls both models. Everywhere else it checks that the server is up and both models are present, and stops with a clear message if not.

# %%
def ollama_version():
    try:
        return json.load(urllib.request.urlopen(f"{CONFIG['ollama_host']}/api/version", timeout=3))["version"]
    except OSError:
        return None


if IN_COLAB:
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if ollama_version():
            break
        time.sleep(1)
    for m in (CONFIG["agent_model"], CONFIG["judge_model"]):
        subprocess.run(["ollama", "pull", m], check=True)

version = ollama_version()
if version is None:
    raise RuntimeError(f"Ollama is not reachable at {CONFIG['ollama_host']}: start the Ollama app or `ollama serve`.")
pulled = {m["name"] for m in json.load(urllib.request.urlopen(f"{CONFIG['ollama_host']}/api/tags"))["models"]}
missing = [m for m in (CONFIG["agent_model"], CONFIG["judge_model"]) if m not in pulled and f"{m}:latest" not in pulled]
if missing:
    raise RuntimeError(f"Pull these Ollama models first: {missing}")
print(f"Ollama {version} at {CONFIG['ollama_host']}; models ready: {CONFIG['agent_model']}, {CONFIG['judge_model']}")


# %% [markdown]
# ### 2.4 Helpers
#
# `get_llm` builds a chat model with fixed temperature and seed. `JsonlStore` is an append-only file keyed by id: every long LLM job writes one line per item, so a crashed run resumes where it stopped and a re-run reuses earlier results.

# %%
from langchain_ollama import ChatOllama


def get_llm(model, temperature=0.0, num_predict=512, seed=SEED, **kw):
    return ChatOllama(model=model, base_url=CONFIG["ollama_host"], temperature=temperature,
                      num_predict=num_predict, seed=seed, **kw)


class JsonlStore:

    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.rows[row["id"]] = row

    def __contains__(self, key):
        return key in self.rows

    def __getitem__(self, key):
        return self.rows[key]

    def __len__(self):
        return len(self.rows)

    def add(self, row):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows[row["id"]] = row


# %% [markdown]
# ## 3. Data
#
# ### 3.1 Load the dataset
#
# `virattt/financial-qa-10K` has 7,000 question/answer pairs from the 2023 10-Ks of 69 companies. Each row has the question, the gold answer, the passage it was written from and the ticker. The revision is pinned so reruns see the same data. Each row gets a stable id `q<row>`.

# %%
from datasets import load_dataset

ds = load_dataset(CONFIG["dataset"], revision=CONFIG["dataset_revision"])["train"]
print(ds)
raw = ds.to_pandas()
raw.insert(0, "qid", [f"q{i}" for i in range(len(raw))])
display(raw.head(3))


# %% [markdown]
# ### 3.2 Cleaning and exploration
#
# Strip whitespace, drop rows with an empty field and exact duplicates, then look at the shape of the data: rows per company, the `filing` column (one value only, so no use as a filter), and word counts. Passages are short (median about 27 words), which matters for the next step.

# %%
df = raw.copy()
for c in ["question", "answer", "context", "ticker"]:
    df[c] = df[c].str.strip()
empty = (df[["question", "answer", "context"]] == "").any(axis=1)
dupes = df.duplicated(["question", "answer", "context", "ticker"]) & ~empty
df = df[~empty & ~dupes].reset_index(drop=True)
print(f"dropped {int(empty.sum())} rows with an empty field and {int(dupes.sum())} exact duplicate rows -> {len(df)} rows")
print(f"tickers: {df.ticker.nunique()}; rows per ticker: {dict(Counter(df.ticker.value_counts()))}")
print(f"filing values: {sorted(df.filing.unique())} (a single value, so useless as a filter)")
print(f"duplicate questions within a ticker: {int(df.duplicated(['question', 'ticker']).sum())}")

words = df[["question", "answer", "context"]].apply(lambda s: s.str.split().str.len())
display(words.describe(percentiles=[.5, .95, .99]).round(1).T)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.2))
words.context.drop_duplicates().plot.hist(bins=60, ax=axes[0], log=True, title="context length (words, log count)")
words.answer.plot.hist(bins=60, ax=axes[1], title="answer length (words)")
df.ticker.value_counts().plot.hist(bins=20, ax=axes[2], title="questions per ticker")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 3.3 Knowledge base
#
# Each unique (ticker, passage) becomes one document with an id `D00000`. We don't chunk: passages are already short, and chunking would break the link between a question and its gold passage that Recall@k depends on.
#
# Some passages are contained in another passage from the same company. Both then count as relevant for that question, so retrieving either is a hit. The last line checks how many passages exceed the embedder's 512-token window.

# %%
from transformers import AutoTokenizer


def norm_text(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


df["ctx_norm"] = df.context.map(norm_text)
docs = df[["ticker", "context", "ctx_norm"]].drop_duplicates(["ticker", "ctx_norm"]).reset_index(drop=True)
docs.insert(0, "doc_id", [f"D{i:05d}" for i in range(len(docs))])
doc_key = {(t, n): d for d, t, n in docs[["doc_id", "ticker", "ctx_norm"]].itertuples(index=False)}
df["gold_doc"] = [doc_key[(t, n)] for t, n in zip(df.ticker, df.ctx_norm)]

related = {d: {d} for d in docs.doc_id}
for _, g in docs.groupby("ticker"):
    items = list(zip(g.doc_id, g.ctx_norm))
    for da, a in items:
        for db, b in items:
            if da != db and a in b:
                related[da].add(db)
                related[db].add(da)
df["relevant_docs"] = df.gold_doc.map(lambda d: sorted(related[d]))

emb_tokenizer = AutoTokenizer.from_pretrained(CONFIG["embedder"])
n_tokens = docs.context.map(lambda t: len(emb_tokenizer(t, verbose=False)["input_ids"]) if len(t) > 1500 else 0)
print(f"documents: {len(docs)} unique passages across {docs.ticker.nunique()} tickers")
print(f"documents with a same-ticker near-duplicate: {sum(len(v) > 1 for v in related.values())}")
print(f"questions whose relevant set has more than one passage: {int((df.relevant_docs.str.len() > 1).sum())}")
print(f"passages longer than the embedder's 512-token window (tail truncated): {int((n_tokens > 512).sum())}")


# %% [markdown]
# ### 3.4 Number parsing
#
# 10-K figures come as `$1,234.5 million`, `(6)%`, `-34.7%` or table values in thousands. `extract_numbers` turns text into a list of numbers with sign and scale applied, skipping dates, years, section references like `10(b)` and list markers like `1)`. It also records how many decimals were shown, so a rounded answer can be matched.
#
# `same_up_to_presentation` treats two numbers as equal if they differ only by a factor of 1,000 (a table in thousands vs text in millions). `matches_target` asks whether a computed value reproduces a parsed number. The asserts at the end pin the tricky cases.

# %%
MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE_RE = re.compile(rf"\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\b{MONTHS}\.?\s+\d{{4}}"
                     rf"|\b{MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I)
NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(\()?(-|−)?\$?\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(\))?"
    r"(?!\s*\([A-Za-z]\))(?![A-Za-z0-9])\s*(%|percent\b)?\s*(thousand|million|billion|trillion|bn)?\b", re.I)
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}


def extract_numbers(text):
    out = []
    for m in NUM_RE.finditer(DATE_RE.sub(" ", text)):
        lp, minus, digits, rp, pct, scale = m.groups()
        if rp and not lp:
            continue
        shown = float(digits.replace(",", ""))
        bare = not pct and not scale and "$" not in m.group(0) and "," not in digits and "." not in digits
        if bare and 1900 <= shown <= 2099:
            continue
        if minus or (lp and rp):
            shown = -shown
        out.append({"value": shown * SCALE.get((scale or "").lower(), 1.0), "shown": shown, "pct": bool(pct),
                    "scaled": bool(scale), "scale_word": (scale or "").lower(), "dollar": "$" in m.group(0),
                    "dec": len(digits.split(".")[1]) if "." in digits else 0})
    return out


def same_up_to_presentation(a, b, rel=0.005):
    a, b = abs(a), abs(b)
    if a == 0 or b == 0:
        return a == b
    return any(math.isclose(a, b * 1000.0 ** k, rel_tol=rel) for k in range(-3, 4))


def matches_target(target, x, rel=0.01):
    t = abs(target["value"])
    if t == 0:
        return abs(x) < 1e-9
    unit = t / abs(target["shown"])
    for k in range(-3, 4):
        y = abs(x) * 1000.0 ** k
        if math.isclose(t, y, rel_tol=rel) or round(y / unit, target["dec"]) == round(abs(target["shown"]), target["dec"]):
            return True
    return False


def values(text):
    return [n["value"] for n in extract_numbers(text)]


assert values("Revenue was $1,234.5 million in 2023, up 12% from (45) in 2022.") == [1234.5e6, 12.0, -45.0]
assert values("Sections 10(b) and 20(a); on August 29, 2021 and in fiscal 2023") == []
assert values("The three goals are: 1) growth, 2) margin") == []
assert values("decreased by 34.7%") == [34.7] and values("-6%") == [-6.0] and values("$5.2 billion") == [5.2e9]
assert values("Q4 2023 EPS of $1.50") == [1.5]
assert same_up_to_presentation(74_203, 74.2e6) and not same_up_to_presentation(5.2e6, 5.2e9 * 1.1)
assert matches_target(extract_numbers("about 31 percent")[0], 4 / 13 * 100)
print("number parser tests passed")


# %% [markdown]
# ### 3.5 Question types
#
# Every question is tagged by rule:
#
# | Type | Rule |
# |---|---|
# | `text_lookup` | the gold answer has no number other than dates |
# | `numeric_lookup` | every number in the answer appears in the passage or question |
# | `calculation` | each new number in the answer comes from one operation on two passage numbers: a−b, a+b, a/b, a/b×100 or (a−b)/b×100 |
# | `unexplained` | a new number no single operation produces, usually a gold answer with facts missing from the passage; left out of the test set |
#
# `reproducible` tries every operation on every pair of passage numbers. Because the arithmetic has to reproduce the gold number, the rule also catches gold answers that are wrong. The cell prints the counts and three examples of each harder type.

# %%
def reproducible(target, pool):
    vals = [p["value"] for p in pool]
    for i, a in enumerate(vals):
        for j, b in enumerate(vals):
            if i == j:
                continue
            cands = [a - b, a + b] + ([a / b, a / b * 100, (a - b) / abs(b) * 100] if b else [])
            if any(matches_target(target, c) for c in cands):
                return True
    return False


def question_type(question, answer, context):
    ans = extract_numbers(answer)
    if not ans:
        return "text_lookup"
    pool = extract_numbers(context) + extract_numbers(question)
    new = [n for n in ans if not any(same_up_to_presentation(n["value"], p["value"]) for p in pool)]
    if not new:
        return "numeric_lookup"
    ctx_nums = extract_numbers(context)
    return "calculation" if all(reproducible(n, ctx_nums) for n in new) else "unexplained"


df["qtype"] = [question_type(q, a, c) for q, a, c in zip(df.question, df.answer, df.context)]
display(df.qtype.value_counts().rename("questions").to_frame())
for kind in ("calculation", "unexplained"):
    print(f"\n--- {kind}: 3 examples ---")
    for r in df[df.qtype == kind].sample(3, random_state=SEED).itertuples():
        print(f"Q: {r.question}\nA: {r.answer}\nC: {r.context[:240]}\n")


# %% [markdown]
# ### 3.6 Label checks
#
# Rules can match by coincidence, so two files are exported for review: a blind sample of 100 questions (stratified, rule tag hidden) and every real calculation question. `export_once` never overwrites a file, and `read_verified` reads back the ids marked `yes`. The completed files were filled in with AI assistance under the team's supervision.
#
# If the labelled sample is present, the cell reports how often the rule agrees with the labels and Cohen's κ (agreement corrected for chance).

# %%
LABELS = ART / "labels"


def export_once(frame, name):
    path = LABELS / name
    if path.exists():
        print(f"kept existing {path} (not overwritten)")
    else:
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")


def read_verified(name, key):
    path = LABELS / name
    if not path.exists():
        return None
    done = pd.read_csv(path)
    return set(done.loc[done.verified.astype(str).str.strip().str.lower() == "yes", key])


blind = pd.concat([df[df.qtype == "text_lookup"].sample(40, random_state=SEED),
                   df[df.qtype == "numeric_lookup"].sample(35, random_state=SEED),
                   df[df.qtype == "calculation"].sample(15, random_state=SEED),
                   df[df.qtype == "unexplained"].sample(10, random_state=SEED)]).sample(frac=1, random_state=SEED)
export_once(blind[["qid", "ticker", "question", "answer", "context"]].assign(human_label=""), "qtype_sample_to_label.csv")
export_once(df[df.qtype == "calculation"][["qid", "ticker", "question", "answer", "context"]].assign(verified="", note=""),
            "real_calc_to_verify.csv")

human_path = LABELS / "qtype_human_labels.csv"
if human_path.exists():
    from sklearn.metrics import cohen_kappa_score
    h = pd.read_csv(human_path).merge(df[["qid", "qtype"]], on="qid")
    h = h[h.human_label.notna() & (h.human_label.astype(str).str.strip() != "")]
    print(f"rule vs labels on {len(h)} rows: agreement {np.mean(h.human_label == h.qtype):.1%}, "
          f"Cohen's kappa {cohen_kappa_score(h.human_label, h.qtype):.2f}")
    display(pd.crosstab(h.human_label, h.qtype, rownames=["label"], colnames=["rule"]))
else:
    print(f"PENDING: {human_path} not provided yet, so rule-vs-label agreement is not computed")

real_calc_ok = read_verified("real_calc_verified.csv", "qid")
print("real calculation items:", int((df.qtype == "calculation").sum()),
      "| verified:", "pending" if real_calc_ok is None else len(real_calc_ok))


# %% [markdown]
# ### 3.7 Derived calculation questions
#
# Only 48 real questions need arithmetic, too few to compare systems. We generate more from the same passages.
#
# **3.7.1 Generator.** For each of five operations (difference, percent change, share of total, ratio, sum) there is a definition, an example phrasing and words the question must contain. `phi4-mini`, a different model from the agent, returns schema-constrained JSON: the question, the operation (fixed per call, so it can't drift to an easier one) and the two operands copied verbatim.

# %%
from typing import Literal

from pydantic import BaseModel, Field


OP_SPEC = {
    "difference": ("operand_a minus operand_b, reported as a positive amount (operand_a is the later period or larger item)",
                   "By how much did X increase from 2022 to 2023?",
                   ("difference", "how much", "increase", "decrease", "more", "less", "change", "higher", "lower", "grow")),
    "percent_change": ("(operand_a - operand_b) / operand_b * 100, where operand_a is the later value and operand_b the earlier",
                       "By what percentage did X change from 2022 to 2023?", ("percent", "%")),
    "share_of_total": ("operand_a / operand_b * 100, where operand_a is a part and operand_b is the total it belongs to",
                       "What percentage of total X did Y represent?", ("percent", "%", "share", "proportion", "portion")),
    "ratio": ("operand_a / operand_b", "What is the ratio of X to Y?", ("ratio", "times", "multiple")),
    "sum": ("operand_a plus operand_b", "What were X and Y combined?", ("total", "combined", "sum", "together", "aggregate")),
}
OP_CYCLE = list(OP_SPEC)
OPS = {"difference": lambda a, b: abs(a - b), "sum": lambda a, b: a + b, "ratio": lambda a, b: a / b,
       "share_of_total": lambda a, b: a / b * 100, "percent_change": lambda a, b: (a - b) / abs(b) * 100}


def derived_schema(op):

    class DerivedQuestion(BaseModel):
        question: str = Field(description="One question that needs exactly this arithmetic step on two numbers "
                                          "from the passage. Do not state the answer.")
        operation: Literal[op]
        operand_a: str = Field(description="The first number copied exactly as written in the passage, "
                                           "with its $ sign or unit word")
        operand_b: str = Field(description="The second number copied exactly as written in the passage")

    return DerivedQuestion


DERIVE_PROMPT = """You write evaluation questions for a financial analyst assistant.

Passage from the company's 10-K filing:
\"\"\"{context}\"\"\"

Write ONE question whose answer requires exactly one arithmetic step on two financial amounts that appear in
the passage. Use only money amounts or counts of business units (stores, employees, customers). Never use page
numbers, dates, addresses, phone numbers, or note/item numbers.
The arithmetic step must be: {operation} = {definition}
Phrase the question like: "{example}"
The answer must NOT already be written in the passage. Refer to the company as "the company".
Copy operand_a and operand_b exactly as they are written in the passage."""


# %% [markdown]
# **3.7.2 Checks and the gold answer.** `check_derived` rejects an item if the question doesn't ask for its operation, an operand isn't verbatim in the passage or doesn't parse to one number, the operands are the same or of different kinds, or the result is already stated somewhere. Otherwise Python computes the gold value and formats it like the operands (`$1,031 million`, `12.5%`, `1.04`). The LLM never computes an answer.

# %%
def fmt_amount(r, a, b):
    unit = max(SCALE.get(a["scale_word"], 1.0), SCALE.get(b["scale_word"], 1.0))
    while unit > 1 and abs(r) < unit:
        unit /= 1000
    word = {1e3: "thousand", 1e6: "million", 1e9: "billion", 1e12: "trillion"}.get(unit, "")
    dec = max([n["dec"] for n in (a, b) if SCALE.get(n["scale_word"], 1.0) == unit] or [1])
    return ("-" if r < 0 else "") + ("$" if a["dollar"] else "") + f"{abs(r) / unit:,.{dec}f}" + (f" {word}" if word else "")


def check_derived(context, dq):
    if not any(w in dq.question.lower() for w in OP_SPEC[dq.operation][2]):
        raise ValueError(f"question wording does not ask for a {dq.operation}")
    for o in (dq.operand_a, dq.operand_b):
        if norm_text(o) not in norm_text(context):
            raise ValueError("operand not verbatim in passage")
    na, nb = extract_numbers(dq.operand_a), extract_numbers(dq.operand_b)
    if len(na) != 1 or len(nb) != 1:
        raise ValueError("operand does not parse to exactly one number")
    a, b = na[0], nb[0]
    if a["value"] == b["value"]:
        raise ValueError("identical operands")
    if dq.operation in ("difference", "sum") and (a["pct"], a["dollar"]) != (b["pct"], b["dollar"]):
        raise ValueError("operands are different kinds (percent, dollar or other amount)")
    if dq.operation in ("percent_change", "ratio", "share_of_total") and (a["pct"] or b["pct"] or b["value"] == 0):
        raise ValueError("ratio-type operation needs two non-zero amounts")
    r = OPS[dq.operation](a["value"], b["value"])
    if dq.operation == "share_of_total" and not 0 < r < 100:
        raise ValueError("share of total outside 0-100%")
    if any(same_up_to_presentation(r, n["value"]) for n in extract_numbers(context) + extract_numbers(dq.question)):
        raise ValueError("result already stated in passage or question")
    if dq.operation in ("percent_change", "share_of_total"):
        return r, f"{r:.1f}%"
    if dq.operation == "ratio":
        return r, f"{r:.2f}"
    if a["pct"]:
        return r, f"{r:.1f} percentage points"
    return r, fmt_amount(r, a, b)


def money_count(text):
    return sum(1 for n in extract_numbers(text) if (n["dollar"] or n["scaled"]) and not n["pct"])


def generate_derived(context, op):
    gen = get_llm(CONFIG["judge_model"], num_predict=300).with_structured_output(derived_schema(op), method="json_schema")
    spec = OP_SPEC[op]
    return gen.invoke(DERIVE_PROMPT.format(context=context, operation=op, definition=spec[0], example=spec[1]))


# %% [markdown]
# **3.7.3 Run.** Candidate passages are short (under 600 characters), contain at least two money amounts, and are not the source of a real calculation or unexplained question. Operations are assigned in rotation so each gets about the same number of questions. Results are cached, so this cell only calls the LLM when `GENERATE_DERIVED = 1` and items are missing. Surviving items are exported for review.

# %%
excluded = set(df.loc[df.qtype.isin(["calculation", "unexplained"]), "gold_doc"])
pool = docs[(docs.context.str.len() < 600) & (docs.context.map(money_count) >= 2) & ~docs.doc_id.isin(excluded)]
candidates = pool.sample(min(CONFIG["n_derived_candidates"], len(pool)), random_state=SEED)
print(f"passages eligible for derived questions: {len(pool)}; sampled: {len(candidates)}")

derived_store = JsonlStore(ART / "derived" / "derived_calc_raw_v3.jsonl")
todo = [r for r in candidates.itertuples() if r.doc_id not in derived_store]
if todo and not GENERATE_DERIVED:
    raise RuntimeError(f"{len(todo)} derived candidates have no saved result in {derived_store.path}; "
                       "set GENERATE_DERIVED = True (about 3 s per item) to generate them")
if todo:
    t0 = time.time()
    op_for = {d: OP_CYCLE[i % len(OP_CYCLE)] for i, d in enumerate(candidates.doc_id)}
    for i, r in enumerate(todo, 1):
        row = {"id": r.doc_id, "doc_id": r.doc_id, "ticker": r.ticker, "context": r.context,
               "requested_operation": op_for[r.doc_id]}
        try:
            dq = generate_derived(r.context, op_for[r.doc_id])
            row.update(dq.model_dump())
            row["gold_value"], row["gold_answer"] = check_derived(r.context, dq)
            row["status"] = "ok"
        except ValueError as e:
            row["status"] = f"rejected: {str(e).splitlines()[0][:80]}"
        derived_store.add(row)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} generated ({time.time() - t0:.0f}s)")

raw_derived = pd.DataFrame([derived_store[d] for d in candidates.doc_id])
display(raw_derived.status.value_counts().rename("items").to_frame())
derived = raw_derived[raw_derived.status == "ok"].reset_index(drop=True)
derived.insert(0, "qid", [f"x{d[1:]}_{hashlib.sha1(q.encode()).hexdigest()[:6]}"
                          for d, q in zip(derived.doc_id, derived.question)])
display(derived.operation.value_counts().rename("derived questions").to_frame())
export_once(derived[["qid", "ticker", "question", "operation", "operand_a", "operand_b", "gold_answer", "context"]]
            .assign(verified="", note=""), "derived_calc_to_verify.csv")
derived_ok = read_verified("derived_calc_verified.csv", "qid")
print(f"derived calculation questions passing the automatic checks: {len(derived)}; verified: "
      f"{'pending' if derived_ok is None else len(derived_ok & set(derived.qid))}")
display(derived[["question", "operation", "operand_a", "operand_b", "gold_answer"]].head(5))


# %% [markdown]
# ### 3.8 Dev and test splits
#
# All draws use `SEED`. Dev is only for prompt work; every reported number is from test.
#
# | Split | Contents |
# |---|---|
# | test | 100 `text_lookup`, 100 `numeric_lookup`, all real calculation questions, all derived questions except 20 |
# | dev | 15 `text_lookup`, 15 `numeric_lookup`, 20 derived |
# | quick | 10 test questions per type, run live when `QUICK_RUN = 1` |
#
# `gold_numbers` are the figures the numeric metric checks: for real calculation questions, only the numbers the passage doesn't already state. Items rejected in review are removed **after** the assignment, so a rejection never moves a question between dev and test. The splits are saved to `artifacts/`.

# %%
def real_rows(frame, kind):
    return pd.DataFrame({"qid": frame.qid, "source": "real", "ticker": frame.ticker, "question": frame.question,
                         "gold_answer": frame.answer, "qtype": kind, "gold_doc": frame.gold_doc,
                         "relevant_docs": frame.relevant_docs})


n_dev, n_test = CONFIG["n_dev_lookup_per_type"], CONFIG["n_test_lookup_per_type"]
parts = []
for kind in ("text_lookup", "numeric_lookup"):
    pick = df[df.qtype == kind].sample(n_dev + n_test, random_state=SEED)
    parts += [real_rows(pick.iloc[:n_dev], kind).assign(split="dev"),
              real_rows(pick.iloc[n_dev:], kind).assign(split="test")]
parts.append(real_rows(df[df.qtype == "calculation"], "calculation").assign(split="test"))
der = pd.DataFrame({"qid": derived.qid, "source": "derived", "ticker": derived.ticker, "question": derived.question,
                    "gold_answer": derived.gold_answer, "qtype": "derived_calculation", "gold_doc": derived.doc_id,
                    "relevant_docs": derived.doc_id.map(lambda d: sorted(related[d]))})
der_order = der.sample(frac=1, random_state=SEED)
parts += [der_order.iloc[:CONFIG["n_dev_derived"]].assign(split="dev"),
          der_order.iloc[CONFIG["n_dev_derived"]:].assign(split="test")]
questions = pd.concat(parts, ignore_index=True)

calc_gold = {r.qid: [n["value"] for n in extract_numbers(r.answer)
                     if not any(same_up_to_presentation(n["value"], p["value"]) for p in extract_numbers(r.context))]
             for r in df[df.qtype == "calculation"].itertuples()}
gold_value = dict(zip(derived.qid, derived.gold_value))
questions["gold_numbers"] = [calc_gold[q] if t == "calculation" else [gold_value[q]] if t == "derived_calculation"
                             else values(a) for q, t, a in zip(questions.qid, questions.qtype, questions.gold_answer)]

quick_ids = set()
for kind, g in questions[questions.split == "test"].groupby("qtype"):
    quick_ids |= set(g.sample(min(CONFIG["n_quick_per_type"], len(g)), random_state=SEED).qid)
questions["quick"] = questions.qid.isin(quick_ids)

keep = pd.Series(True, index=questions.index)
for kind, ok in (("calculation", real_calc_ok), ("derived_calculation", derived_ok)):
    if ok is not None:
        keep &= (questions.qtype != kind) | questions.qid.isin(ok)
print(f"dropped {int((~keep).sum())} items that failed verification")
questions = questions[keep].reset_index(drop=True)

(ART / "splits.json").write_text(json.dumps({s: sorted(g.qid) for s, g in questions.groupby("split")}, indent=1))
questions.to_json(ART / "questions.jsonl", orient="records", lines=True, force_ascii=False)
display(pd.crosstab(questions.qtype, questions.split, margins=True))
print(f"quick subset: {int(questions.quick.sum())} questions")


# %% [markdown]
# ## 4. Evaluation Harness and Baseline
#
# ### 4.1 Evaluation set and prediction cache
#
# `eval_q` is the full test set, or the quick subset when `QUICK_RUN = 1`. `run_condition` loads saved predictions for a condition and only runs the missing ones. It stops if a cached prediction was made for different question text, and refuses to run live when `USE_CACHED_PREDS = 1`.

# %%
eval_q = questions[(questions.split == "test") & (questions.quick | (not QUICK_RUN))].reset_index(drop=True)
MODEL_TAG = re.sub(r"[^A-Za-z0-9]+", "-", CONFIG["agent_model"]).strip("-")
print(f"evaluation set: {'quick subset' if QUICK_RUN else 'full test set'}, {len(eval_q)} questions")
display(eval_q.qtype.value_counts().rename("questions").to_frame())


# the version is part of the cache file name; bump it when a prompt or the graph changes
BASELINE_VERSION, AGENT_VERSION = "v1", "v3"


def live_allowed(kind):
    if USE_CACHED_PREDS and not QUICK_RUN:
        raise RuntimeError(f"{kind} missing from the cache and USE_CACHED_PREDS = True. Run the full evaluation once "
                           "with USE_CACHED_PREDS = False (several hours on a laptop GPU) or set QUICK_RUN = True.")


def run_condition(name, answer_fn, version, qs=None):
    qs = eval_q if qs is None else qs
    store = JsonlStore(ART / "preds" / f"{name}__{MODEL_TAG}__{version}.jsonl")
    stale = [r.qid for r in qs.itertuples() if r.qid in store and store[r.qid]["question"] != r.question]
    if stale:
        raise RuntimeError(f"cached predictions in {store.path} were made for different question text: {stale[:5]}")
    missing = [r for r in qs.itertuples() if r.qid not in store]
    if missing:
        live_allowed(f"{len(missing)} predictions for {name!r} are")
        t0 = time.time()
        for i, r in enumerate(missing, 1):
            store.add({"id": r.qid, "condition": name, "question": r.question, **answer_fn(r)})
            if i % 20 == 0 or i == len(missing):
                print(f"  {name}: {i}/{len(missing)} ({time.time() - t0:.0f}s)")
    return pd.DataFrame([store[q] for q in qs.qid])


# %% [markdown]
# ### 4.2 LLM judge
#
# `phi4-mini`, from a different model family than the agent, compares each answer with the gold answer under a strict rubric and replies YES or NO. Verdicts are cached by a hash of (question, gold, answer), so an answer the judge has seen is never re-judged. `REFUSAL_RE` spots answers like "the filing does not say".

# %%
JUDGE_TEMPLATE = (
    "You grade answers to questions about company annual reports.\n"
    "Question: {q}\nGold answer: {gold}\nCandidate answer: {pred}\n\n"
    "Rules: the candidate is correct only if it states the same key fact as the gold answer. "
    "Every key number must match the gold number after converting units (million/billion, percent/fraction); "
    "rounding to the precision shown in the gold answer is fine. A different number, the opposite direction "
    "(increase vs decrease), different entities, or no answer means NO. Extra correct detail is fine.\n"
    "Reply with exactly one word: YES or NO.")
# changing the judge prompt must not reuse old verdicts
JUDGE_VERSION = "strict-v1"
judge_llm = get_llm(CONFIG["judge_model"], num_predict=5)
judge_store = JsonlStore(ART / "judge" / f"{CONFIG['judge_model'].replace(':', '-')}__{JUDGE_VERSION}.jsonl")
REFUSAL_RE = re.compile(
    r"\b(?:does not|doesn't|do not|did not|cannot|can't|unable to)\s+(?:\w+\s+){0,3}?"
    r"(?:contain|provide|include|mention|disclose|specify|report|find|answer)\b"
    r"|\bnot (?:available|provided|disclosed|mentioned|found|specified|included)\b"
    r"|\bno (?:information|data|figure|specific)\b|\bI'?m sorry\b|\bdon'?t have that information\b", re.I)


def judge_answer(question, gold, pred):
    raw = judge_llm.invoke(JUDGE_TEMPLATE.format(q=question, gold=gold, pred=pred)).content.strip()
    m = re.match(r"\W*(YES|NO)\b", raw.upper())
    return (m.group(1) == "YES") if m else None, raw[:40]


def judge_key(question, gold, answer):
    return hashlib.sha1(f"{question}\n{gold}\n{answer}".encode()).hexdigest()[:20]


def judge_condition(name, preds, qs=None):
    qs = eval_q if qs is None else qs
    answers = dict(zip(preds.id, preds.answer))
    keys = {r.qid: judge_key(r.question, r.gold_answer, answers[r.qid]) for r in qs.itertuples()}
    missing = [r for r in qs.itertuples() if keys[r.qid] not in judge_store]
    if missing:
        live_allowed(f"{len(missing)} judge verdicts for {name!r} are")
        for r in missing:
            ok, raw = judge_answer(r.question, r.gold_answer, answers[r.qid])
            judge_store.add({"id": keys[r.qid], "qid": r.qid, "answer": answers[r.qid], "correct": ok, "raw": raw})
    return {q: judge_store[k]["correct"] for q, k in keys.items()}


# %% [markdown]
# ### 4.3 Numeric accuracy
#
# For calculation questions, every gold number must appear in the answer within 1%, either exactly or as rounded in the gold text (exact 0.963%, gold "1.0%", answer "1%"). Answers written ×100 or ÷100 count, as in Program-of-Thoughts' `finqa_equal`. Scale words are compared only when the gold has one: "$1,318 million" matches a gold "$1,318.0" from a table in millions. The sign is ignored because decreases are written in words.

# %%
def numeric_correct(answer, gold_numbers, gold_answer):
    preds = extract_numbers(answer)
    gold_shown = extract_numbers(gold_answer)
    gold_unscaled = not any(n["scaled"] for n in gold_shown)

    def hit(g):
        g = abs(g)
        for p in preds:
            v = abs(p["value"])
            rescale = not p["scaled"] or gold_unscaled
            cands = [v, v * 100, v / 100] + ([v * 1000.0 ** k for k in (-3, -2, -1, 1, 2, 3)] if rescale else [])
            if any(math.isclose(c, g, rel_tol=0.01) for c in cands):
                return True
        return False

    def matched(g):
        rounded = [n["value"] for n in gold_shown if math.isclose(abs(n["value"]), abs(g), rel_tol=0.05)]
        return hit(g) or any(hit(r) for r in rounded)

    return all(matched(g) for g in gold_numbers) if gold_numbers else None


assert numeric_correct("It grew by $5,606 million.", [5.606e9], "$5.6 billion") and numeric_correct("about 30.8%", [31.0], "31%")
assert not numeric_correct("It grew by $5.6 billion and 12%.", [8.548e9], "$8.5 billion")
assert numeric_correct("Services were 0.125 of revenue", [12.5], "12.5%")
assert not numeric_correct("$5.2 billion", [5.2e6], "$5.2 million") and numeric_correct("$1,318 million", [1318.0], "$1,318.0")
assert numeric_correct("It rose 1%", [0.9626], "1.0%") and not numeric_correct("about $1 billion", [1.4e9], "$1.4 billion")


# %% [markdown]
# ### 4.4 Scoring
#
# An answer counts as correct only if it passes three checks in order:
# 1. It is not a refusal. Every gold answer exists, so refusing is wrong.
# 2. The number rule: numeric lookups must contain at least one gold figure, calculation answers all of them.
# 3. The LLM judge says YES.
#
# On dev, `phi4-mini` alone accepted refusals and wrong figures, which is why the rule checks run first. `score` also records calculator use, whether the cited passage is relevant, whether the gold passage was retrieved, and cost (seconds, LLM calls, tokens).

# %%
CALC_TYPES = ("calculation", "derived_calculation")
assert REFUSAL_RE.search("The 2023 annual report does not contain information about it")
assert REFUSAL_RE.search("I'm sorry, but there is no publicly available report") and not REFUSAL_RE.search("Revenue was $5 billion")


def rule_gate(qtype, answer, gold_numbers, gold_answer):
    if REFUSAL_RE.search(answer):
        return False, "refusal"
    if qtype in CALC_TYPES and not numeric_correct(answer, gold_numbers, gold_answer):
        return False, "number mismatch"
    if qtype == "numeric_lookup" and gold_numbers and not any(numeric_correct(answer, [g], gold_answer)
                                                              for g in gold_numbers):
        return False, "number mismatch"
    return True, None


def score(name, preds, qs=None):
    qs = eval_q if qs is None else qs
    verdicts = judge_condition(name, preds, qs)
    p = qs.merge(preds.drop(columns="question").rename(columns={"id": "qid"}), on="qid")
    p["condition"] = name
    gates = [rule_gate(t, a, g, ga) for t, a, g, ga in zip(p.qtype, p.answer, p.gold_numbers, p.gold_answer)]
    p["refusal"] = [why == "refusal" for _, why in gates]
    p["llm_judge_ok"] = p.qid.map(verdicts)
    p["judge_ok"] = [False if not gate_ok else llm for (gate_ok, _), llm in zip(gates, p.llm_judge_ok)]
    p["fail_reason"] = [why if not ok else ("LLM judge" if llm is False else None)
                        for (ok, why), llm in zip(gates, p.llm_judge_ok)]
    p["numeric_ok"] = [numeric_correct(a, g, ga) if t in CALC_TYPES else None
                       for a, g, ga, t in zip(p.answer, p.gold_numbers, p.gold_answer, p.qtype)]
    p["calc_called"] = [any(c["name"] == "run_python" for c in tc) for tc in p.get("tool_calls", [[]] * len(p))]
    p["cited_ok"] = [c in rel if isinstance(c, str) else False for c, rel in zip(p.get("cited_doc", [None] * len(p)),
                                                                                p.relevant_docs)]
    p["gold_retrieved"] = [bool(set(r) & set(rel)) if isinstance(r, list) else None
                           for r, rel in zip(p.get("retrieved_docs", [None] * len(p)), p.relevant_docs)]
    return p


def answer_from_llm(llm, prompt):
    t0 = time.time()
    msg = llm.invoke(prompt)
    u = msg.usage_metadata or {}
    return msg.content.strip(), {"latency_s": round(time.time() - t0, 2), "llm_calls": 1,
                                 "input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}


# %% [markdown]
# ### 4.5 Baseline B0: closed book
#
# The agent model answers from memory, with no retrieval and no tools. This shows how much it knows about these filings on its own.

# %%
agent_llm = get_llm(CONFIG["agent_model"], num_ctx=CONFIG["agent_num_ctx"])
CLOSED_BOOK = ("You are a financial analyst. Answer the question about the company with ticker {ticker} from its "
               "2023 annual report (10-K). Answer in one or two sentences with the figure and its unit.\n"
               "Question: {question}")


def closed_book(r):
    text, cost = answer_from_llm(agent_llm, CLOSED_BOOK.format(ticker=r.ticker, question=r.question))
    return {"answer": text, **cost}


preds_b0 = run_condition("B0_closed_book", closed_book, BASELINE_VERSION)
scored = {"B0_closed_book": score("B0_closed_book", preds_b0)}
display(scored["B0_closed_book"].groupby("qtype")[["judge_ok", "numeric_ok"]].mean().round(3))


# %% [markdown]
# ## 5. System
#
# ### 5.1 Retrieval indexes
#
# Adapted from the Module 3 RAG lab and `Agentic_app/relay_ops/rag.py` (Module 4).
#
# - **Dense**: passages are embedded with `bge-small` (and `MiniLM` for comparison) and searched by cosine similarity in FAISS. Embeddings are cached. With a ticker, FAISS searches only that company's rows.
# - **BM25**: keyword scoring where rare words count more. The tokenizer keeps `1,234.5` as one token (the lab's split it into three). With a ticker, scores are masked to that company.

# %%
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

doc_ids = docs.doc_id.to_numpy()
doc_text = docs.context.tolist()
doc_ticker = docs.ticker.to_numpy()
row_of = {d: i for i, d in enumerate(doc_ids)}
ticker_rows = {t: np.flatnonzero(doc_ticker == t).astype("int64") for t in np.unique(doc_ticker)}


class DenseIndex:
    def __init__(self, model_name, query_prefix=""):
        self.name, self.prefix = model_name, query_prefix
        self.model = SentenceTransformer(model_name, device=DEVICE)
        key = hashlib.sha1(("\n".join(doc_text) + model_name).encode()).hexdigest()[:12]
        path = ART / "cache" / f"emb_{model_name.split('/')[-1]}_{key}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            emb = self.model.encode(doc_text, batch_size=64, normalize_embeddings=True,
                                    show_progress_bar=False).astype("float32")
            np.save(path, emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query, k, ticker=None):
        q = self.model.encode([self.prefix + query], normalize_embeddings=True).astype("float32")
        params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ticker_rows[ticker])) if ticker else None
        scores, rows = self.index.search(q, k, params=params)
        return [(int(i), float(s)) for i, s in zip(rows[0], scores[0]) if i >= 0]


def bm25_tokens(text):
    return [t.replace(",", "") for t in re.findall(r"\d+(?:[.,]\d+)*|[a-z]+", text.lower())]


bm25 = BM25Okapi([bm25_tokens(t) for t in doc_text])


def bm25_search(query, k, ticker=None):
    scores = bm25.get_scores(bm25_tokens(query))
    rows = ticker_rows[ticker] if ticker else np.arange(len(doc_text))
    top = rows[np.argsort(-scores[rows], kind="stable")[:k]]
    return [(int(i), float(scores[i])) for i in top]


# %% [markdown]
# ### 5.2 Fusion, re-ranking and `retrieve`
#
# Reciprocal Rank Fusion merges the dense and BM25 lists by rank (each passage scores Σ 1/(60 + rank)), so their scores never have to be calibrated against each other. A cross-encoder then reads the query and each of the top 20 passages together and re-orders them. `retrieve` exposes all four methods; the cell ends with a sample query.

# %%
def rrf_fuse(*ranked_lists, k_const=CONFIG["rrf_k"]):
    scores = {}
    for ranked in ranked_lists:
        for rank, (i, _) in enumerate(ranked):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k_const + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


reranker = CrossEncoder(CONFIG["reranker"], device=DEVICE)


def rerank(query, candidates, k):
    if not candidates:
        return []
    s = reranker.predict([(query, doc_text[i]) for i, _ in candidates], show_progress_bar=False)
    return [(candidates[j][0], float(s[j])) for j in np.argsort(-s, kind="stable")[:k]]


dense_bge = DenseIndex(CONFIG["embedder"], query_prefix="Represent this sentence for searching relevant passages: ")
dense_minilm = DenseIndex(CONFIG["embedder_baseline"])


def retrieve(query, ticker=None, k=CONFIG["top_k"], method="hybrid_rerank", dense=dense_bge):
    pool = CONFIG["rerank_pool"]
    if method == "dense":
        return dense.search(query, k, ticker)
    if method == "bm25":
        return bm25_search(query, k, ticker)
    fused = rrf_fuse(dense.search(query, pool, ticker), bm25_search(query, pool, ticker))
    if method == "hybrid":
        return fused[:k]
    if method == "hybrid_rerank":
        return rerank(query, fused[:pool], k)
    raise ValueError(f"unknown retrieval method {method!r}")


print(f"indexed {len(doc_text)} passages; dense dims: bge {dense_bge.index.d}, MiniLM {dense_minilm.index.d}")
for i, s in retrieve("What was total revenue in fiscal 2023?", ticker="NVDA", k=3):
    print(f"  {doc_ids[i]} [{doc_ticker[i]}] {s:6.2f}  {doc_text[i][:110]}")


# %% [markdown]
# ### 5.3 Retrieval on its own
#
# Each test question is the query and its relevant passages are the target. For every method, with the ticker filter on and off, we measure Recall@k (is a relevant passage in the top k?) and MRR@10 (the average of 1/rank of the first relevant passage). The last table breaks Recall@5 down by question type.

# %%
RETRIEVERS = {"dense (MiniLM)": ("dense", dense_minilm), "dense (bge-small)": ("dense", dense_bge),
              "BM25": ("bm25", dense_bge), "hybrid RRF": ("hybrid", dense_bge),
              "hybrid RRF + rerank": ("hybrid_rerank", dense_bge)}
KS = (1, 3, 5, 10)
test_q = questions[questions.split == "test"]


def retrieval_eval(method, dense, use_filter, qs):
    first_hit, t0 = [], time.time()
    for r in qs.itertuples():
        got = [doc_ids[i] for i, _ in retrieve(r.question, r.ticker if use_filter else None, max(KS), method, dense)]
        rel = set(r.relevant_docs)
        first_hit.append(next((p for p, d in enumerate(got) if d in rel), None))
    out = {f"R@{k}": np.mean([h is not None and h < k for h in first_hit]) for k in KS}
    out["MRR@10"] = np.mean([0.0 if h is None else 1 / (h + 1) for h in first_hit])
    out["ms/query"] = 1000 * (time.time() - t0) / len(qs)
    return out, first_hit


ret_rows, ret_hits = [], {}
for name, (method, dense) in RETRIEVERS.items():
    for use_filter in (True, False):
        res, hits = retrieval_eval(method, dense, use_filter, test_q)
        ret_rows.append({"retriever": name, "ticker filter": "on" if use_filter else "off", **res})
        ret_hits[(name, use_filter)] = hits
retrieval_table = pd.DataFrame(ret_rows)
display(retrieval_table.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for name in RETRIEVERS:
    row = retrieval_table[(retrieval_table.retriever == name) & (retrieval_table["ticker filter"] == "on")].iloc[0]
    axes[0].plot(KS, [row[f"R@{k}"] for k in KS], marker="o", label=name)
axes[0].set(title="Recall@k with ticker filter (test set)", xlabel="k", ylabel="Recall@k", xticks=KS)
axes[0].legend(fontsize=8)
piv = retrieval_table.pivot(index="retriever", columns="ticker filter", values="R@5").loc[list(RETRIEVERS)]
piv[["off", "on"]].plot.barh(ax=axes[1], title="Recall@5: ticker filter off vs on")
axes[1].set_xlim(0, 1)
plt.tight_layout()
plt.show()

hits5 = pd.Series([h is not None and h < 5 for h in ret_hits[("hybrid RRF + rerank", True)]], index=test_q.index)
display(test_q.assign(hit=hits5).groupby("qtype").hit.agg(["mean", "size"])
        .rename(columns={"mean": "R@5 (hybrid+rerank, filter on)", "size": "questions"}))


# %% [markdown]
# ### 5.4 Calculator: checking the code
#
# Adapted from `Agentic_app/relay_ops/tools.py` (Module 4) and the PAL cell of the Module 2 lab. The lab's version ran `exec` inside the agent behind a banned-word list, so `while True: pass` hung the agent and `rev = 1,234.5` silently became a tuple.
#
# Here the code is parsed first and every syntax node checked against an allow-list: arithmetic, assignments, comparisons, `for`, comprehensions, f-strings, and calls to safe built-ins and `math.*`. Imports, `while`, `def`, `lambda` and other attributes are rejected. `prepare_code` also strips thousands separators and turns a final expression into `answer = ...`.

# %%
import ast

from langchain_core.tools import tool

SANDBOX_TIMEOUT_S = 5
SAFE_CALLS = {"abs", "round", "min", "max", "sum", "len", "float", "int", "pow", "range", "print", "sorted", "list"}
ALLOWED_NODES = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store, ast.Constant,
                 ast.BinOp, ast.UnaryOp, ast.operator, ast.unaryop, ast.Compare, ast.cmpop, ast.BoolOp, ast.boolop,
                 ast.IfExp, ast.Call, ast.keyword, ast.Attribute, ast.List, ast.Tuple, ast.Subscript, ast.Slice,
                 ast.For, ast.Pass, ast.comprehension, ast.ListComp, ast.GeneratorExp, ast.JoinedStr,
                 ast.FormattedValue)
SANDBOX_RUNNER = r"""
import builtins, contextlib, io, json, math, sys
safe = {n: getattr(builtins, n) for n in sys.argv[1].split(",")}
ns, buf = {"__builtins__": safe, "math": math}, io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(sys.stdin.read(), "<run_python>", "exec"), ns)
ans = ns.get("answer")
print(json.dumps({"stdout": buf.getvalue()[-1500:],
                  "answer": ans if isinstance(ans, (int, float, str, type(None))) else repr(ans)}))
"""


def check_sandbox_code(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"{type(node).__name__} is not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and not (isinstance(node.value, ast.Name) and node.value.id == "math"
                                                    and not node.attr.startswith("_")):
            raise ValueError("only math.<function> attributes are allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in SAFE_CALLS:
            raise ValueError(f"calling {node.func.id}() is not allowed; use arithmetic, math.* or {sorted(SAFE_CALLS)}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp = node.right.operand if isinstance(node.right, ast.UnaryOp) else node.right
            small_const = isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)) and abs(exp.value) <= 100
            if not (small_const or isinstance(exp, ast.Name)):
                raise ValueError("exponent too large (use a number up to 100 or a variable)")


def prepare_code(code):
    code = code.strip().strip("`").strip()
    code = re.sub(r"^python\s*\n", "", code)
    code = re.sub(r"(?<![\w.])(\d{1,3}(?:,\d{3})+)(?![\d,])",
                  lambda m: m.group(1).replace(",", ""), code)
    tree = ast.parse(code)
    check_sandbox_code(tree)
    if tree.body and isinstance(tree.body[-1], ast.Expr) and not (
            isinstance(tree.body[-1].value, ast.Call) and getattr(tree.body[-1].value.func, "id", "") == "print"):
        tree.body[-1] = ast.Assign(targets=[ast.Name("answer", ast.Store())], value=tree.body[-1].value)
        ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# %% [markdown]
# ### 5.5 Calculator: running the code
#
# The checked code runs in a separate `python -I` process with only the safe built-ins and a 5-second timeout. Errors come back as text the agent can read and fix. `run_python` is the tool the agent sees; its docstring is the description the model reads.

# %%
def fmt_number(x):
    if isinstance(x, float):
        if not math.isfinite(x):
            return str(x)
        x = round(x, 6)
        return f"{x:,.6f}".rstrip("0").rstrip(".")
    return f"{x:,}" if isinstance(x, int) else str(x)


def run_python_code(code):
    try:
        prepared = prepare_code(code)
    except SyntaxError as e:
        return f"Error: invalid Python syntax ({e.msg}, line {e.lineno})."
    except ValueError as e:
        return f"Error: {e}."
    try:
        proc = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", SANDBOX_RUNNER, ",".join(sorted(SAFE_CALLS))],
                              input=prepared, capture_output=True, text=True, encoding="utf-8",
                              timeout=SANDBOX_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"Error: the code ran longer than {SANDBOX_TIMEOUT_S} s and was stopped."
    if proc.returncode != 0:
        last = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        return f"Error while running the code: {last}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    parts = [out["stdout"].strip()] if out["stdout"].strip() else []
    if out["answer"] is not None:
        parts.append(f"answer = {fmt_number(out['answer'])}")
    return "\n".join(parts) or "The code ran but produced no output. Assign the result to `answer` or print it."


@tool
def run_python(code: str) -> str:
    """Calculator. Runs a short Python snippet and returns the result. Use it for ANY arithmetic instead of computing
    in your head. Write plain Python with the numbers copied from the passages, for example:
    answer = (52527 - 46921) / 46921 * 100
    Only arithmetic, math functions, round, min, max, sum and print are available. No imports."""
    return run_python_code(code)


# %% [markdown]
# ### 5.6 Calculator tests
#
# Fourteen cases: normal arithmetic, thousands separators, f-strings, and attacks that must fail (imports, file access, dunder tricks, infinite loops, huge exponents).

# %%
SANDBOX_TESTS = {
    "answer = 52527 - 46921": "answer = 5,606",
    "52,527 - 46,921": "answer = 5,606",
    "round((12508 - 10537) / 10537 * 100, 2)": "answer = 18.71",
    "rev = 1,234.5\nrev * 2": "answer = 2,469",
    "print(f'{1234.5 * 0.17:.2f}')": "209.87",
    "math.sqrt(16)": "answer = 4",
    "import os": "Error: Import is not allowed in the sandbox.",
    "open('x.txt')": "Error: calling open() is not allowed",
    "().__class__.__bases__": "Error: only math.<function> attributes are allowed.",
    "while True: pass": "Error: While is not allowed in the sandbox.",
    "for i in range(10**12): pass": "Error: the code ran longer than 5 s and was stopped.",
    "1 / 0": "Error while running the code: ZeroDivisionError: division by zero",
    "10 ** 10 ** 8": "Error: exponent too large",
    "x = 1.07\nx ** 3": "answer = 1.225043",
}
for code, expected in SANDBOX_TESTS.items():
    got = run_python_code(code)
    assert got.startswith(expected), f"{code!r}: expected {expected!r}, got {got!r}"
print(f"sandbox tests passed ({len(SANDBOX_TESTS)} cases)")


# %% [markdown]
# ### 5.7 Search tool
#
# `search_filings` wraps `retrieve` (hybrid + rerank, top 5). Each passage starts with its id, e.g. `[D01234 | NVDA]`, so the agent can cite it. The question's ticker is the default filter; an unknown ticker returns the valid list; repeating the exact same search is refused, since looping on one query is a common ReAct failure. The three calls show a normal search, a bad ticker and a repeat.

# %%
PASSAGE_ID_RE = re.compile(r"\[(D\d{5}) \|")
MAX_PASSAGE_CHARS = 2000


def make_search_tool(default_ticker=None, use_filter=True, k=CONFIG["top_k"]):
    seen = set()

    @tool
    def search_filings(query: str, ticker: str = "") -> str:
        """Search the 10-K filing passages. Give a short search query, plus the company ticker such as NVDA.
        Returns the most relevant passages. Each passage starts with its id in square brackets; cite that id."""
        key = (norm_text(query), (ticker or "").upper())
        if key in seen:
            return "You already ran this exact search. Use the passages above, or search with different words."
        seen.add(key)
        t = None
        if use_filter:
            t = (ticker or default_ticker or "").strip().upper() or None
            if t and t not in ticker_rows:
                return f"Error: unknown ticker {t!r}. Valid tickers: {', '.join(sorted(ticker_rows))}."
        hits = retrieve(query, t, k=k)
        if not hits:
            return "No passages found."
        return "\n\n".join(f"[{doc_ids[i]} | {doc_ticker[i]}] {doc_text[i][:MAX_PASSAGE_CHARS]}" for i, _ in hits)

    return search_filings


demo_search = make_search_tool(default_ticker="DAL")
print(demo_search.invoke({"query": "total operating expenses 2023 2022"})[:700])
print(demo_search.invoke({"query": "revenue", "ticker": "DELTA"}))
print(demo_search.invoke({"query": "total operating expenses 2023 2022"}))


# %% [markdown]
# ### 5.8 Agent prompt and nudges
#
# Adapted from `build_react_agent` and `ainvoke_requiring_tool` in the Module 4 `Agentic_app`.
#
# The system prompt tells the model to search first, use `run_python` for any arithmetic, and finish with an `Answer:` line and a `Source:` line. Its worked example uses a made-up company, so no test question leaks in. The nudges are short messages the harness sends when the model breaks a rule. `CALC_INTENT_RE` spots questions that ask for a change, total, ratio or percentage.

# %%
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

PROMPT_RULES = {
    True: ("2. Find the exact figures in the passages. If the question needs ANY arithmetic (a difference, change, "
           "percentage, ratio, total or share), call run_python with the numbers copied from the passages. "
           "Never do arithmetic in your head."),
    False: "2. Find the exact figures in the passages. If the question needs arithmetic, compute it carefully.",
}
EXAMPLE = {
    True: """Example (an invented company):
User: Company ticker: ACME. Question: By how much did ACME's revenue grow from 2022 to 2023?
You call search_filings(query="revenue 2023 2022", ticker="ACME") and read:
[D99999 | ACME] Revenue was $1,250.4 million in 2023 compared with $1,102.9 million in 2022.
You call run_python(code="answer = 1250.4 - 1102.9") and read: answer = 147.5
Answer: Revenue grew by $147.5 million, from $1,102.9 million in 2022 to $1,250.4 million in 2023.
Source: [D99999]""",
    False: """Example (an invented company):
User: Company ticker: ACME. Question: What was ACME's revenue in 2023?
You call search_filings(query="revenue 2023", ticker="ACME") and read:
[D99999 | ACME] Revenue was $1,250.4 million in 2023 compared with $1,102.9 million in 2022.
Answer: Revenue was $1,250.4 million in 2023.
Source: [D99999]""",
}


def system_prompt(calculator=True):
    tools_line = ("Tools: search_filings finds passages from the company's 10-K filing; run_python is a calculator."
                  if calculator else "Tool: search_filings finds passages from the company's 10-K filing.")
    return f"""You are a careful financial analyst answering questions about a company's 10-K filing.
{tools_line}
Rules:
1. Always call search_filings first. Never answer from memory.
{PROMPT_RULES[calculator]}
3. If the passages do not contain the answer, search again with different words (at most 3 searches). If it is
still not there, say that the filing passages do not contain the answer.
4. Finish with exactly two lines:
Answer: <one or two sentences with the figure and its unit>
Source: [<passage id>]

{EXAMPLE[calculator]}"""


NUDGE_SEARCH = "You must call search_filings before answering. Do not answer from memory."
NUDGE_CALC = ("Your answer contains a number that is not written in the retrieved passages. Compute it with "
              "run_python using the numbers from the passages, then give the final answer in the required format.")
NUDGE_CALC_Q = ("This question asks for a calculated figure (a change, total, ratio or percentage). Call run_python now "
                "with the numbers from the passages, for example run_python(code=\"answer = (a - b) / b * 100\") with "
                "a and b replaced by the figures. Give the final answer in the required format only after it returns.")
CALC_INTENT_RE = re.compile(
    r"\b(?:combined|total|sum|difference|chang(?:e|ed)|increas(?:e|ed)|decreas(?:e|ed)|grow|grew|growth|"
    r"declin(?:e|ed)|ratio|percent(?:age)?|share|proportion|by how much|how much (?:more|less|higher|lower))\b", re.I)
FINALIZE = ("Step limit reached. Do not call any tools. Give your best final answer now from the passages above, "
            "in the required two-line format.")


# %% [markdown]
# ### 5.9 Agent graph
#
# The LangGraph graph has four nodes:
# - **agent**: the model either calls tools (returned as structured `tool_calls` by Ollama) or answers
# - **tools**: runs the calls; errors come back as observations
# - **finalize**: after 6 steps, makes the model answer without tools
# - **guard**: checks a draft answer before accepting it. It nudges if the model never searched, or if the answer has a number but `run_python` was never called and either the number is in no retrieved passage or the question asks for a calculation. Each nudge fires at most once (search at most twice). The checks only use what is known at answer time, never the gold.

# %%
class AgentState(MessagesState):
    steps: int
    nudges: list


def answer_numbers_unsupported(text, passages):
    pool = [n["value"] for p in passages for n in extract_numbers(p)]
    return [n["value"] for n in extract_numbers(text)
            if not any(same_up_to_presentation(n["value"], v) for v in pool)]


def build_agent(ticker=None, calculator=True, use_filter=True, guard=True, temperature=0.0, seed=SEED):
    tools = [make_search_tool(ticker if use_filter else None, use_filter=use_filter)]
    if calculator:
        tools.append(run_python)
    llm = get_llm(CONFIG["agent_model"], temperature=temperature, seed=seed, num_ctx=CONFIG["agent_num_ctx"])
    model, plain = llm.bind_tools(tools), llm
    sys_msg = SystemMessage(system_prompt(calculator))

    def agent(state: AgentState):
        reply = model.invoke([sys_msg] + state["messages"])
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    def route_after_agent(state: AgentState):
        if state["messages"][-1].tool_calls:
            return "tools" if state["steps"] < CONFIG["max_steps"] else "finalize"
        return "guard"

    def finalize(state: AgentState):
        reply = plain.invoke([sys_msg] + state["messages"] + [HumanMessage(FINALIZE)])
        return {"messages": [reply], "nudges": state.get("nudges", []) + ["step_limit"]}

    def guard_node(state: AgentState):
        nudges = state.get("nudges", [])
        msgs = state["messages"]
        passages = [m.content for m in msgs if isinstance(m, ToolMessage) and m.name == "search_filings"]
        used_calc = any(m.name == "run_python" for m in msgs if isinstance(m, ToolMessage))
        if guard and not passages and nudges.count("search") < 2:
            return Command(goto="agent", update={"messages": [HumanMessage(NUDGE_SEARCH)], "nudges": nudges + ["search"]})
        draft = msgs[-1].content.split("Source:")[0]
        if guard and calculator and not used_calc and passages and extract_numbers(draft):
            if "calc" not in nudges and answer_numbers_unsupported(draft, passages):
                return Command(goto="agent", update={"messages": [HumanMessage(NUDGE_CALC)], "nudges": nudges + ["calc"]})
            if not {"calc", "calc_question"} & set(nudges) and CALC_INTENT_RE.search(msgs[0].content.split("Question:")[-1]):
                return Command(goto="agent", update={"messages": [HumanMessage(NUDGE_CALC_Q)],
                                                     "nudges": nudges + ["calc_question"]})
        return Command(goto=END)

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_node("finalize", finalize)
    g.add_node("guard", guard_node, destinations=("agent", END))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "finalize", "guard"])
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile()


# %% [markdown]
# ### 5.10 Running the agent
#
# `run_agent` builds the graph for one question, runs it, and returns the answer, the cited passage, every tool call, the retrieved passage ids, nudges, LLM calls, tokens and time. The cell prints the graph in Mermaid form; `show_trace` prints a run step by step.

# %%
ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n\s*Source:|$)", re.S)
SOURCE_RE = re.compile(r"Source:\s*\[?\s*(D\d{5})")


def run_agent(question, ticker=None, **agent_kw):
    app = build_agent(ticker=ticker, **agent_kw)
    user = f"Company ticker: {ticker}\nQuestion: {question}" if ticker and agent_kw.get("use_filter", True) else question
    t0 = time.time()
    out = app.invoke({"messages": [HumanMessage(user)], "steps": 0, "nudges": []},
                     {"recursion_limit": 4 * CONFIG["max_steps"] + 10})
    latency = time.time() - t0
    msgs = out["messages"]
    final = msgs[-1].content.strip()
    m_ans, m_src = ANSWER_RE.search(final), SOURCE_RE.search(final)
    calls = [c for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    usage = [m.usage_metadata or {} for m in msgs if isinstance(m, AIMessage)]
    return {
        "answer": m_ans.group(1).strip() if m_ans else final,
        "raw_final": final,
        "cited_doc": m_src.group(1) if m_src else None,
        "tool_calls": [{"name": c["name"], "args": c["args"]} for c in calls],
        "retrieved_docs": sorted({d for m in tool_msgs if m.name == "search_filings" for d in PASSAGE_ID_RE.findall(m.content)}),
        "tool_errors": sum(m.content.startswith("Error") for m in tool_msgs),
        "llm_calls": sum(isinstance(m, AIMessage) for m in msgs),
        "nudges": out.get("nudges", []),
        "steps": out.get("steps", 0),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "latency_s": round(latency, 2),
        "trace": [{"role": type(m).__name__, "content": m.content[:1500],
                   "tool_calls": [{"name": c["name"], "args": c["args"]} for c in (getattr(m, "tool_calls", None) or [])],
                   "name": getattr(m, "name", None)} for m in msgs],
    }


print(build_agent(ticker="DAL").get_graph().draw_mermaid())


def show_trace(rec):
    for step in rec["trace"]:
        if step["role"] == "HumanMessage":
            print(f"USER   {step['content'][:300]}")
        elif step["role"] == "AIMessage":
            for c in step["tool_calls"]:
                print(f"ACT    {c['name']}({json.dumps(c['args'])[:300]})")
            if step["content"]:
                print(f"MODEL  {step['content'][:400]}")
        elif step["role"] == "ToolMessage":
            print(f"OBS    [{step['name']}] {step['content'][:300]}")
    print(f"-> answer: {rec['answer']} | cited {rec['cited_doc']} | {rec['latency_s']}s, {rec['llm_calls']} LLM calls, "
          f"nudges {rec['nudges']}")


# %% [markdown]
# ### 5.11 Agent on dev questions
#
# Two dev questions of each type, run live, with full traces. Prompts and guards were tuned on dev only. A live run can differ slightly from earlier runs (section 9).

# %%
dev_q = questions[questions.split == "dev"]
dev_sample = pd.concat([g.sample(2, random_state=SEED) for _, g in dev_q.groupby("qtype")])
dev_runs = []
for r in dev_sample.itertuples():
    rec = run_agent(r.question, r.ticker)
    dev_runs.append({"qtype": r.qtype, "calc_called": any(c["name"] == "run_python" for c in rec["tool_calls"]),
                     "gold_retrieved": bool(set(rec["retrieved_docs"]) & set(r.relevant_docs)),
                     "numeric_ok": (numeric_correct(rec["answer"], r.gold_numbers, r.gold_answer)
                                    if r.qtype in CALC_TYPES else None),
                     "latency_s": rec["latency_s"], "llm_calls": rec["llm_calls"]})
    print(f"\n=== [{r.qtype}] {r.question}\nGOLD   {r.gold_answer}")
    show_trace(rec)
display(pd.DataFrame(dev_runs))


# %% [markdown]
# ## 6. Evaluation
#
# ### 6.1 Conditions
#
# All conditions use the same questions, agent model, judge and metrics.
#
# | Condition | Description |
# |---|---|
# | B0 closed book | the model alone (4.5) |
# | B1 plain RAG | top 5 passages from `retrieve`, Module 3 RAG prompt |
# | B2 RAG + CoT | same passages; the model quotes the figures and does the arithmetic step by step in text |
# | S full agent | ReAct with `search_filings`, `run_python` and the guard |
# | A1 no ticker filter | S searching every company |
# | A2 no calculator | S with `search_filings` only |
#
# The cell defines the two RAG prompts, then runs (or loads) and scores every condition.

# %%
RAG_PLAIN = """Answer the question using ONLY the passages below. If the answer is not in the passages, say "I don't have that information."
Passages:
{passages}

Question: {question}
Finish with exactly two lines:
Answer: <one or two sentences with the figure and its unit>
Source: [<passage id>]"""
RAG_COT = """Answer the question using ONLY the passages below. If the answer is not in the passages, say "I don't have that information."
Passages:
{passages}

Question: {question}
Think step by step: first quote the figures you need from the passages, then do any arithmetic step by step.
Then finish with exactly two lines:
Answer: <one or two sentences with the figure and its unit>
Source: [<passage id>]"""


def format_passages(hits):
    return "\n\n".join(f"[{doc_ids[i]} | {doc_ticker[i]}] {doc_text[i][:MAX_PASSAGE_CHARS]}" for i, _ in hits)


def parse_final(text):
    answers = ANSWER_RE.findall(text)
    src = SOURCE_RE.findall(text)
    return (answers[-1].strip() if answers else text.strip()), (src[-1] if src else None)


def rag_answer(r, template):
    hits = retrieve(r.question, r.ticker, k=CONFIG["top_k"])
    text, cost = answer_from_llm(agent_llm, template.format(passages=format_passages(hits), question=r.question))
    answer, cited = parse_final(text)
    return {"answer": answer, "raw_final": text, "cited_doc": cited,
            "retrieved_docs": [doc_ids[i] for i, _ in hits], **cost}


def agent_condition(**kw):
    return lambda r: run_agent(r.question, r.ticker, **kw)


CONDITIONS = {
    "B0_closed_book": (closed_book, BASELINE_VERSION),
    "B1_plain_rag": (lambda r: rag_answer(r, RAG_PLAIN), BASELINE_VERSION),
    "B2_rag_cot": (lambda r: rag_answer(r, RAG_COT), BASELINE_VERSION),
    "S_agent": (agent_condition(), AGENT_VERSION),
    "A1_agent_no_ticker_filter": (agent_condition(use_filter=False), AGENT_VERSION),
    "A2_agent_no_calculator": (agent_condition(calculator=False), AGENT_VERSION),
}
for name, (fn, version) in CONDITIONS.items():
    if name not in scored:
        scored[name] = score(name, run_condition(name, fn, version))
print("scored:", list(scored))


# %% [markdown]
# ### 6.2 Results
#
# One row per condition: judge accuracy overall and by question type, numeric accuracy on calculations, calculator use, whether the gold passage was retrieved, citation accuracy, refusal rate and cost. The chart shows judge accuracy by question type.

# %%
TYPE_ORDER = ["text_lookup", "numeric_lookup", "calculation", "derived_calculation"]
HAS_TOOLS = {"S_agent", "A1_agent_no_ticker_filter", "A2_agent_no_calculator"}


def summarize(name, p):
    acc = lambda s: s.astype(float).mean()
    calc, look = p[p.qtype.isin(CALC_TYPES)], p[~p.qtype.isin(CALC_TYPES)]
    row = {"condition": name, "judge acc": acc(p.judge_ok)}
    row |= {f"judge: {t}": acc(p.loc[p.qtype == t, "judge_ok"]) for t in TYPE_ORDER}
    row["numeric acc (calc)"] = acc(calc.numeric_ok)
    row["calculator used (calc)"] = calc.calc_called.mean() if name in HAS_TOOLS else np.nan
    row["calculator used (lookup)"] = look.calc_called.mean() if name in HAS_TOOLS else np.nan
    row["gold passage retrieved"] = p.gold_retrieved.astype(float).mean()
    row["citation acc"] = p.cited_ok.mean() if name != "B0_closed_book" else np.nan
    row["refusal rate"] = p.refusal.mean()
    row["s / question"] = p.latency_s.mean()
    row["LLM calls / q"] = p.llm_calls.mean()
    row["tokens / q"] = (p.input_tokens + p.output_tokens).mean()
    row["no judge verdict"] = int(p.judge_ok.isna().sum())
    return row


results = pd.DataFrame([summarize(n, p) for n, p in scored.items()]).set_index("condition")
display(results.round(3))

fig, ax = plt.subplots(figsize=(12, 4))
results[[f"judge: {t}" for t in TYPE_ORDER]].rename(columns=lambda c: c.replace("judge: ", "")).plot.bar(ax=ax, width=0.8)
ax.set(title=f"Judge accuracy by question type ({'quick subset' if QUICK_RUN else 'test set'}, n={len(eval_q)})",
       ylabel="accuracy", ylim=(0, 1))
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 6.3 Uncertainty and paired comparisons
#
# With a few hundred questions, a gap of a few points can be noise. Each accuracy gets a 95% bootstrap interval (10,000 resamples of the questions). Since every condition answers the same questions, S is compared with each other condition pairwise: a bootstrap interval on the per-question difference, plus how many questions only one of the two got right. A difference whose interval includes 0 could be noise. The last table checks the targets from section 1.

# %%
N_BOOT = 10_000
boot_rng = np.random.default_rng(SEED)


def in_group(p, group):
    if group == "all":
        return p
    return p[p.qtype.isin(CALC_TYPES)] if group == "calculation" else p[~p.qtype.isin(CALC_TYPES)]


def outcome(p, col):
    s = p.set_index("qid")[col]
    return s[s.notna()].astype(float)


def boot_ci(x):
    x = np.asarray(x, dtype=float)
    means = x[boot_rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def fmt_ci(x):
    lo, hi = boot_ci(x)
    return f"{np.mean(x):.3f} [{lo:.3f}, {hi:.3f}]"


METRICS = [("judge, all", "judge_ok", "all"), ("judge, lookup", "judge_ok", "lookup"),
           ("judge, calculation", "judge_ok", "calculation"), ("numeric, calculation", "numeric_ok", "calculation")]
ci_table = pd.DataFrame({name: {m: fmt_ci(outcome(in_group(p, g), col)) for m, col, g in METRICS}
                         for name, p in scored.items()}).T
ci_table.index.name = "condition (accuracy [95% CI])"
display(ci_table)

paired_rows = []
for other in ("B2_rag_cot", "B1_plain_rag", "A2_agent_no_calculator", "A1_agent_no_ticker_filter"):
    for m, col, g in METRICS:
        both = pd.concat([outcome(in_group(scored["S_agent"], g), col).rename("S"),
                          outcome(in_group(scored[other], g), col).rename("other")], axis=1, join="inner")
        diff = both.S - both.other
        lo, hi = boot_ci(diff)
        s_only, other_only = int((diff > 0).sum()), int((diff < 0).sum())
        paired_rows.append({"S vs": other, "metric": m, "n": len(both), "S − other": round(diff.mean(), 3),
                            "95% CI": f"[{lo:.3f}, {hi:.3f}]", "only S right": s_only,
                            "only other right": other_only})
paired = pd.DataFrame(paired_rows).set_index(["S vs", "metric"])
display(paired)

r5 = retrieval_table.set_index(["retriever", "ticker filter"]).loc[("hybrid RRF + rerank", "on"), "R@5"]
num_gap = paired.loc[("B2_rag_cot", "numeric, calculation"), "S − other"]
look_gap = paired.loc[("B1_plain_rag", "judge, lookup"), "S − other"]
calc_use = results.loc["S_agent", "calculator used (calc)"]
targets = pd.DataFrame([
    ("retrieval Recall@5 (hybrid + rerank, filter on)", "≥ 0.90", r5, r5 >= 0.90),
    ("numeric acc on calculation, S − B2", "≥ +0.15", num_gap, num_gap >= 0.15),
    ("judge acc on lookup, S − B1", "≥ −0.03", look_gap, look_gap >= -0.03),
    ("calculator used on calculation questions (S)", "≥ 0.90", calc_use, calc_use >= 0.90),
], columns=["target", "goal", "result", "met"]).set_index("target")
display(targets.round(3))


# %% [markdown]
# ### 6.4 Judge validation
#
# A stratified sample of answers from B1, B2 and S is labelled without seeing the judge's verdict (labels as in 3.6). We report agreement and Cohen's κ for the combined verdict and for `phi4-mini` alone. Answers to questions rejected in review are not labelled. A label for an answer the predictions no longer contain stops the notebook.

# %%
val_pool = pd.concat([scored[c].assign(condition=c) for c in ("B1_plain_rag", "B2_rag_cot", "S_agent")])
val_pool["answer_key"] = ["h" + hashlib.sha1(a.encode()).hexdigest()[:12] for a in val_pool.answer]
val_sample = pd.concat([g.sample(min(20, len(g)), random_state=SEED) for _, g in val_pool.groupby("condition")])
export_once(val_sample[["condition", "qid", "answer_key", "question", "gold_answer", "answer"]].assign(human_correct=""),
            "judge_validation_to_label.csv")
val_path = LABELS / "judge_validation_labels.csv"
if val_path.exists():
    from sklearn.metrics import cohen_kappa_score
    hv = pd.read_csv(val_path)
    hv = hv[hv.human_correct.astype(str).str.strip().str.lower().isin(["yes", "no"])]
    hv["human"] = hv.human_correct.str.strip().str.lower() == "yes"
    n_labelled = len(hv)
    hv = hv.drop(columns="answer").merge(val_pool[["condition", "qid", "answer_key", "judge_ok", "llm_judge_ok"]],
                                         on=["condition", "qid", "answer_key"])
    if len(hv) != n_labelled:
        raise RuntimeError(f"{n_labelled - len(hv)} labels in {val_path} are for answers that the current "
                           "predictions no longer contain (the system changed after labelling): re-label them")
    for col in ("judge_ok", "llm_judge_ok"):
        ok = hv[col].notna()
        print(f"{col:13s} vs labels on {int(ok.sum())} answers: agreement {np.mean(hv.human[ok] == hv[col][ok].astype(bool)):.1%}, "
              f"kappa {cohen_kappa_score(hv.human[ok], hv[col][ok].astype(bool)):.2f}")
else:
    print(f"PENDING: {val_path} not provided yet, so judge agreement is not computed")


# %% [markdown]
# ### 6.5 Self-consistency (stretch goal)
#
# The agent runs 3 times per calculation question (temperature 0.7, different seeds). We vote on the first figure in each answer, rounded to 3 significant digits, and keep the winning run's answer and citation. Cost is the sum of the three runs. The table compares it with the single greedy run.

# %%
def vote_key(answer):
    nums = extract_numbers(answer)
    return float(f"{abs(nums[0]['value']):.3g}") if nums else None


def self_consistency(r, n=3):
    runs = [run_agent(r.question, r.ticker, temperature=0.7, seed=SEED + i) for i in range(1, n + 1)]
    keys = [vote_key(x["answer"]) for x in runs]
    counts = Counter(k for k in keys if k is not None)
    winner = max(counts, key=lambda k: (counts[k], -keys.index(k))) if counts else None
    chosen = runs[keys.index(winner)] if winner is not None else runs[0]
    return {**chosen, "votes": keys, "latency_s": round(sum(x["latency_s"] for x in runs), 2),
            "llm_calls": sum(x["llm_calls"] for x in runs), "input_tokens": sum(x["input_tokens"] for x in runs),
            "output_tokens": sum(x["output_tokens"] for x in runs)}


calc_q = eval_q[eval_q.qtype.isin(CALC_TYPES)]
scored["SC3_agent"] = score("SC3_agent", run_condition("SC3_agent", self_consistency, AGENT_VERSION, calc_q), calc_q)
s_calc = scored["S_agent"][scored["S_agent"].qtype.isin(CALC_TYPES)]
sc_table = pd.DataFrame([{"condition": n, "judge acc": p.judge_ok.astype(float).mean(),
                          "numeric acc": p.numeric_ok.astype(float).mean(), "s / question": p.latency_s.mean(),
                          "LLM calls / q": p.llm_calls.mean(), "tokens / q": (p.input_tokens + p.output_tokens).mean()}
                         for n, p in (("S_agent (greedy, 1 run)", s_calc), ("SC3_agent (3 runs, vote)", scored["SC3_agent"]))])
display(sc_table.set_index("condition").round(3))


# %% [markdown]
# ### 6.6 Results at a glance
#
# The numbers the findings below refer to, in one table: judge accuracy overall and on lookups, numeric accuracy on calculations, how often the calculator was used, and tokens per question as a measure of cost.

# %%
look_acc = {n: scored[n].loc[~scored[n].qtype.isin(CALC_TYPES), "judge_ok"].astype(float).mean() for n in results.index}
glance = pd.DataFrame({
    "overall": results["judge acc"],
    "lookups": pd.Series(look_acc),
    "calculations (numeric)": results["numeric acc (calc)"],
    "calculator used": results["calculator used (calc)"],
    "tokens / question": results["tokens / q"].round(0),
})
display(glance.round(3))


# %% [markdown]
# ### 6.7 Findings
#
# Test set after verification: 318 questions (200 lookups, 42 real and 76 derived calculations). Differences are S minus the other condition, with 95% bootstrap intervals.
#
# - Overall S is level with RAG + CoT: 0.852 vs 0.849 (difference +0.003, 95% CI −0.041 to 0.047), and with A2 (0.855). It is 4 points above plain RAG, within noise (CI −0.006 to 0.088). So the answer to the research question is no.
# - On lookups S is best: 0.890 vs 0.825 (B1) and 0.835 (B2); both intervals lie just above 0 (CI 0.005 to 0.125 and 0.005 to 0.105). The lookup target is met.
# - On calculations S is worse than RAG + CoT: numeric accuracy 0.805 vs 0.898 (−0.093, CI −0.169 to −0.017). The +15 point target is missed by a wide margin.
# - The calculator is called on 66% of calculation questions (target 90%). Removing it costs 4 points, within noise (CI −0.017 to 0.110). Section 7 shows why.
# - The ticker filter matters most. Without it accuracy drops 20 points (0.852 to 0.654, CI 0.148 to 0.252) because answers come from the wrong company. A1's latency is inflated by a GPU memory slowdown during that run.
# - Self-consistency lifts numeric accuracy from 0.805 to 0.839 for 2.9× the calls and tokens.
# - S uses about 3 calls and 3,100 tokens per question, B2 one call and 640 tokens.
# - The combined judge agrees with the labels on 93% of answers (κ = 0.73); `phi4-mini` alone on 89% (κ = 0.50).

# %% [markdown]
# ## 7. Error Analysis
#
# ### 7.1 Failure categories
#
# Every wrong S answer gets an automatic category from its trace: retrieval miss, refused (the right passage was retrieved), calculator not called, wrong figure or formula, wrong figure (lookup), step limit, or judge disagrees (rules passed, LLM judge said no). The failures are exported, then each is checked, re-categorised where needed and given a fix (`failures_categorized__test.csv`).

# %%
def auto_category(r):
    if "step_limit" in r.nudges:
        return "step limit"
    if r.gold_retrieved is False:
        return "retrieval miss"
    if r.refusal:
        return "refused (passage was retrieved)"
    if r.qtype in CALC_TYPES and not r.calc_called:
        return "calculator not called"
    if r.qtype in CALC_TYPES:
        return "wrong figure or formula"
    if r.fail_reason == "number mismatch":
        return "wrong figure (lookup)"
    return "judge disagrees (text)"


s_res = scored["S_agent"]
failures = s_res[s_res.judge_ok.map(lambda v: v is False)].copy()
failures["auto_category"] = [auto_category(r) for r in failures.itertuples()]
failures["answer_key"] = ["h" + hashlib.sha1(a.encode()).hexdigest()[:12] for a in failures.answer]
print(f"S failures: {len(failures)} of {len(s_res)}")
display(failures.auto_category.value_counts().rename("failures").to_frame())
pd.set_option("display.max_colwidth", 160)
display(failures[["qid", "qtype", "question", "gold_answer", "answer", "auto_category"]].head(25))
eval_tag = "quick" if QUICK_RUN else "test"
export_once(failures[["qid", "answer_key", "qtype", "question", "gold_answer", "answer", "auto_category"]]
            .assign(human_category="", what_to_change=""), f"failures_to_categorize__{eval_tag}.csv")
cat_path = LABELS / f"failures_categorized__{eval_tag}.csv"
if cat_path.exists():
    fc = pd.read_csv(cat_path)
    stale = set(zip(fc.qid, fc.answer_key)) - set(zip(failures.qid, failures.answer_key))
    if stale:
        raise RuntimeError(f"{len(stale)} rows of {cat_path} describe answers the current predictions no longer "
                           "contain (the system changed after categorising): re-categorise them")
    display(fc.human_category.value_counts().rename("failures (reviewed categories)").to_frame())
else:
    print(f"PENDING: {cat_path} not provided yet")


# %% [markdown]
# ### 7.2 What the failures show
#
# 47 of 318 S answers are wrong.
#
# - 14 are not really system errors. 8 come from ambiguous questions or golds, e.g. a "percentage decrease" between two margins where the gold gives the point change and the agent the relative change. 6 are correct answers marked wrong, e.g. a correct difference where the gold lists both values, or 1.014 against a gold of 1.01.
# - 12 are on the calculator path: not called (5, usually copying a rounded figure the passage states or adding in its head), wrong operands (4), or the tool result reported wrongly (3; once the raw `run_python(...)` call became the answer).
# - 13 are reading errors: wrong passage or figure (8) and incomplete answers (5). Two of the incomplete ones happened because the calculator nudge pushed a lookup into arithmetic.
# - 6 are search failures. In 4 the agent never searched despite two nudges; in 2 the gold passage wasn't retrieved. 2 more refused although the passage was there.
#
# First fixes: force a search on step one, send any arithmetic in the final answer through `run_python` and execute tool-call text instead of returning it, and only nudge for the calculator when the question asks for a number.

# %% [markdown]
# ## 8. Demo
#
# `ask(question, ticker, gold)` runs the full system live and prints each step: `USER` is the question or a guard nudge, `ACT` a tool call, `OBS` what the tool returned, `MODEL` the model's text. It ends with the answer next to the gold answer, and the cited passage. Both examples are dev questions picked before any test result was seen. Each takes about 10 seconds.
#
# ### 8.1 Success
#
# Hasbro's goodwill impairment plus loss on disposal (gold $1,499.0 million). The model adds $960.0M and $539.0M in its head and writes $1,500.0 million. The guard sees a number no passage states, `run_python` returns 1,499 and the answer is fixed.

# %%
def ask(question: str, ticker: str = None, gold: str = None):
    rec = run_agent(question, ticker)
    show_trace(rec)
    if gold:
        print(f"\nANSWER {rec['answer']}\nGOLD   {gold}")
    if rec["cited_doc"] in row_of:
        print(f"\ncited passage {rec['cited_doc']}: {doc_text[row_of[rec['cited_doc']]]}")
    return rec


demo_success = ask("What were the non-cash goodwill impairment charge and the loss on disposal of business combined?",
                   ticker="HAS", gold="$1,499.0 million")


# %% [markdown]
# ### 8.2 Failure
#
# How much did NVIDIA's Compute & Networking gross margin decrease? The filing gives only the reason (inventory provisions), not a figure. The agent finds the passage but answers "decreased by 36%", which is the segment's revenue growth. The calculator nudge doesn't change it. Asked "how much", the model would rather give a wrong number than none, and only the citation lets a reader catch it.

# %%
demo_failure = ask("How much did the Compute & Networking segment's gross margin decrease in fiscal year 2023?",
                   ticker="NVDA", gold="The gross margin of the Compute & Networking segment decreased during fiscal year 2023, "
                        "primarily because of inventory provisions.")


# %% [markdown]
# ### 8.3 Spare question
#
# A third dev question, if time allows or the judges want another run: what share of Apple's notes falls due within 12 months. The agent should search, call `run_python` with 9.9 / 106.6 and answer about 9.3%.

# %%
demo_spare = ask("What percentage of total Notes did the $9.9 billion payable within 12 months represent?",
                 ticker="AAPL", gold="9.3%")


# %% [markdown]
# ## 9. Limitations and Ethics
#
# - **Failures** (section 7). The model still picks the wrong figure: wrong year, wrong line item, thousands read as millions. The calculator fixes arithmetic, not extraction, which matches the Program-of-Thoughts error analysis. Questions that only say "the company" are hard to retrieve for, and without the ticker filter answers can come from another company.
# - **Data.** The Q&A pairs were LLM-generated from one 2023 10-K for each of 69 mostly large US companies. About 50 gold answers contain facts not in their passage. The derived questions are synthetic, and 82 of 169 were rejected on review. Results may not carry over to full filings, other years, tables or smaller firms.
# - **Harm.** A user could trust a confident wrong figure. Every answer cites its passage. This is a reading aid, not financial advice.
# - **Security.** The calculator runs model-written code. The allow-list and separate process block imports, file access and hangs, but a production system would need a container. Passages could carry prompt injections; a 4B model can be steered by them.
# - **Privacy and misuse.** Filings are public and no personal data is used. The same pipeline could mass-produce plausible but wrong financial summaries, another reason for citations.
# - **Evaluation.** Each condition was run once on 318 questions, so small gaps are noise (6.3). The judge is a small LLM checked against the 3.6 labels. Numeric matching has a 1% tolerance, ignores sign, and can't check scale when the gold has none (gold "$1.60" matches "$1.60 billion").
# - **Run-to-run variation.** Ollama at T = 0 with a fixed seed is not fully deterministic. A dev question on Coca-Cola's interest paid was right in some runs and wrong in others. Reported numbers are the saved predictions.

# %% [markdown]
# ## 10. References
#
# - Dataset: `virattt/financial-qa-10K` (Hugging Face), revision `30e48b49ef40389cce6488b0bddef7d6f00f4c7a`.
# - Models: Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct`, agent and baselines); Phi-4-mini-instruct (`phi4-mini`, judge and question generator); BAAI/bge-small-en-v1.5 and all-MiniLM-L6-v2 (embeddings); cross-encoder/ms-marco-MiniLM-L-6-v2 (re-ranker).
# - Papers: Yao et al. 2023, ReAct. Chen et al. 2023, Program of Thoughts. Gao et al. 2023, PAL. Wang et al. 2023, Self-Consistency. Chen et al. 2021, FinQA. Zhu et al. 2021, TAT-QA. Islam et al. 2023, FinanceBench. Cormack et al. 2009, Reciprocal Rank Fusion. Robertson and Zaragoza 2009, BM25. Xiao et al. 2023, C-Pack (BGE). Zheng et al. 2023, Judging LLM-as-a-Judge.
# - Course labs: Module 2 `4_Structured_Generation_Lab` (PAL, self-consistency, structured output); Module 3 `1_RAG_Pipeline_Lab` (BM25, RRF, re-ranking, Recall@k, MRR); Module 4 `1_Multi_Agent_System_End_to_End` and `Agentic_app` (tools, `run_python`, ReAct graph, tool nudge, `check_setup.py`); Module 4 MCP lab (AST calculator).
# - Software: Ollama, LangGraph, LangChain, FAISS, rank-bm25, sentence-transformers, Hugging Face `datasets`, pandas, matplotlib, jupytext.

# %%
