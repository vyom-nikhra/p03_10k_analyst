"""Run evaluation conditions on the DEV split with the notebook's own code. This is how the agent was tuned.

    python tools/dev_eval.py P03_Financial_10K_Analyst.py artifacts/dev_tuning S_agent [B2_rag_cot ...]

Run from the project folder with the venv active. The script executes the notebook's code cells up to the
definition of CONDITIONS in section 6. It skips the retrieval grid, the 5b dev check and the section 6 loop, so
nothing runs on test questions. Predictions and judge verdicts for dev questions go to OUT_DIR, never to
artifacts/preds or artifacts/judge.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

path, out_dir, conds = sys.argv[1], Path(sys.argv[2]).resolve(), sys.argv[3:]
if not Path("artifacts/questions.jsonl").exists():
    raise SystemExit("run from the project folder")
src = Path(path).read_text(encoding="utf-8")
cells = [c for c in src.split("\n# %%") if not c.lstrip().startswith("[markdown]")]
ns = {"__name__": "__main__"}
SKIP = ("RETRIEVERS = {", "dev_sample = pd.concat", "%pip install")
for c in cells:
    if "RAG_PLAIN =" in c:
        head, sep, _ = c.partition("\nfor name, ")      # stop before the loop that runs every condition on eval_q
        if not sep or "CONDITIONS = {" not in head:
            raise SystemExit("could not find the CONDITIONS loop in the section 6 cell; update tools/dev_eval.py")
        exec(compile(head, path, "exec"), ns)
        break
    if any(s in c for s in SKIP):
        continue
    exec(compile(c, path, "exec"), ns)

for sub in ("preds", "judge"):
    (out_dir / sub).mkdir(parents=True, exist_ok=True)
ns["ART"] = out_dir
ns["judge_store"] = ns["JsonlStore"](out_dir / "judge" / ns["judge_store"].path.name)   # was opened under artifacts/
dev_q = ns["questions"][ns["questions"].split == "dev"].reset_index(drop=True)
print("dev questions:", len(dev_q))
for name in conds:
    fn, version = ns["CONDITIONS"][name]
    p = ns["score"](name, ns["run_condition"](name, fn, version, qs=dev_q), dev_q)
    print(f"{name} ({version}) on dev:")
    print(p.groupby("qtype")[["judge_ok", "numeric_ok", "calc_called", "refusal"]]
          .agg(lambda s: s.astype(float).mean()).round(3).to_string())
