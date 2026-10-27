"""Check that this machine can run the P03 notebook, and say how to fix anything that isn't ready.

    python check_setup.py          # all checks (first run downloads the dataset and two small HF models)
    python check_setup.py --no-llm # skip the Ollama checks

Model names and the Ollama address come from the environment, matching the notebook's CONFIG:
    AGENT_MODEL (default qwen3:4b-instruct), JUDGE_MODEL (default phi4-mini), OLLAMA_HOST (default http://localhost:11434)

Adapted from GenAI/Module 4/Agentic_app/check_setup.py (Module 4 lab).
"""
import ast
import json
import operator
import os
import sys
import urllib.request

AGENT_MODEL = os.environ.get("AGENT_MODEL", "qwen3:4b-instruct")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "phi4-mini")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if "://" not in OLLAMA_HOST:  # Ollama itself accepts a bare host:port in OLLAMA_HOST
    OLLAMA_HOST = "http://" + OLLAMA_HOST

OK, FAIL, INFO = "  [ OK ]", "  [FAIL]", "  [INFO]"
problems = []


def check(label, fn, fix):
    try:
        detail = fn()
        print(f"{OK} {label}" + (f" - {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"{FAIL} {label} - {type(e).__name__}: {e}\n         fix: {fix}")
        problems.append(label)
        return False


def python_version():
    assert sys.version_info >= (3, 10), f"found {sys.version.split()[0]}; langgraph needs 3.10+"
    return sys.version.split()[0]


def packages():
    import importlib.util
    mods = ("torch", "datasets", "sentence_transformers", "faiss", "rank_bm25", "sklearn",
            "langgraph", "langchain_core", "langchain_ollama", "pydantic", "pandas", "matplotlib")
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    assert not missing, f"missing: {', '.join(missing)}"


def device():
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"cuda: {p.name}, {p.total_memory / 2**30:.1f} GiB, torch {torch.__version__}"
    return f"cpu only (no CUDA build of torch or no GPU), torch {torch.__version__}"


def dataset():
    from datasets import load_dataset
    ds = load_dataset("virattt/financial-qa-10K")["train"]
    need = {"question", "answer", "context", "ticker", "filing"}
    assert need <= set(ds.column_names), f"columns are {ds.column_names}"
    return f"{ds.num_rows} rows, {len(set(ds['ticker']))} tickers, filing values {sorted(set(ds['filing']))}"


def embedder_and_reranker():
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
    v = emb.encode(["Represent this sentence for searching relevant passages: total revenue",
                    "Total revenue was $60.9 billion."], normalize_embeddings=True)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=dev)
    s = ce.predict([("total revenue", "Total revenue was $60.9 billion."), ("total revenue", "The CEO is Jensen Huang.")])
    assert s[0] > s[1], f"reranker scored the relevant passage lower: {s}"
    return f"bge-small dim {v.shape[1]}, cosine {float(v[0] @ v[1]):.2f}; reranker {s[0]:.1f} > {s[1]:.1f} ({dev})"


def ollama_server():
    try:
        version = json.load(urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=5))["version"]
        tags = json.load(urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5))
    except OSError as e:
        raise RuntimeError(f"Ollama not reachable at {OLLAMA_HOST} ({e})")
    pulled = {m["name"] for m in tags["models"]}
    missing = [m for m in (AGENT_MODEL, JUDGE_MODEL) if m not in pulled and f"{m}:latest" not in pulled]
    assert not missing, f"version {version}; not pulled: {missing} (have: {sorted(pulled)})"
    return f"version {version} at {OLLAMA_HOST}; {AGENT_MODEL} and {JUDGE_MODEL} pulled"


# Arithmetic-only evaluator from the Module 4 MCP lab (mcp_knowledge_server.py). The real sandboxed
# run_python tool is built in notebook section 5b; this is only enough to exercise a tool loop.
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / are allowed")


def _calculator_tool():
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression such as 847 * 23 + 156 and return the result."""
        return str(_safe_eval(ast.parse(expression, mode="eval").body))

    return calculator


def _llm(model, **kw):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0, num_predict=256, seed=42, **kw)


def agent_tool_call():
    reply = _llm(AGENT_MODEL).bind_tools([_calculator_tool()]).invoke(
        "Use the calculator tool to compute 1234.5 * 0.17. Do not compute it yourself.")
    assert reply.tool_calls, f"no tool call; replied: {reply.content[:120]!r}"
    call = reply.tool_calls[0]
    assert call["name"] == "calculator", f"called {call['name']}"
    return f"{call['name']}({json.dumps(call['args'])})"


def agent_reasoning_flag():
    # Open question from PLAN.md: does ChatOllama(reasoning=False) work for a model with no thinking mode?
    return _llm(AGENT_MODEL, reasoning=False).invoke("Reply with the single word: ready").content.strip()[:40]


def react_graph():
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    calc = _calculator_tool()
    model = _llm(AGENT_MODEL).bind_tools([calc])

    def call_model(state: MessagesState):
        sys_msg = SystemMessage("Use the calculator tool for ANY arithmetic. Never compute numbers yourself.")
        return {"messages": [model.invoke([sys_msg] + state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode([calc], handle_tool_errors=True))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    out = g.compile().invoke({"messages": [HumanMessage("What is 847 times 23, plus 156?")]}, {"recursion_limit": 8})
    called = [c["name"] for m in out["messages"] for c in (getattr(m, "tool_calls", None) or [])]
    final = out["messages"][-1].content
    assert "calculator" in called, f"calculator never called; final: {final[:120]!r}"
    assert "19637" in final.replace(",", ""), f"expected 19637 in final answer, got {final[:120]!r}"
    return f"tools called {called}; final answer contains 19637"


# Same rubric as the notebook's judge (section 6). A lenient "numbers may be rounded" wording let
# gemma3:4b accept a wrong number in the P0 probe; see logbook.md 2026-10-28.
JUDGE_TEMPLATE = (
    "You grade answers to questions about company annual reports.\n"
    "Question: {q}\nGold answer: {gold}\nCandidate answer: {pred}\n\n"
    "Rules: the candidate is correct only if it states the same key fact as the gold answer. "
    "Every key number must match the gold number after converting units (million/billion, percent/fraction); "
    "rounding to the precision shown in the gold answer is fine. A different number, the opposite direction "
    "(increase vs decrease), different entities, or no answer means NO. Extra correct detail is fine.\n"
    "Reply with exactly one word: YES or NO.")


def judge():
    llm = _llm(JUDGE_MODEL)
    q, gold = "What was total revenue in fiscal 2023?", "Total revenue for fiscal 2023 was $60.9 billion."
    same = llm.invoke(JUDGE_TEMPLATE.format(q=q, gold=gold, pred="Revenue was about $60.9B in FY2023.")).content
    diff = llm.invoke(JUDGE_TEMPLATE.format(q=q, gold=gold, pred="Revenue was $45.2 billion in FY2023.")).content
    same, diff = same.strip().upper(), diff.strip().upper()
    assert same.startswith("YES") and diff.startswith("NO"), f"judge said {same!r} for a match and {diff!r} for a mismatch"
    return f"YES for the matching answer, NO for the wrong one ({JUDGE_MODEL})"


if __name__ == "__main__":
    print("P03 Financial 10-K Analyst - setup check\n")
    if not check("Python 3.10 or newer", python_version, "create the venv with Python 3.10-3.12 (README, Setup)"):
        sys.exit(1)
    if not check("Python packages installed", packages, "pip install -r requirements.txt (install torch first, see README)"):
        sys.exit(1)
    check("Compute device", device, "for GPU, install the CUDA build of torch (README, Setup)")
    check("Dataset downloads from Hugging Face", dataset, "check the internet connection")
    check("Embedder and re-ranker load and score sensibly", embedder_and_reranker, "check the internet connection")
    if "--no-llm" not in sys.argv:
        if check("Ollama running with both models pulled", ollama_server,
                 f"install/update Ollama from https://ollama.com, then: ollama pull {AGENT_MODEL} && ollama pull {JUDGE_MODEL}"):
            check("Agent model emits a native tool call", agent_tool_call, "update Ollama; check the model tag supports tools")
            try:
                print(f"{INFO} reasoning=False accepted by {AGENT_MODEL} - replied {agent_reasoning_flag()!r}")
            except Exception as e:
                print(f"{INFO} reasoning=False rejected by {AGENT_MODEL} - {type(e).__name__}: {e} (omit it for this model)")
            check("LangGraph ReAct loop calls the tool and answers", react_graph, "see the error above; try re-running")
            check("Judge model separates a match from a mismatch", judge, "update Ollama; check the judge model tag")
    print("\nAll checks passed." if not problems else f"\n{len(problems)} problem(s) to fix: {', '.join(problems)}")
    sys.exit(1 if problems else 0)
