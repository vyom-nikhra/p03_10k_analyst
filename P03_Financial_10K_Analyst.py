# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project P03: Financial 10-K Analyst. Grounded Answers with Exact Calculations
#
# **Team:** Vyom Nikhra, <member A>, <member B>, <member C>
#
# | Member | Contribution (sections, code, experiments) |
# |---|---|
# | Vyom Nikhra | System design and all code: retrieval (5a), tools and sandbox (5b), agent graph (5b), evaluation harness (6), demo (8) |
# | <member A> | Hand-verified the derived calculation set and hand-labelled 100 question-type tags (3); data and design slides |
# | <member B> | Hand-labelled answers for judge validation (6); results and ablation slides |
# | <member C> | Categorised the failure cases (7); wrote limitations and ethics (9); error-analysis slides |

# %% [markdown]
# ## 1. Problem Statement
#
# - **Problem (our own words):** Analysts ask questions about annual reports (10-K filings) that mix
#   *lookup* ("what was services revenue?") with *arithmetic* ("what share of revenue came from
#   services?"). Small language models misread figures from long passages and make token-level
#   arithmetic slips. We build a single ReAct agent that retrieves the right filing passage, extracts the
#   numbers, and computes any result with a sandboxed Python tool, so numbers are never guessed.
# - **Research question:** Does a ≤4B model that delegates retrieval (hybrid RAG with a ticker filter) and
#   arithmetic (a sandboxed Python tool) through a ReAct loop beat plain RAG and RAG + chain-of-thought on
#   *calculation* questions without losing accuracy on *lookup* questions, and at what cost?
# - **What success looks like (targets fixed before any test results):**
#   - retrieval Recall@5 ≥ 0.90 with the ticker filter;
#   - on calculation questions, numeric accuracy at least 15 points above RAG + CoT;
#   - on lookup questions, judge accuracy within 3 points of plain RAG;
#   - the calculator is called on ≥ 90% of calculation questions.
