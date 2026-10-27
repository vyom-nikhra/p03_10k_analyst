# Logbook

An append-only, timestamped record of every atomic change made to this project. When you pick up work, read `PROGRESS.md` for the current status, then the tail of this file for what happened most recently. The design is in `PLAN.md`.

**Rules**
- Add an entry immediately after each change, not in batches at the end of a session.
- Use one line per atomic change, newest at the bottom, and never edit or delete old entries. To correct something, add a new entry.
- Format: `- YYYY-MM-DD HH:MM:SS +ZZZZ | who | files/area | what changed (and why, if not obvious)`
- Log runs and results too, for example "evaluation run started or finished", "Recall@5 = …" or "smoke test failed: …", with the artefact path.
- Commit each entry together with the change it describes. A commit needs no entry of its own, since `git log` holds the hash.

## Entries
- 2026-10-28 02:00:19 +0530 | Claude | `p03_10k_analyst/`, `artifacts/`, `PLAN.md` | Created the project folder and an empty `artifacts/`; copied the approved plan into `PLAN.md`.
- 2026-10-28 02:00:29 +0530 | Claude | `PROGRESS.md` | Created the progress log: phase status table, decisions, open items, environment notes.
- 2026-10-28 02:00:29 +0530 | Claude | `.gitignore` | Ignore `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` and `artifacts/cache/`.
- 2026-10-28 02:00:34 +0530 | Claude | git | Ran `git init`, set the repo-local author to Vyom Nikhra <vyomnikhra@gmail.com>, and made the first commit `3ab6b0e` ("Add project plan and progress log").
- 2026-10-28 02:01:19 +0530 | Claude | `logbook.md` | Created this logbook at Vyom's request and backfilled the entries above from file times and git.
- 2026-10-28 02:01:39 +0530 | Claude | `PROGRESS.md` | Header now points readers to `logbook.md` and states the rule that every change is logged immediately.
