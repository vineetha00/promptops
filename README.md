# promptops

[![PyPI version](https://img.shields.io/pypi/v/promptops.svg)](https://pypi.org/project/promptops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> *I changed one word in a system prompt and broke production. I had no diff, no rollback, no way to know which version was running. So I built Git for prompts.*

**promptops** is a framework-agnostic, Git-native CLI that version-controls LLM prompts. Semantic diff. Thompson Sampling A/B routing. LLM-as-judge regression eval. MCP server. Drop-in GitHub Actions gate.

```bash
pip install promptops
```

![promptops demo — init, commit, semantic diff, Thompson Sampling A/B test, rollback](docs/demo.gif)

---

## 60-second quickstart

```bash
promptops init
promptops commit payment-agent "You are a payment assistant. Be precise." --tag v1
promptops commit payment-agent "You are a helpful payment assistant. Be precise and friendly." --tag v2
promptops diff payment-agent v1..v2
promptops ab-test payment-agent v1 v2 --metric quality
promptops rollback payment-agent --to v1
```

That's six commands from zero to versioned, A/B-tested, instantly-rollback-able prompts.

---

## Architecture

```
        ┌──────────────────────────────────────────────────────────┐
        │                       promptops                          │
        │                                                          │
  CLI ──┤  store.py        diff.py        ab.py         eval.py    │
        │  ───────         ───────        ─────         ───────    │
        │  .promptops/     sentence-      Thompson      Claude     │
        │  registry.json   transformers   Sampling      as judge   │
        │  prompts/*.txt   cosine sim     Beta(α,β)     1-5 score  │
        │       │              │            │              │       │
        │       └──────┬───────┴────────────┴──────────────┘       │
        │              │                                           │
        │       mcp_server.py  ◄───── stdio ─────► Claude Code     │
        │                                                          │
        │       prompt-regression.yml  ◄────►  GitHub Actions PR    │
        └──────────────────────────────────────────────────────────┘
```

Each prompt is a plain `.txt` file under `.promptops/prompts/`. Metadata — versions, scores, Beta posteriors, traffic — lives in `.promptops/registry.json`. Git already knows how to diff, branch, and merge that.

---

## Commands

| Command | What it does |
|---|---|
| `promptops init` | Create `.promptops/` in the current directory. |
| `promptops commit NAME TEXT --tag v1` | Commit a new version of a prompt. |
| `promptops log NAME` | Version history with scores, traffic, and active marker. |
| `promptops diff NAME v1..v2` | Semantic diff: cosine similarity + sentence-level changes. |
| `promptops ab-test NAME v1 v2 --metric quality` | Thompson Sampling routing; declares a winner at p > 0.95. |
| `promptops rollback NAME --to v1` | Instant rollback to a prior version. |
| `promptops status` | All tracked prompts, active versions, total traffic. |
| `promptops eval --compare main..branch --fail-on-regression` | LLM-as-judge eval; CI gate. |
| `promptops serve --port 3333` | Start the MCP server (stdio). |

---

## How it works

**Semantic diff** — both prompt versions are embedded with `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity gives an overall semantic distance. A sentence-level diff highlights what actually changed, colored by `rich`.

**Thompson Sampling A/B** — every version maintains a `Beta(α, β)` posterior over its success rate. A routing call samples once from each posterior and picks the highest. Wins bump α, losses bump β. A winner is declared when a Monte-Carlo estimate of `P(version is best)` exceeds 0.95 — i.e., statistically significant before declaring.

**LLM-as-judge eval** — `claude-sonnet-4-20250514` scores each version 1-5 on quality, relevance, and safety. Scores feed back into the A/B Beta posteriors and persist to `registry.json`. `--fail-on-regression` exits code 1 when quality drops more than 10%.

**MCP server** — stdio transport exposing three tools to any MCP client (Claude Code included):
- `get_active_version(prompt_name)` — current winner + text
- `list_prompts()` — all tracked prompts and versions
- `get_metrics(prompt_name)` — A/B scores, win rates, traffic splits

---

## vs. the alternatives

|  | promptops | Langfuse | LangSmith | Weights & Biases |
|---|:---:|:---:|:---:|:---:|
| Pip-installable, no account | ✅ | ❌ | ❌ | ❌ |
| Git-native storage (diffable, branchable) | ✅ | ❌ | ❌ | ❌ |
| Semantic diff between versions | ✅ | ❌ | partial | ❌ |
| Thompson Sampling A/B router | ✅ | ❌ | ❌ | ❌ |
| LLM-as-judge eval | ✅ | ✅ | ✅ | partial |
| MCP server out of the box | ✅ | ❌ | ❌ | ❌ |
| Framework-agnostic | ✅ | ✅ | weighted to LangChain | ✅ |
| Drop-in GitHub Actions gate | ✅ | ❌ | ❌ | ❌ |
| Local-first, no telemetry | ✅ | self-host | ❌ | ❌ |

---

## MCP integration with Claude Code

Add this to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "promptops": {
      "command": "promptops",
      "args": ["serve"]
    }
  }
}
```

Now Claude Code can `list_prompts`, fetch the active version, and read live A/B metrics directly from your repo.

---

## GitHub Actions: drop-in regression gate

Copy `.github/workflows/prompt-regression.yml` from this repo. Add `ANTHROPIC_API_KEY` to your repo secrets. Every PR touching prompts now runs:

```bash
promptops eval --compare origin/main..HEAD --fail-on-regression
```

A >10% drop in any prompt's eval score blocks the merge.

---

## Citation

If you use promptops in academic work, please cite the `CITATION.cff` in this repo.

## License

MIT — see [LICENSE](LICENSE).
