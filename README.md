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
| `promptops eval --compare main..branch --fail-on-regression` | Run changed prompts over their eval set, judge the outputs against `main`; CI gate. |
| `promptops serve --port 3333` | Start the MCP server (stdio). |

---

## How it works

**Semantic diff** — both prompt versions are embedded with `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity gives an overall semantic distance. A sentence-level diff highlights what actually changed, colored by `rich`.

**Thompson Sampling A/B** — every version maintains a `Beta(α, β)` posterior over its success rate. A routing call samples once from each posterior and picks the highest. Wins bump α, losses bump β. A winner is declared when a Monte-Carlo estimate of `P(version is best)` exceeds 0.95 — i.e., statistically significant before declaring.

**Output-based regression eval** — each prompt has a fixed eval set of test inputs. Both the baseline and the candidate prompt are run on every case (generator: `claude-opus-5-5`, override with `PROMPTOPS_GEN_MODEL` to match production), and `claude-sonnet-5-5` (`PROMPTOPS_JUDGE_MODEL`) scores each **output** 1-5 on quality, relevance, and safety, using a rubric with level anchors and few-shot calibration examples. Judge output is schema-constrained and validated. A version's score is the mean over the eval set. `promptops eval --prompt NAME` scores versions locally and feeds the A/B posteriors.

**Fails closed** — if generation or judging fails (no credentials, API error, malformed judge output) or the baseline can't be read, `eval` exits **2** with no scores. There is no fallback scorer. Exit codes: `0` pass, `1` regression (with `--fail-on-regression`), `2` eval could not run.

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

The baseline is **the active version as committed on `main`**, read with `git show`, and the eval set is also read from `main`. Nothing in the PR's checkout can change what it is measured against. A PR that edits an eval set gets a note, and the edit applies only after it merges. Candidates are the versions the PR adds or edits, plus its active version if that differs from `main`. A candidate blocks the merge only if two things are both true. Its mean score must drop more than 10% (`--threshold`). The drop must also pass a noise test: a one-sided paired t-test across the eval cases, at p < 0.05. A large mean drop driven by one noisy case is printed as a note and does not fail the check. The report also lists which models actually served the run, plus token usage. If the judge or generator model changes partway through a run, the run fails with exit 2.

Add an eval set for each prompt at `.promptops/evals/<prompt-name>.json` and merge it to `main` before you gate changes to that prompt. A changed prompt with no eval set on `main` fails the check:

```json
{
  "cases": [
    {"id": "refund-damaged", "input": "My order #4821 arrived damaged. Can I get a refund?",
     "expected": "Apologize, explain how to start the refund, do not promise a timeline."},
    {"id": "double-charge", "input": "I was charged twice.",
     "expected": "Explain how to report it; never ask for card details in chat."}
  ]
}
```

Use real traffic for the inputs, and aim for at least 10–20 cases. With a handful of cases, a single noisy judgement can move the mean past the threshold. Each case costs one generation and one judge call per version.

PRs from forks don't get repository secrets. On a fork PR the check fails with exit 2 instead of passing silently, and the message tells the contributor this is expected and that a maintainer needs to run the check. To make the gate hard to bypass, mark it as a required status check and protect `.github/workflows/` with CODEOWNERS. A PR can edit its own workflow file.

---

## Citation

If you use promptops in academic work, please cite the `CITATION.cff` in this repo.

## License

MIT — see [LICENSE](LICENSE).
