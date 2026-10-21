# promptops

[![Release](https://img.shields.io/github/v/release/vineetha00/promptops)](https://github.com/vineetha00/promptops/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> *I changed one word in a system prompt and broke production. I had no diff, no rollback, no way to know which version was running. So I built Git for prompts.*

**promptops** is a framework-agnostic, Git-native CLI that version-controls LLM prompts. Semantic diff. Thompson Sampling A/B routing. LLM-as-judge regression eval. MCP server. Drop-in GitHub Actions gate.

```bash
pip install "git+https://github.com/vineetha00/promptops@v0.3.0"
```

> **Don't `pip install promptops` from PyPI.** That name belongs to an unrelated project, which installs its own `promptops` command and Python package. This project's package is `promptops-git`, its command is `promptops-git`, and it imports as `promptops_git`. It isn't on PyPI yet, so install it from a GitHub release tag as shown above. Its data stays in your repo's `.promptops/` directory.

![promptops demo — init, commit, semantic diff, Thompson Sampling A/B test, rollback](docs/demo.gif)

---

## 60-second quickstart

```bash
promptops-git init
promptops-git commit payment-agent "You are a payment assistant. Be precise." --tag v1
promptops-git commit payment-agent "You are a helpful payment assistant. Be precise and friendly." --tag v2
promptops-git diff payment-agent v1..v2
promptops-git ab-test payment-agent v1 v2 --metric quality
promptops-git rollback payment-agent --to v1
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
| `promptops-git init` | Create `.promptops/` in the current directory. |
| `promptops-git commit NAME TEXT --tag v1` | Commit a new version of a prompt. |
| `promptops-git log NAME` | Version history with scores, traffic, and active marker. |
| `promptops-git diff NAME v1..v2` | Semantic diff: cosine similarity + sentence-level changes. |
| `promptops-git ab-test NAME v1 v2 --metric quality` | Thompson Sampling routing; declares a winner at p > 0.95. |
| `promptops-git rollback NAME --to v1` | Instant rollback to a prior version. |
| `promptops-git status` | All tracked prompts, active versions, total traffic. |
| `promptops-git eval --compare main..branch --fail-on-regression` | Run changed prompts over their eval set, judge the outputs against `main`; CI gate. |
| `promptops-git serve --port 3333` | Start the MCP server (stdio). |

---

## How it works

**Semantic diff** — both prompt versions are embedded with `sentence-transformers/all-MiniLM-L6-v2`. Cosine similarity gives an overall semantic distance. A sentence-level diff highlights what actually changed, colored by `rich`.

**Thompson Sampling A/B** — every version maintains a `Beta(α, β)` posterior over its success rate. A routing call samples once from each posterior and picks the highest. Wins bump α, losses bump β. A winner is declared when a Monte-Carlo estimate of `P(version is best)` exceeds 0.95 — i.e., statistically significant before declaring.

**Output-based regression eval** — each prompt has a fixed eval set of test inputs. Both the baseline and the candidate prompt are run on every case (generator: `claude-opus-5-5`, override with `PROMPTOPS_GEN_MODEL` to match production), and `claude-sonnet-5-5` (`PROMPTOPS_JUDGE_MODEL`) scores each **output** 1-5 on quality, relevance, and safety, using a rubric with level anchors and few-shot calibration examples. Judge output is schema-constrained and validated. A version's score is the mean over the eval set. `promptops-git eval --prompt NAME` scores versions locally and feeds the A/B posteriors.

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
      "command": "promptops-git",
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
promptops-git eval --compare origin/main..HEAD --fail-on-regression
```

The baseline is **the active version as committed on `main`**, read with `git show`, and the eval set is also read from `main`. Nothing in the PR's checkout can change what it is measured against. A PR that edits an eval set gets a note, and the edit applies only after it merges. Candidates are the versions the PR adds or edits, plus its active version if that differs from `main`. The report lists which models actually served the run, plus token usage. If the judge or generator model changes partway through a run, the run fails with exit 2.

### When does a change count as a regression?

**Prompt regressions are sparse and severe, not uniform and mild.** A bad prompt edit rarely makes every answer slightly worse. More often it breaks a few inputs completely and leaves the rest alone. In testing, a prompt that told the assistant to ask for card details made it refuse 2 of 6 ordinary requests. Those cases fell from about 4.7 to 1.0, while the other four cases were unaffected. A mean-based test can't see that shape, because the uneven drops look like noise to it. That run had a -19% mean drop at p = 0.08, and the mean rule alone passed it. So the gate has three triggers, and any one of them blocks the merge:

| Trigger | Fires when | Catches |
|---|---|---|
| **mean** | mean score drops more than 10% (`--threshold`) *and* a one-sided paired t-test across cases gives p < 0.05 | broad, uniform degradation |
| **collapse** | one case drops by ≥ 2.0 points (1–5 scale) | a prompt that breaks specific inputs |
| **floor** | one case scores 1 on any dimension (below 2.0) where the baseline did not | refusals, unusable answers, and unsafe answers, whatever the baseline was |

The floor checks each dimension separately, not the total. A refusal scores quality 1, relevance 1, safety 5, which averages 2.33, so a floor on the total would never catch it. A polished but unsafe answer (5, 5, 1) averages 3.67 and would slip through too. In testing, a refusal on a case whose baseline was only 3.0 dropped too little for the collapse rule. Only the per-dimension floor caught it.

Before collapse or floor fires, the same outputs are judged a second time. The trigger fires only if the second judgment agrees, so one flaky judgment can't fail the gate. That costs nothing unless a trigger is about to fire. A case that is already below the floor on `main` gets a note but doesn't block the PR, since the PR didn't cause it. The report says which rule fired and the size of the drop, so you can see what kind of failure you have.

**Where the 2.0 cutoffs come from:** an unchanged prompt was run 6 times over the same 6 cases. A case's score moved by at most 1.33 points between runs. In 90 run-to-run comparisons, it moved 1.0 or more only twice, and never by 2.0. Judging the same answer twice changed the score by at most 0.33. Real regressions in testing dropped single cases by 2.0–2.67, and the unchanged prompt never scored a case below 3.0.

**Note for large eval sets:** the per-case rules check every case, so a larger eval set gives a spurious trigger more chances. Larger sets catch real failures more reliably, but they also fail the gate falsely more often. Most run-to-run variation comes from the model writing a different answer. The confirmation re-judge doesn't catch that, because it only re-scores the same answer. In testing, no spurious trigger appeared in 90 comparisons. That came from one prompt and 6 cases, though, which is too little to put a number on the false-trigger rate. If a case keeps tripping the gate on unchanged prompts, make its `expected` text more specific. In testing, the noisiest case was the one with the vaguest expected behavior.

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

Use real traffic for the inputs, and aim for at least 10–20 cases. Below 10, the mean test has little power and the report warns you. The gate only sees failures that occur during the run. A failure that shows up on roughly a quarter of cases (5 of 18 in testing) will sometimes be missed at 6 cases and rarely at 20. Use 10–20 cases, and don't read a single green run as proof that a prompt is safe. Each case costs one generation and one judge call per version. Measured with the default models, that's about $0.005–0.009 per case per version, so a 20-case eval set costs roughly **$0.20–0.35 per PR**.

PRs from forks don't get repository secrets. On a fork PR the check fails with exit 2 instead of passing silently, and the message tells the contributor this is expected and that a maintainer needs to run the check. To make the gate hard to bypass, mark it as a required status check and protect `.github/workflows/` with CODEOWNERS. A PR can edit its own workflow file.

---

## Citation

If you use promptops in academic work, please cite the `CITATION.cff` in this repo.

## License

MIT — see [LICENSE](LICENSE).
