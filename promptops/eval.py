"""LLM-as-judge scoring of prompt versions.

Uses the Anthropic SDK to score prompt OUTPUTS on quality/relevance/safety.
When the SDK or API key is unavailable, falls back to a deterministic
heuristic so demos and CI smoke runs still work.

The eval also wires scores back into the A/B engine: head-to-head
comparisons treat the higher-scoring version as a "win" and update
the Beta posteriors via `ab.record`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import ab, store


JUDGE_MODEL = "claude-sonnet-4-20250514"

JUDGE_SYSTEM = """You are an impartial evaluator of LLM system prompts.
Score the prompt 1-5 on each of: quality, relevance, safety.
Respond ONLY with a compact JSON object:
{"quality": <1-5>, "relevance": <1-5>, "safety": <1-5>, "rationale": "<one sentence>"}"""


@dataclass
class Score:
    quality: float
    relevance: float
    safety: float
    rationale: str = ""

    @property
    def total(self) -> float:
        return (self.quality + self.relevance + self.safety) / 3.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "quality": self.quality,
            "relevance": self.relevance,
            "safety": self.safety,
            "total": self.total,
            "rationale": self.rationale,
        }


@dataclass
class EvalReport:
    prompt: str
    versions: dict[str, Score] = field(default_factory=dict)
    regressions: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "versions": {t: s.to_dict() for t, s in self.versions.items()},
            "regressions": [
                {"from": a, "to": b, "drop_pct": d} for a, b, d in self.regressions
            ],
        }


def _anthropic_client():
    try:
        import anthropic  # type: ignore
    except Exception:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return anthropic.Anthropic()
    except Exception:
        return None


def _heuristic_score(prompt_text: str) -> Score:
    """Deterministic fallback when no LLM judge is available."""
    words = re.findall(r"\w+", prompt_text)
    length = len(words)
    # Reward clear, well-sized prompts; penalize very short or huge ones.
    if length < 5:
        quality = 1.0
    elif length < 20:
        quality = 3.0
    elif length < 200:
        quality = 4.0 + min(1.0, length / 400.0)
    else:
        quality = 3.5
    relevance = 3.5 + (1.0 if any(w.lower() in {"you", "assistant", "agent"} for w in words) else 0.0)
    # Penalize obvious unsafe markers
    unsafe = any(w.lower() in {"ignore", "bypass", "jailbreak"} for w in words)
    safety = 2.0 if unsafe else 5.0
    return Score(
        quality=min(5.0, quality),
        relevance=min(5.0, relevance),
        safety=safety,
        rationale="heuristic fallback (no ANTHROPIC_API_KEY)",
    )


def judge(prompt_text: str) -> Score:
    """Score a single prompt via LLM-as-judge."""
    client = _anthropic_client()
    if client is None:
        return _heuristic_score(prompt_text)
    try:
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            system=JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Evaluate this system prompt:\n\n{prompt_text}"
            }],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        # Strip code fences if present
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        data = json.loads(text)
        return Score(
            quality=float(data.get("quality", 3.0)),
            relevance=float(data.get("relevance", 3.0)),
            safety=float(data.get("safety", 3.0)),
            rationale=str(data.get("rationale", "")),
        )
    except Exception as e:
        return _heuristic_score(prompt_text)


def evaluate_prompt(name: str, tags: list[str] | None = None,
                    start: Path | None = None) -> EvalReport:
    """Score every (or specified) version of a prompt and persist results."""
    versions = store.get_versions(name, start)
    if tags:
        versions = [v for v in versions if v.tag in tags]
    report = EvalReport(prompt=name)
    for v in versions:
        text = store.read_text(name, v.tag, start)
        s = judge(text)
        v.scores = s.to_dict()  # type: ignore[assignment]
        store.update_version(name, v, start)
        report.versions[v.tag] = s
    return report


def head_to_head(name: str, tag_a: str, tag_b: str,
                 start: Path | None = None) -> tuple[str, Score, Score]:
    """Score both versions, feed the result into the A/B Beta posteriors.

    Returns (winner_tag, score_a, score_b).
    """
    sa = judge(store.read_text(name, tag_a, start))
    sb = judge(store.read_text(name, tag_b, start))
    if sa.total >= sb.total:
        winner_tag, loser_tag = tag_a, tag_b
    else:
        winner_tag, loser_tag = tag_b, tag_a
    ab.record(name, winner_tag, won=True, start=start)
    ab.record(name, loser_tag, won=False, start=start)
    return winner_tag, sa, sb


def compare_branches(base_ref: str = "main", head_ref: str = "HEAD",
                     start: Path | None = None,
                     regression_threshold: float = 0.10) -> EvalReport:
    """Compare prompts on two git refs and flag regressions.

    For each prompt that has at least 2 versions, compare the score of
    the newest version against the previous one. If the score drops by
    more than `regression_threshold` (fraction), flag it.

    Note: a true cross-ref diff would `git show` files at each ref.
    Here we evaluate the current registry's two most recent versions per
    prompt, which is the simple, useful default for CI gates on PRs that
    have already been checked out.
    """
    reg = store.load_registry(start)
    report = EvalReport(prompt="<repo>")
    for name, entry in reg.items():
        versions = entry.get("versions", [])
        if len(versions) < 2:
            continue
        prev, curr = versions[-2], versions[-1]
        prev_score = judge((store._root(start) / prev["path"]).read_text())
        curr_score = judge((store._root(start) / curr["path"]).read_text())
        report.versions[f"{name}@{prev['tag']}"] = prev_score
        report.versions[f"{name}@{curr['tag']}"] = curr_score
        if prev_score.total > 0:
            drop = (prev_score.total - curr_score.total) / prev_score.total
            if drop > regression_threshold:
                report.regressions.append(
                    (f"{name}@{prev['tag']}", f"{name}@{curr['tag']}", drop)
                )
    return report
