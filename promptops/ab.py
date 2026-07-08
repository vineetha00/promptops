"""Thompson Sampling A/B router for prompt versions.

Each version maintains a Beta(alpha, beta) posterior on its success
rate. To route a request, we sample once from each posterior and send
traffic to the highest sample. Wins → alpha+1, losses → beta+1.

A winner is declared once Pr(version X is best) >= threshold via a
Monte-Carlo estimate over the posteriors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as _np  # type: ignore
    from scipy.stats import beta as _beta_dist  # type: ignore
except Exception:  # pragma: no cover
    _np = None
    _beta_dist = None

from . import store


@dataclass
class ABResult:
    winner: str | None
    probability: float
    samples: int
    versions: dict[str, dict]


def _sample_beta(a: float, b: float) -> float:
    if _beta_dist is not None:
        return float(_beta_dist.rvs(a, b))
    # Python's random.betavariate is a fine fallback
    return random.betavariate(a, b)


def route(name: str, candidate_tags: list[str], start: Path | None = None) -> str:
    """Pick a version for the next request via Thompson Sampling."""
    versions = {t: store.get_version(name, t, start) for t in candidate_tags}
    best_tag, best_sample = candidate_tags[0], -1.0
    for tag, v in versions.items():
        s = _sample_beta(v.alpha, v.beta)
        if s > best_sample:
            best_sample = s
            best_tag = tag
    return best_tag


def record(name: str, tag: str, won: bool, start: Path | None = None) -> None:
    """Update posterior + traffic count for a version."""
    v = store.get_version(name, tag, start)
    if won:
        v.alpha += 1
    else:
        v.beta += 1
    v.traffic += 1
    store.update_version(name, v, start)


def _prob_best(versions: dict[str, dict], n_samples: int = 5000) -> tuple[str, float]:
    """Monte-Carlo estimate of Pr(each version is best)."""
    counts = {t: 0 for t in versions}
    tags = list(versions.keys())
    for _ in range(n_samples):
        best_t, best_s = tags[0], -1.0
        for t in tags:
            s = _sample_beta(versions[t]["alpha"], versions[t]["beta"])
            if s > best_s:
                best_s, best_t = s, t
        counts[best_t] += 1
    winner = max(counts, key=counts.get)
    return winner, counts[winner] / n_samples


def winner(name: str, candidate_tags: list[str], threshold: float = 0.95,
           start: Path | None = None) -> ABResult:
    """Declare a winner if posterior probability exceeds threshold."""
    versions = {}
    for t in candidate_tags:
        v = store.get_version(name, t, start)
        versions[t] = {"alpha": v.alpha, "beta": v.beta, "traffic": v.traffic}
    total = sum(v["traffic"] for v in versions.values())
    if total < max(10, 2 * len(candidate_tags)):
        return ABResult(winner=None, probability=0.0, samples=total, versions=versions)
    w, p = _prob_best(versions)
    return ABResult(
        winner=w if p >= threshold else None,
        probability=p,
        samples=total,
        versions=versions,
    )


def simulate(name: str, candidate_tags: list[str], n: int, true_rates: dict[str, float],
             start: Path | None = None, seed: int | None = None) -> ABResult:
    """Drive `n` simulated trials using the given true success rates.

    Useful for demos and tests when no real LLM-judge is hooked up.
    """
    if seed is not None:
        random.seed(seed)
        if _np is not None:
            # scipy's beta.rvs draws from numpy's global RNG, not `random`
            _np.random.seed(seed)
    for _ in range(n):
        tag = route(name, candidate_tags, start)
        rate = true_rates.get(tag, 0.5)
        won = random.random() < rate
        record(name, tag, won, start)
    return winner(name, candidate_tags, start=start)
