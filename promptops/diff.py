"""Semantic diff between two prompt versions.

Uses sentence-transformers for embedding similarity, with a graceful
fallback to a hashing-based bag-of-words cosine if the model is not
available (e.g. CI without internet).
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


_MODEL = None


def _load_model():
    """Lazy-load sentence-transformers, fall back to None on failure."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _MODEL
    except Exception:
        _MODEL = False
        return None


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """Cheap hashed bag-of-words fallback embedding."""
    vec = [0.0] * dim
    for tok in re.findall(r"\w+", text.lower()):
        h = hash(tok) % dim
        vec[h] += 1.0
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _cosine(a, b) -> float:
    if hasattr(a, "tolist"):
        a = a.tolist()
    if hasattr(b, "tolist"):
        b = b.tolist()
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity in [-1, 1] (typically [0, 1] for prompts)."""
    model = _load_model()
    if model:
        emb = model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
        return float(_cosine(emb[0], emb[1]))
    return float(_cosine(_hash_embed(a), _hash_embed(b)))


def _split_sentences(text: str) -> list[str]:
    # crude sentence split on punctuation or newline
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


@dataclass
class DiffResult:
    similarity: float
    a_text: str
    b_text: str
    changes: list[tuple[str, str]] = field(default_factory=list)

    def render(self, console: Console | None = None, a_label: str = "a", b_label: str = "b") -> None:
        console = console or Console()
        pct = self.similarity * 100
        color = "green" if pct >= 90 else "yellow" if pct >= 70 else "red"
        header = Text()
        header.append("semantic similarity: ", style="bold")
        header.append(f"{pct:.2f}%", style=f"bold {color}")
        console.print(Panel(header, title=f"diff {a_label} → {b_label}", expand=False))

        if not self.changes:
            console.print("[dim]no sentence-level differences[/dim]")
            return

        for kind, line in self.changes:
            if kind == "-":
                console.print(Text(f"- {line}", style="red"))
            elif kind == "+":
                console.print(Text(f"+ {line}", style="green"))
            elif kind == "?":
                console.print(Text(f"  {line}", style="dim"))


def semantic_diff(a: str, b: str) -> DiffResult:
    sim = semantic_similarity(a, b)
    sa = _split_sentences(a)
    sb = _split_sentences(b)
    sm = difflib.SequenceMatcher(a=sa, b=sb)
    changes: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            for s in sa[i1:i2]:
                changes.append(("-", s))
        if tag in ("replace", "insert"):
            for s in sb[j1:j2]:
                changes.append(("+", s))
    return DiffResult(similarity=sim, a_text=a, b_text=b, changes=changes)
