"""Instant rollback to a previous prompt version."""

from __future__ import annotations

from pathlib import Path

from . import store


def rollback(name: str, to_tag: str, start: Path | None = None) -> store.Version:
    """Mark `to_tag` as the active version for the given prompt."""
    store.set_active(name, to_tag, start)
    return store.get_version(name, to_tag, start)
