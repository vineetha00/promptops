"""Git-native storage for prompt versions.

Layout on disk:
    .promptops/
        registry.json
        prompts/
            {name}/
                {timestamp}_{tag}.txt
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


PROMPTOPS_DIR = ".promptops"
PROMPTS_DIR = "prompts"
REGISTRY_FILE = "registry.json"


def _root(start: Path | None = None) -> Path:
    """Walk up from cwd looking for a .promptops dir. Falls back to cwd."""
    cur = (start or Path.cwd()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / PROMPTOPS_DIR).is_dir():
            return parent
    return cur


def promptops_path(start: Path | None = None) -> Path:
    return _root(start) / PROMPTOPS_DIR


def registry_path(start: Path | None = None) -> Path:
    return promptops_path(start) / REGISTRY_FILE


def prompts_path(start: Path | None = None) -> Path:
    return promptops_path(start) / PROMPTS_DIR


@dataclass
class Version:
    tag: str
    path: str
    timestamp: float
    scores: dict[str, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 1.0
    traffic: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Version":
        return cls(
            tag=d["tag"],
            path=d["path"],
            timestamp=d["timestamp"],
            scores=d.get("scores", {}),
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 1.0),
            traffic=d.get("traffic", 0),
        )


def is_initialized(start: Path | None = None) -> bool:
    return promptops_path(start).is_dir() and registry_path(start).is_file()


def init(start: Path | None = None) -> Path:
    """Create .promptops/ in the given directory."""
    base = (start or Path.cwd()).resolve()
    po = base / PROMPTOPS_DIR
    po.mkdir(exist_ok=True)
    (po / PROMPTS_DIR).mkdir(exist_ok=True)
    reg = po / REGISTRY_FILE
    if not reg.exists():
        reg.write_text(json.dumps({}, indent=2))
    return po


def load_registry(start: Path | None = None) -> dict[str, dict]:
    if not registry_path(start).is_file():
        return {}
    with open(registry_path(start)) as f:
        return json.load(f)


def save_registry(reg: dict[str, dict], start: Path | None = None) -> None:
    with open(registry_path(start), "w") as f:
        json.dump(reg, f, indent=2)


def _ensure_init(start: Path | None = None) -> None:
    if not is_initialized(start):
        raise RuntimeError(
            "Not a promptops repo. Run `promptops init` first."
        )


def commit(name: str, text: str, tag: str | None = None, start: Path | None = None) -> Version:
    """Add a new version of a prompt."""
    _ensure_init(start)
    reg = load_registry(start)
    entry = reg.setdefault(name, {"versions": [], "active": None})
    ts = time.time()
    if tag is None:
        tag = f"v{len(entry['versions']) + 1}"
    # Tag uniqueness — bump if necessary
    existing_tags = {v["tag"] for v in entry["versions"]}
    if tag in existing_tags:
        raise ValueError(f"Tag {tag!r} already exists for prompt {name!r}")

    safe_name = name.replace("/", "_").replace(" ", "_")
    pdir = prompts_path(start) / safe_name
    pdir.mkdir(parents=True, exist_ok=True)
    fname = f"{int(ts)}_{tag}.txt"
    fpath = pdir / fname
    fpath.write_text(text)

    rel_path = str(fpath.relative_to(_root(start)))
    v = Version(tag=tag, path=rel_path, timestamp=ts)
    entry["versions"].append(v.to_dict())
    if entry.get("active") is None:
        entry["active"] = tag
    reg[name] = entry
    save_registry(reg, start)
    return v


def list_prompts(start: Path | None = None) -> dict[str, dict]:
    return load_registry(start)


def get_versions(name: str, start: Path | None = None) -> list[Version]:
    reg = load_registry(start)
    if name not in reg:
        raise KeyError(f"Unknown prompt: {name!r}")
    return [Version.from_dict(v) for v in reg[name]["versions"]]


def get_version(name: str, tag: str, start: Path | None = None) -> Version:
    for v in get_versions(name, start):
        if v.tag == tag:
            return v
    raise KeyError(f"Tag {tag!r} not found for prompt {name!r}")


def read_text(name: str, tag: str, start: Path | None = None) -> str:
    v = get_version(name, tag, start)
    return (_root(start) / v.path).read_text()


def update_version(name: str, version: Version, start: Path | None = None) -> None:
    reg = load_registry(start)
    if name not in reg:
        raise KeyError(name)
    for i, v in enumerate(reg[name]["versions"]):
        if v["tag"] == version.tag:
            reg[name]["versions"][i] = version.to_dict()
            save_registry(reg, start)
            return
    raise KeyError(f"Tag {version.tag!r} not found for prompt {name!r}")


def set_active(name: str, tag: str, start: Path | None = None) -> None:
    reg = load_registry(start)
    if name not in reg:
        raise KeyError(name)
    if tag not in {v["tag"] for v in reg[name]["versions"]}:
        raise KeyError(f"Tag {tag!r} not found for prompt {name!r}")
    reg[name]["active"] = tag
    save_registry(reg, start)


def get_active(name: str, start: Path | None = None) -> Version:
    reg = load_registry(start)
    if name not in reg:
        raise KeyError(name)
    tag = reg[name].get("active")
    if tag is None:
        raise KeyError(f"No active version for prompt {name!r}")
    return get_version(name, tag, start)
