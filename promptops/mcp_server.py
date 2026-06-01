"""MCP server exposing the prompt registry.

Tools:
    get_active_version(prompt_name) -> {tag, text}
    list_prompts() -> {prompts: [{name, active, versions: [tag,...]}]}
    get_metrics(prompt_name) -> {versions: [{tag, alpha, beta, traffic, scores}]}

Transport: stdio (the MCP standard). Compatible with Claude Code and
any MCP client.

If the official `mcp` library isn't installed, this module exposes a
minimal stdio JSON-RPC fallback that supports `tools/list` and
`tools/call` per the MCP spec — enough for smoke tests and demos.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from . import store


def _get_active_version(prompt_name: str) -> dict[str, Any]:
    v = store.get_active(prompt_name)
    return {
        "name": prompt_name,
        "tag": v.tag,
        "text": store.read_text(prompt_name, v.tag),
        "alpha": v.alpha,
        "beta": v.beta,
        "traffic": v.traffic,
    }


def _list_prompts() -> dict[str, Any]:
    reg = store.load_registry()
    out = []
    for name, entry in reg.items():
        out.append({
            "name": name,
            "active": entry.get("active"),
            "versions": [v["tag"] for v in entry.get("versions", [])],
        })
    return {"prompts": out}


def _get_metrics(prompt_name: str) -> dict[str, Any]:
    versions = store.get_versions(prompt_name)
    total_traffic = sum(v.traffic for v in versions) or 1
    return {
        "name": prompt_name,
        "versions": [
            {
                "tag": v.tag,
                "alpha": v.alpha,
                "beta": v.beta,
                "traffic": v.traffic,
                "traffic_share": v.traffic / total_traffic,
                "scores": v.scores,
                "win_rate": v.alpha / (v.alpha + v.beta),
            }
            for v in versions
        ],
    }


TOOLS = [
    {
        "name": "get_active_version",
        "description": "Get the currently winning version of a prompt, including its text.",
        "inputSchema": {
            "type": "object",
            "properties": {"prompt_name": {"type": "string"}},
            "required": ["prompt_name"],
        },
        "_fn": _get_active_version,
    },
    {
        "name": "list_prompts",
        "description": "List all tracked prompts and their available versions.",
        "inputSchema": {"type": "object", "properties": {}},
        "_fn": lambda: _list_prompts(),
    },
    {
        "name": "get_metrics",
        "description": "Get A/B metrics for all versions of a prompt: alpha, beta, traffic, scores, win rate.",
        "inputSchema": {
            "type": "object",
            "properties": {"prompt_name": {"type": "string"}},
            "required": ["prompt_name"],
        },
        "_fn": _get_metrics,
    },
]


def _dispatch(name: str, args: dict[str, Any]) -> dict[str, Any]:
    for t in TOOLS:
        if t["name"] == name:
            fn = t["_fn"]
            if name == "list_prompts":
                return fn()
            return fn(**(args or {}))
    raise ValueError(f"unknown tool: {name}")


def _run_with_official_sdk() -> bool:
    """Try to run via the official `mcp` package. Returns True on success."""
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        import mcp.types as mtypes  # type: ignore
        import asyncio
    except Exception:
        return False

    server = Server("promptops")

    @server.list_tools()
    async def list_tools():  # type: ignore
        return [
            mtypes.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None):  # type: ignore
        result = _dispatch(name, arguments or {})
        return [mtypes.TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_main())
    return True


def _run_fallback_stdio() -> None:
    """Minimal JSON-RPC 2.0 stdio loop matching the MCP wire format."""
    def _send(obj: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                _send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "promptops", "version": "0.1.0"},
                }})
            elif method == "tools/list":
                _send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "tools": [
                        {k: v for k, v in t.items() if not k.startswith("_")}
                        for t in TOOLS
                    ]
                }})
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                result = _dispatch(name, args)
                _send({"jsonrpc": "2.0", "id": msg_id, "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                }})
            elif method in ("notifications/initialized", "notifications/cancelled"):
                continue
            else:
                _send({"jsonrpc": "2.0", "id": msg_id, "error": {
                    "code": -32601, "message": f"method not found: {method}",
                }})
        except Exception as e:
            _send({"jsonrpc": "2.0", "id": msg_id, "error": {
                "code": -32000, "message": str(e),
            }})


def run() -> None:
    if not _run_with_official_sdk():
        _run_fallback_stdio()


if __name__ == "__main__":
    run()
