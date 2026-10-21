"""Call this plugin over MCP when the host has not refreshed its native tool list."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def call(tool: str, arguments: dict) -> int:
    env = {
        k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if k in os.environ
    }
    env.update(PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    if os.environ.get("NOVELAI_PRIVATE_CONFIG"):
        env["NOVELAI_PRIVATE_CONFIG"] = os.environ["NOVELAI_PRIVATE_CONFIG"]
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "novelai_private.server"],
        env=env,
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            if result.isError:
                # Server errors are sanitized before becoming MCP TextContent.
                for item in result.content:
                    if item.type == "text":
                        print(item.text)
                return 1
            if result.structuredContent:
                print(json.dumps(result.structuredContent, ensure_ascii=False))
            else:
                print(json.dumps({"completed": True}, ensure_ascii=False))
            return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool")
    parser.add_argument("arguments_file", nargs="?", type=Path)
    args = parser.parse_args()
    arguments = (
        json.loads(args.arguments_file.read_text("utf-8-sig")) if args.arguments_file else {}
    )
    raise SystemExit(asyncio.run(call(args.tool, arguments)))


if __name__ == "__main__":
    main()
