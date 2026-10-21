from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from test_contracts import PROJECT_ROOT, VENV_PYTHON, result_data


@pytest.mark.asyncio
async def test_metadata_tool_reads_file_and_run_without_token_or_generation(tmp_path: Path):
    image_path = tmp_path / "original.png"
    info = PngInfo()
    info.add_text("Software", "NovelAI")
    info.add_text("Source", "NovelAI Diffusion V5 fixture")
    info.add_itxt(
        "Comment",
        json.dumps(
            {
                "model_name": "nai-diffusion-5-full",
                "prompt": "metadata only",
                "uc": "blur",
                "seed": 42,
                "steps": 28,
                "scale": 6,
                "future_parameter": {"preserve": True},
            }
        ),
    )
    Image.new("RGBA", (64, 64), (1, 2, 3, 255)).save(image_path, pnginfo=info)
    original = image_path.read_bytes()
    outputs = tmp_path / "outputs"
    run_id = "20261021T000000-112233aabbcc"
    run = outputs / run_id
    run.mkdir(parents=True)
    (run / "record.json").write_text(
        json.dumps(
            {
                "id": run_id,
                "operation": "generate",
                "artifacts": [
                    {"kind": "image", "path": str(image_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "request.json").write_text(json.dumps({"parameters": {"seed": 777}}), encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'token_file = "absent-token.txt"\n'
        'output_dir = "outputs"\n'
        'base_url = "https://image.novelai.net"\n'
        "allowed_image_roots = [" + json.dumps(tmp_path.as_posix()) + "]\n",
        encoding="utf-8",
    )
    env = {
        k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if k in os.environ
    }
    env.update(PYTHONPATH=str(PROJECT_ROOT / "src"), PYTHONIOENCODING="utf-8")
    server = StdioServerParameters(
        command=str(VENV_PYTHON),
        args=["-m", "novelai_private.server", "--config", str(config_path)],
        env=env,
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            catalog = {t.name: t for t in (await session.list_tools()).tools}
            tool = catalog["novelai_metadata"]
            assert tool.annotations.readOnlyHint is True
            assert tool.annotations.openWorldHint is False
            assert "cost_approved" not in tool.inputSchema["properties"]
            by_path = result_data(
                await session.call_tool("novelai_metadata", {"path": str(image_path)})
            )
            assert by_path["found"] is True
            assert by_path["generation"]["seed"] == 42
            assert by_path["generation"]["model"] == "nai-diffusion-5-full"
            assert by_path["parameters"]["future_parameter"] == {"preserve": True}
            by_run = result_data(
                await session.call_tool(
                    "novelai_metadata",
                    {
                        "run_id": run_id,
                        "include_raw": False,
                    },
                )
            )
            assert by_run["generation"]["seed"] == 42
            assert by_run["parameters"] == by_path["parameters"]
            assert "raw" not in by_run
            invalid = await session.call_tool(
                "novelai_metadata", {"run_id": run_id, "image_index": 1}
            )
            assert invalid.isError is True
            missing = await session.call_tool("novelai_metadata", {})
            assert missing.isError is True
    assert image_path.read_bytes() == original
    assert {p.name for p in outputs.iterdir()} == {run_id}
    assert not (tmp_path / "absent-token.txt").exists()
