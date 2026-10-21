from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgpack
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image
from test_contracts import PROJECT_ROOT, VENV_PYTHON, image_bytes, result_data

from novelai_private.client import ApiReply, msgpack_frames, stream_frames
from novelai_private.config import NovelAIError, Settings
from novelai_private.service import ImageService


def start_gateway_mcp_server(tmp_path: Path, quote: dict, image: bytes | None = None):
    received = []
    encoded = base64.b64encode(image or image_bytes((36, 102, 189), (64, 64))).decode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            assert self.headers["Authorization"] == "Bearer synthetic-gateway-token"
            assert b"synthetic-gateway-token" not in raw
            received.append((self.command, self.path, body))
            if self.path == "/api/quote":
                response = json.dumps({"quote": quote}).encode()
            else:
                response = json.dumps({"images": [{"image": encoded, "seed": 7}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "config.toml"
    config.write_text(
        f'base_url = "http://127.0.0.1:{http.server_port}/origin"\n'
        'output_dir = "outputs"\ntoken_file = "unused-token.txt"\n',
        encoding="utf-8",
    )
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if key in os.environ
    }
    env.update(
        NOVELAI_PRIVATE_CONFIG=str(config),
        NOVELAI_API_TOKEN="synthetic-gateway-token",
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        NO_PROXY="127.0.0.1",
    )
    params = StdioServerParameters(
        command=str(VENV_PYTHON), args=["-m", "novelai_private.server"], env=env
    )
    return http, thread, received, params


def test_webp_output_keeps_format_and_original_bytes(tmp_path: Path):
    buffer = io.BytesIO()
    Image.new("RGBA", (64, 64), (18, 60, 90, 125)).save(buffer, "WEBP", lossless=True)
    original = buffer.getvalue()
    service = ImageService(Settings(tmp_path / "config.toml", tmp_path, tmp_path / "no-token"))
    data = json.dumps({"images": [{"image": base64.b64encode(original).decode()}]}).encode()
    artifacts = service.save_reply(
        tmp_path, "generate", ApiReply(data, "application/json", "test01")
    )
    output = Path(artifacts[0]["path"])
    assert output.suffix == ".webp"
    assert output.read_bytes() == original
    with Image.open(output) as image:
        assert image.getpixel((1, 1)) == (18, 60, 90, 125)


def test_intermediate_only_stream_is_not_success():
    with pytest.raises(NovelAIError, match="没有最终图片"):
        stream_frames(b'event: intermediate\ndata: {"image":"preview-only"}\n\n')


def test_official_big_endian_msgpack_binary_final_frames():
    first = msgpack.packb({"event_type": "intermediate", "image": b"not-final", "samp_ix": 0})
    last = msgpack.packb({"event_type": "final", "image": b"binary-image", "samp_ix": 0})
    raw = len(first).to_bytes(4, "big") + first + len(last).to_bytes(4, "big") + last
    frames = msgpack_frames(raw)
    assert len(frames) == 1
    assert base64.b64decode(frames[0]["image"]) == b"binary-image"
    with pytest.raises(NovelAIError, match="截断"):
        msgpack_frames(raw[:-1])


@pytest.mark.asyncio
async def test_mcp_through_http_generate_view_revise_and_all_image_endpoints(tmp_path: Path):
    png = image_bytes((36, 102, 189), (64, 64))
    encoded = base64.b64encode(png).decode()
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.headers["Authorization"] == "Bearer synthetic-pipeline-token"
            received.append((self.path, body))
            if self.path.endswith("encode-vibe"):
                raw, mime = b"opaque-vibe-test", "application/binary"
            elif self.path.endswith("generate-image-stream"):
                if body["parameters"].get("stream") == "msgpack":
                    packed = msgpack.packb({"event_type": "final", "image": png, "samp_ix": 0})
                    raw, mime = len(packed).to_bytes(4, "big") + packed, "application/msgpack"
                else:
                    raw = (
                        'event: final\ndata: {"event_type":"final","image":"'
                        + encoded
                        + '","samp_ix":0}\n\n'
                    ).encode()
                    mime = "text/event-stream"
            else:
                raw = json.dumps({"images": [{"image": encoded, "seed": 77}]}).encode()
                mime = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            received.append((self.path, None))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"tags":[{"tag":"rain","confidence":0.9}]}')

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "source.png"
    source.write_bytes(png)
    config = tmp_path / "config.toml"
    config.write_text(
        f'base_url = "http://127.0.0.1:{http.server_port}"\n'
        'output_dir = "outputs"\npreview_max_side = 64\n',
        encoding="utf-8",
    )
    env = {
        k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP") if k in os.environ
    }
    env.update(
        NOVELAI_PRIVATE_CONFIG=str(config),
        NOVELAI_API_TOKEN="synthetic-pipeline-token",
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        NO_PROXY="127.0.0.1",
    )
    params = StdioServerParameters(
        command=str(VENV_PYTHON), args=["-m", "novelai_private.server"], env=env
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                async def call(name, args):
                    # These approvals exercise paid paths against this test-only local handler.
                    if name in {
                        "novelai_generate",
                        "novelai_revise",
                        "novelai_encode_vibe",
                        "novelai_augment",
                        "novelai_upscale",
                    }:
                        args = {
                            **args,
                            "cost_approved": True,
                            "high_resolution_approved": True,
                        }
                    result = await session.call_tool(name, args)
                    assert not result.isError, result
                    return result, result_data(result)

                request = {
                    "input": "rain",
                    "model": "nai-diffusion-4-5-full",
                    "action": "generate",
                    "parameters": {
                        "width": 64,
                        "height": 64,
                        "seed": 77,
                        "future_value": {"preserve": True},
                    },
                }
                generated, run = await call("novelai_generate", {"request": request})
                assert any(c.type == "image" for c in generated.content)
                expected_request = {
                    **request,
                    "parameters": {
                        **request["parameters"],
                        "steps": 28,
                        "scale": 6,
                        "n_samples": 1,
                    },
                    "use_new_shared_trial": True,
                }
                assert received[-1][1] == expected_request
                original = Path(run["artifacts"][0]["path"])
                assert await asyncio.to_thread(original.read_bytes) == png
                viewed, _ = await call("novelai_view_image", {"run_id": run["id"]})
                assert any(c.type == "image" for c in viewed.content)
                _, revised = await call(
                    "novelai_revise",
                    {
                        "run_id": run["id"],
                        "prompt": "soft rain",
                        "use_output_as_input": True,
                        "patch": {"parameters": {"strength": 0.35}},
                    },
                )
                assert revised["parent_id"] == run["id"]
                assert received[-1][1]["action"] == "img2img"
                assert received[-1][1]["parameters"]["future_value"] == {"preserve": True}
                assert await asyncio.to_thread(original.read_bytes) == png
                _, mask = await call(
                    "novelai_create_mask",
                    {
                        "image_path": str(source),
                        "regions": [{"shape": "rectangle", "box": [10, 10, 30, 30]}],
                    },
                )
                infill = {
                    **request,
                    "model": "nai-diffusion-4-5-full-inpainting",
                    "action": "infill",
                }
                await call(
                    "novelai_generate",
                    {
                        "request": infill,
                        "files": [
                            {"pointer": "/parameters/image", "path": str(source)},
                            {"pointer": "/parameters/mask", "path": mask["path"]},
                        ],
                    },
                )
                assert received[-1][1]["parameters"]["mask"]
                await call("novelai_generate", {"request": request, "stream": True})
                assert received[-1][0] == "/ai/generate-image-stream"
                assert received[-1][1]["parameters"]["stream"] == "sse"
                await call(
                    "novelai_generate",
                    {
                        "request": {
                            **request,
                            "parameters": {**request["parameters"], "stream": "msgpack"},
                        },
                        "stream": True,
                    },
                )
                assert received[-1][1]["parameters"]["stream"] == "msgpack"
                _, vibe = await call(
                    "novelai_encode_vibe",
                    {
                        "request": {
                            "model": "nai-diffusion-4-5-full",
                            "information_extracted": 0.8,
                        },
                        "files": [{"pointer": "/image", "path": str(source)}],
                    },
                )
                await call(
                    "novelai_generate",
                    {
                        "request": request,
                        "files": [
                            {
                                "pointer": "/parameters/reference_image_multiple/0",
                                "path": vibe["artifacts"][0]["path"],
                                "mode": "vibe",
                            }
                        ],
                    },
                )
                assert (
                    base64.b64decode(received[-1][1]["parameters"]["reference_image_multiple"][0])
                    == b"opaque-vibe-test"
                )
                await call(
                    "novelai_generate",
                    {
                        "request": request,
                        "files": [
                            {
                                "pointer": "/parameters/director_reference_images/0",
                                "path": str(source),
                                "prepare": "precise_reference",
                            }
                        ],
                    },
                )
                reference = received[-1][1]["parameters"]["director_reference_images"][0]
                with Image.open(io.BytesIO(base64.b64decode(reference))) as image:
                    assert image.size == (1472, 1472)
                await call(
                    "novelai_augment",
                    {
                        "request": {"req_type": "lineart", "width": 64, "height": 64},
                        "files": [{"pointer": "/image", "path": str(source)}],
                    },
                )
                await call(
                    "novelai_upscale",
                    {
                        "request": {"model": "nai-diffusion-4-5-full"},
                        "files": [{"pointer": "/image", "path": str(source)}],
                    },
                )
                _, tags = await call(
                    "novelai_suggest_tags",
                    {
                        "request": {
                            "model": "nai-diffusion-4-5-full",
                            "prompt": "rain",
                            "lang": "en",
                        }
                    },
                )
                assert tags["artifacts"][0]["data"]["tags"][0]["tag"] == "rain"
                assert {p.split("?")[0] for p, _ in received} == {
                    "/ai/generate-image",
                    "/ai/generate-image-stream",
                    "/ai/encode-vibe",
                    "/ai/augment-image",
                    "/ai/upscale",
                    "/ai/generate-image/suggest-tags",
                }
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_mcp_quote_tool_only_calls_quote_endpoint_and_hides_token(tmp_path: Path):
    http, thread, received, params = start_gateway_mcp_server(
        tmp_path,
        {"source": "platform", "currency": "credits", "credits": 60, "providerAnlas": 0},
    )
    request = {
        "input": "synthetic quote scene",
        "model": "nai-diffusion-4-5-full",
        "action": "generate",
        "parameters": {"width": 1024, "height": 1024},
    }
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                quote_schema = next(
                    tool.inputSchema for tool in listed.tools if tool.name == "novelai_quote"
                )
                assert quote_schema["properties"]["operation"]["enum"] == [
                    "generate",
                    "generate_stream",
                    "encode_vibe",
                    "augment",
                    "upscale",
                ]
                assert quote_schema["properties"]["request"]["type"] == "object"
                result = await session.call_tool(
                    "novelai_quote", {"operation": "generate", "request": request}
                )

        assert not result.isError
        data = result_data(result)
        assert data["status"] == "quote"
        assert data["cost_status"] == "paid"
        assert data["estimated_credits"] == 60
        assert "synthetic-gateway-token" not in json.dumps(data)
        assert [item[1] for item in received] == ["/api/quote"]
        assert received[0][2]["width"] == 1024
        assert received[0][2]["model"] == "nai-diffusion-4-5-full"
        assert not any(item.type == "image" for item in result.content)
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_unapproved_mcp_paid_generation_only_makes_read_only_quote(tmp_path: Path):
    http, thread, received, params = start_gateway_mcp_server(
        tmp_path,
        {"source": "platform", "currency": "credits", "credits": 60, "providerAnlas": 0},
    )
    request = {
        "input": "quiet local test",
        "model": "nai-diffusion-4-5-full",
        "action": "generate",
        "parameters": {"width": 64, "height": 64},
    }
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("novelai_generate", {"request": request})

        assert not result.isError
        data = result_data(result)
        assert data["status"] == "approval_required"
        assert data["cost_status"] == "paid"
        assert data["estimated_credits"] == 60
        assert [item[1] for item in received] == ["/api/quote"]
        assert not (tmp_path / "outputs").exists()
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_mcp_free_generation_passes_quote_gate_with_mocked_handler(tmp_path: Path):
    image = image_bytes((63, 126, 189), (64, 64))
    http, thread, received, params = start_gateway_mcp_server(
        tmp_path,
        {"source": "platform", "currency": "credits", "credits": 0, "providerAnlas": 0},
        image=image,
    )
    request = {
        "input": "free local test",
        "model": "nai-diffusion-4-5-full",
        "action": "generate",
        "parameters": {"width": 64, "height": 64},
    }
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("novelai_generate", {"request": request})

        assert not result.isError
        data = result_data(result)
        assert data["status"] == "complete"
        assert any(item.type == "image" for item in result.content)
        assert [item[1] for item in received] == ["/api/quote", "/origin/ai/generate-image"]
        record = (tmp_path / "outputs" / data["id"] / "record.json").read_text("utf-8")
        assert "synthetic-gateway-token" not in record
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=3)
