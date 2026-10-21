from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image

from novelai_private.client import NovelAIClient
from novelai_private.config import NovelAIError, Settings
from novelai_private.media import bind_files
from novelai_private.schema import official_document, validate
from novelai_private.service import ImageService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = Path(sys.executable)
SYNTHETIC_TOKEN = "unit-test-token-not-a-real-credential"


def image_bytes(color: tuple[int, int, int], size: tuple[int, int] = (80, 48)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def png_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return output.getvalue()


def value_from_schema(schema: dict[str, Any], definitions: dict[str, Any]) -> Any:
    """Build representative legal values from the bundled official schema itself."""
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        return value_from_schema(definitions[name], definitions)
    if "allOf" in schema:
        merged: dict[str, Any] = {}
        for child in schema["allOf"]:
            value = value_from_schema(child, definitions)
            if isinstance(value, dict):
                merged.update(value)
        return merged
    if "enum" in schema:
        return schema["enum"][0]
    value_type = schema.get("type")
    if value_type == "object":
        return {
            name: value_from_schema(child, definitions)
            for name, child in schema.get("properties", {}).items()
        }
    if value_type == "array":
        return [value_from_schema(schema.get("items", {}), definitions)]
    if value_type == "string":
        return "schema-derived-test-value"
    if value_type == "integer":
        return 1
    if value_type == "number":
        return 0.5
    if value_type == "boolean":
        return True
    raise AssertionError(f"Unsupported schema type in official snapshot: {schema!r}")


def complete_official_generation_request() -> dict[str, Any]:
    document = official_document()
    definitions = document["definitions"]
    request = value_from_schema(definitions["image.ImageGenerationRequest"], definitions)
    request.update(
        input="quiet test scene",
        model="nai-diffusion-4-5-full",
        action="generate",
        future_root_field={"preserve": ["root", 7]},
    )
    parameters = request["parameters"]
    parameters.update(
        params_version=4,
        width=64,
        height=64,
        steps=28,
        scale=5,
        seed=123456,
        n_samples=1,
        sampler="k_euler_ancestral",
        noise_schedule="karras",
        future_parameter={"nested": {"unknown_field": "keep-me"}},
    )
    return request


def make_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        path=tmp_path / "config.toml",
        output_dir=tmp_path / "outputs",
        token_file=tmp_path / "absent-token.txt",
        base_url="https://image.novelai.net",
        timeout_seconds=2,
        preview_max_side=128,
        proxy=None,
        allowed_image_roots=(),
    )


def test_file_binding_reads_official_naiv4vibe_variant_hash_container(
    tmp_path: Path,
) -> None:
    encoded_vibe = base64.b64encode(b"synthetic opaque vibe embedding").decode("ascii")
    container = {
        "identifier": "novelai-vibe-transfer",
        "version": 1,
        "type": "image",
        "image": base64.b64encode(image_bytes((8, 12, 16))).decode("ascii"),
        "id": "synthetic-id",
        "encodings": {
            "v4-5full": {
                "b36a8472fe418d9f80d6bb1c54e3a6e62c62936aa7bf31dae2bcf7e929f6430f": {
                    "encoding": encoded_vibe,
                    "params": {"information_extracted": 0.7},
                }
            }
        },
        "name": "synthetic reference",
        "createdAt": 1,
        "importInfo": {
            "model": "nai-diffusion-4-5-full",
            "information_extracted": 0.7,
            "strength": 0.5,
        },
    }
    reference_path = tmp_path / "synthetic.naiv4vibe"
    reference_path.write_text(json.dumps(container), encoding="utf-8")
    payload = {
        "model": "nai-diffusion-4-5-full",
        "parameters": {"reference_image_multiple": []},
    }

    result = bind_files(
        payload,
        [
            {
                "pointer": "/parameters/reference_image_multiple/0",
                "path": str(reference_path),
                "mode": "vibe",
                "information_extracted": 0.7,
            }
        ],
        make_test_settings(tmp_path),
    )

    assert result["parameters"]["reference_image_multiple"] == [encoded_vibe]


def test_file_binding_selects_vibe_from_official_bundle_extension(
    tmp_path: Path,
) -> None:
    first_encoding = base64.b64encode(b"first synthetic embedding").decode("ascii")
    selected_encoding = base64.b64encode(b"second synthetic embedding").decode("ascii")

    def vibe_entry(encoded: str) -> dict[str, Any]:
        return {
            "identifier": "novelai-vibe-transfer",
            "version": 1,
            "type": "image",
            "image": base64.b64encode(image_bytes((8, 12, 16))).decode("ascii"),
            "id": "synthetic-id",
            "encodings": {
                "v4-5full": {
                    "b36a8472fe418d9f80d6bb1c54e3a6e62c62936aa7bf31dae2bcf7e929f6430f": {
                        "encoding": encoded,
                        "params": {"information_extracted": 0.7},
                    }
                }
            },
            "name": "synthetic reference",
            "createdAt": 1,
            "importInfo": {
                "model": "nai-diffusion-4-5-full",
                "information_extracted": 0.7,
                "strength": 0.5,
            },
        }

    bundle = {
        "identifier": "novelai-vibe-transfer-bundle",
        "version": 1,
        "vibes": [vibe_entry(first_encoding), vibe_entry(selected_encoding)],
    }
    reference_path = tmp_path / "synthetic.naiv4vibeBundle"
    reference_path.write_text(json.dumps(bundle), encoding="utf-8")
    payload = {
        "model": "nai-diffusion-4-5-full",
        "parameters": {"reference_image_multiple": []},
    }

    result = bind_files(
        payload,
        [
            {
                "pointer": "/parameters/reference_image_multiple/0",
                "path": str(reference_path),
                "mode": "vibe",
                "information_extracted": 0.7,
                "vibe_index": 1,
            }
        ],
        make_test_settings(tmp_path),
    )

    assert result["parameters"]["reference_image_multiple"] == [selected_encoding]


@pytest.fixture
def synthetic_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVELAI_API_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.delenv("NOVELAI_PRIVATE_CONFIG", raising=False)


@pytest.mark.asyncio
async def test_http_client_preserves_complete_official_payload_and_keeps_token_in_header(
    tmp_path: Path, synthetic_token: None
) -> None:
    payload = complete_official_generation_request()
    validate("generate", payload)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=png_zip({"image_0.png": image_bytes((30, 70, 110))}),
        )

    client = NovelAIClient(make_test_settings(tmp_path), transport=httpx.MockTransport(handler))
    await client.request("generate", payload)

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/ai/generate-image"
    assert request.headers["authorization"] == f"Bearer {SYNTHETIC_TOKEN}"
    assert all(
        SYNTHETIC_TOKEN not in value
        for name, value in request.headers.items()
        if name.lower() != "authorization"
    )
    assert SYNTHETIC_TOKEN.encode() not in request.content
    assert json.loads(request.content) == payload
    assert (
        len(payload["parameters"])
        == len(official_document()["definitions"]["image.RequestParameters"]["properties"]) + 1
    )
    assert payload["future_root_field"] == {"preserve": ["root", 7]}
    assert payload["parameters"]["future_parameter"] == {"nested": {"unknown_field": "keep-me"}}


@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.asyncio
async def test_http_status_errors_are_not_retried_or_disclose_synthetic_token(
    status: int, tmp_path: Path, synthetic_token: None
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"message": "synthetic rejection"})

    client = NovelAIClient(make_test_settings(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(NovelAIError) as caught:
        await client.request(
            "generate",
            {
                "input": "test",
                "model": "nai-diffusion-4-5-full",
                "action": "generate",
                "parameters": {},
            },
        )

    assert calls == 1
    assert "未自动重试" in str(caught.value)
    assert SYNTHETIC_TOKEN not in str(caught.value)


@pytest.mark.asyncio
async def test_http_timeout_is_not_retried(tmp_path: Path, synthetic_token: None) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    client = NovelAIClient(make_test_settings(tmp_path), transport=httpx.MockTransport(handler))
    with pytest.raises(NovelAIError, match="请求超时") as caught:
        await client.request(
            "generate",
            {
                "input": "test",
                "model": "nai-diffusion-4-5-full",
                "action": "generate",
                "parameters": {},
            },
        )

    assert calls == 1
    assert "未自动重试" in str(caught.value)
    assert SYNTHETIC_TOKEN not in str(caught.value)


@pytest.mark.parametrize("response_kind", ["zip", "json", "sse"])
@pytest.mark.asyncio
async def test_image_service_saves_only_final_images_from_each_response_format(
    response_kind: str, tmp_path: Path, synthetic_token: None
) -> None:
    final_png = image_bytes((24, 178, 94))
    preview_png = image_bytes((210, 20, 35))
    encoded_final = base64.b64encode(final_png).decode("ascii")
    encoded_preview = base64.b64encode(preview_png).decode("ascii")

    if response_kind == "zip":
        content = png_zip({"nested/final.png": final_png})
        headers = {"content-type": "application/zip"}
        operation = "generate"
    elif response_kind == "json":
        content = json.dumps(
            {"images": [{"image": encoded_final, "seed": 991, "index": 0}]}
        ).encode()
        headers = {"content-type": "application/json"}
        operation = "generate"
    else:
        content = (
            'event: intermediate\ndata: {"event_type":"intermediate","image":"'
            + encoded_preview
            + '"}\n\n'
            'event: final\ndata: {"event_type":"final","samp_ix":0,"seed":992,"image":"'
            + encoded_final
            + '"}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        headers = {"content-type": "text/event-stream; charset=utf-8"}
        operation = "generate_stream"
    headers["x-anlas-cost"] = "0"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, content=content)

    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings,
        NovelAIClient(settings, transport=httpx.MockTransport(handler)),
    )
    payload = {
        "input": "save final image",
        "model": "nai-diffusion-4-5-full",
        "action": "generate",
        "parameters": {"width": 64, "height": 64},
    }
    if operation == "generate_stream":
        payload["parameters"]["stream"] = "sse"

    result = await service.execute(
        operation, payload, cost_approved=True, high_resolution_approved=True
    )

    assert result["status"] == "complete"
    record_text = await asyncio.to_thread(Path(result["record_path"]).read_text, "utf-8")
    assert json.loads(record_text)["actual_anlas"] == 0
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    saved = Path(artifact["path"])
    assert file_exists(saved)
    assert file_bytes(saved) == final_png
    assert artifact["width"] == 80
    assert artifact["height"] == 48
    with Image.open(saved) as image:
        assert image.size == (80, 48)
        assert image.getpixel((40, 24)) == (24, 178, 94)
    if response_kind == "zip":
        assert (saved.parent / "response.zip").is_file()
    elif response_kind == "json":
        assert (saved.parent / "response.json").is_file()
        assert artifact["seed"] == 991
    else:
        assert file_bytes(saved.parent / "response.sse") == content
        assert artifact["seed"] == 992
        assert file_bytes(saved) != preview_png


@pytest.mark.asyncio
async def test_revise_keeps_parent_assets_syncs_prompts_and_binds_prior_output(
    tmp_path: Path, synthetic_token: None
) -> None:
    original_png = image_bytes((140, 66, 28))
    revised_png = image_bytes((18, 96, 176))
    response_bodies = [
        {"images": [{"image": base64.b64encode(data).decode(), "seed": index, "index": 0}]}
        for index, data in enumerate((original_png, revised_png))
    ]
    sent: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json=response_bodies[len(sent) - 1],
        )

    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings,
        NovelAIClient(settings, transport=httpx.MockTransport(handler)),
    )
    original_request = {
        "input": "original prompt",
        "model": "nai-diffusion-4-5-full",
        "action": "generate",
        "future_root": {"keep": "root-value"},
        "parameters": {
            "width": 64,
            "height": 64,
            "seed": 41,
            "negative_prompt": "old negative",
            "v4_prompt": {
                "caption": {
                    "base_caption": "original prompt",
                    "char_captions": [{"char_caption": "blue-haired character"}],
                },
                "use_coords": False,
                "use_order": True,
            },
            "v4_negative_prompt": {
                "caption": {"base_caption": "old negative", "char_captions": []},
                "legacy_uc": False,
            },
            "future_parameter": {"mode": "preserve", "nested": {"old": True}},
        },
    }
    parent = await service.execute(
        "generate", original_request, cost_approved=True, high_resolution_approved=True
    )
    parent_run_id = parent["id"]
    parent_record_path = Path(parent["record_path"])
    parent_request_path = Path(parent["request_path"])
    parent_record_before = file_bytes(parent_record_path)
    parent_request_before = file_bytes(parent_request_path)
    original_image_path = Path(parent["artifacts"][0]["path"])
    assert file_bytes(original_image_path) == original_png

    child = await service.revise(
        parent_run_id,
        patch={
            "future_root": {"added": "still-merged"},
            "parameters": {
                "seed": 42,
                "future_parameter": {"nested": {"new": 2}},
            },
        },
        prompt="revised prompt",
        negative_prompt="revised negative",
        use_output_as_input=True,
        cost_approved=True,
        high_resolution_approved=True,
    )

    revised_request = sent[1]
    assert revised_request["action"] == "img2img"
    assert revised_request["input"] == "revised prompt"
    assert revised_request["future_root"] == {
        "keep": "root-value",
        "added": "still-merged",
    }
    assert revised_request["parameters"]["v4_prompt"]["caption"]["base_caption"] == (
        "revised prompt"
    )
    assert revised_request["parameters"]["v4_prompt"]["caption"]["char_captions"] == [
        {"char_caption": "blue-haired character"}
    ]
    assert revised_request["parameters"]["negative_prompt"] == "revised negative"
    assert revised_request["parameters"]["v4_negative_prompt"]["caption"]["base_caption"] == (
        "revised negative"
    )
    assert revised_request["parameters"]["future_parameter"] == {
        "mode": "preserve",
        "nested": {"old": True, "new": 2},
    }
    assert revised_request["parameters"]["seed"] == 42
    assert revised_request["parameters"]["strength"] == 0.5
    assert revised_request["parameters"]["noise"] == 0
    assert revised_request["parameters"]["image"] == base64.b64encode(original_png).decode()
    assert "mask" not in revised_request["parameters"]
    assert "img2img" not in revised_request["parameters"]
    assert child["parent_id"] == parent_run_id
    assert file_bytes(parent_record_path) == parent_record_before
    assert file_bytes(parent_request_path) == parent_request_before
    assert file_bytes(original_image_path) == original_png
    assert file_bytes(Path(child["artifacts"][0]["path"])) == revised_png


PAID_OPERATIONS = ("generate", "generate_stream", "encode_vibe", "augment", "upscale")


def paid_operation_request(operation: str, width: int = 64, height: int = 64) -> dict[str, Any]:
    encoded_image = base64.b64encode(image_bytes((22, 44, 66))).decode("ascii")
    if operation in {"generate", "generate_stream"}:
        parameters: dict[str, Any] = {"width": width, "height": height}
        if operation == "generate_stream":
            parameters["stream"] = "sse"
        return {
            "input": "a small test image",
            "model": "nai-diffusion-4-5-full",
            "action": "generate",
            "parameters": parameters,
        }
    if operation == "encode_vibe":
        return {"model": "nai-diffusion-4-5-full", "image": encoded_image}
    if operation == "augment":
        return {
            "req_type": "lineart",
            "width": width,
            "height": height,
            "image": encoded_image,
        }
    if operation == "upscale":
        return {"model": "nai-diffusion-4-5-full", "image": encoded_image}
    raise AssertionError(f"Unsupported test operation: {operation}")


def json_image_response(image: bytes) -> httpx.Response:
    encoded = base64.b64encode(image).decode("ascii")
    body = json.dumps({"images": [{"image": encoded, "index": 0, "seed": 7}]}).encode()
    return httpx.Response(200, headers={"content-type": "application/json"}, content=body)


def subscription_state(
    *,
    tier: int = 3,
    active: bool = True,
    percent: int | None = 100,
    is_negative: bool = False,
    include_usage: bool = True,
) -> dict[str, Any]:
    state: dict[str, Any] = {"tier": tier, "active": active}
    if include_usage:
        state["usage"] = {
            "percent": percent,
            "isNegative": is_negative,
            "timeUntilNextPercent": 7888,
            "purchasedPercent": 0,
        }
    return state


def mocked_service_handler(
    requests: list[httpx.Request],
    *,
    subscription: dict[str, Any] | None = None,
    gateway_quote: dict[str, Any] | None = None,
    image: bytes | None = None,
):
    """Mock billing reads and image generation; never contacts a live service."""

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/user/subscription"):
            if subscription is None:
                return httpx.Response(503, json={"message": "synthetic billing failure"})
            return httpx.Response(200, json=subscription)
        if request.url.path.endswith("/api/quote"):
            if gateway_quote is None:
                return httpx.Response(503, json={"message": "synthetic quote failure"})
            return httpx.Response(200, json={"quote": gateway_quote})
        return json_image_response(image or image_bytes((1, 2, 3), (2, 2)))

    return handler


def assert_billing_only(requests: list[httpx.Request]) -> None:
    assert requests
    assert all(
        (request.method == "GET" and request.url.path.endswith("/user/subscription"))
        or (request.method == "POST" and request.url.path.endswith("/api/quote"))
        for request in requests
    )
    for request in requests:
        assert request.headers.get("authorization") == f"Bearer {SYNTHETIC_TOKEN}"
        assert SYNTHETIC_TOKEN.encode() not in request.content
        assert all(
            SYNTHETIC_TOKEN not in value
            for name, value in request.headers.items()
            if name.lower() != "authorization"
        )


@pytest.mark.parametrize("operation", PAID_OPERATIONS)
@pytest.mark.asyncio
async def test_unapproved_paid_or_unknown_operation_only_reads_billing_and_writes_nothing(
    operation: str, tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []
    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings,
        NovelAIClient(
            settings,
            transport=httpx.MockTransport(
                mocked_service_handler(requests, subscription=subscription_state(active=False))
            ),
        ),
    )

    result = await service.execute(operation, paid_operation_request(operation))

    assert result["status"] == "approval_required"
    assert result["operation"] == operation
    assert result["cost_status"] in {"paid", "unknown"}
    assert result["may_spend_credits"] is True
    assert all(not request.url.path.startswith("/ai/") for request in requests)
    if requests:
        assert_billing_only(requests)
    assert not settings.output_dir.exists()


@pytest.mark.parametrize("operation", ["generate", "generate_stream"])
@pytest.mark.parametrize(
    ("specified", "expected_steps", "expected_scale"),
    [
        ({}, 28, 6),
        ({"steps": 17}, 17, 6),
        ({"scale": 4.25}, 28, 4.25),
        ({"steps": 12, "scale": 3.5}, 12, 3.5),
    ],
)
@pytest.mark.asyncio
async def test_generation_defaults_only_fill_missing_steps_and_scale(
    operation: str,
    specified: dict[str, Any],
    expected_steps: int,
    expected_scale: float,
    tmp_path: Path,
    synthetic_token: None,
) -> None:
    requests: list[httpx.Request] = []
    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings,
        NovelAIClient(
            settings,
            transport=httpx.MockTransport(
                mocked_service_handler(requests, subscription=subscription_state())
            ),
        ),
    )
    request = paid_operation_request(operation)
    request["parameters"].update(specified)

    result = await service.execute(operation, request)

    assert result["status"] == "complete"
    assert len([item for item in requests if item.url.path == "/user/subscription"]) == 1
    generation = [item for item in requests if item.url.path.startswith("/ai/")]
    assert len(generation) == 1
    sent = json.loads(generation[0].content)
    assert (sent["parameters"]["steps"], sent["parameters"]["scale"]) == (
        expected_steps,
        expected_scale,
    )
    assert sent["parameters"]["n_samples"] == 1
    assert sent["use_new_shared_trial"] is True
    assert (sent["parameters"]["width"], sent["parameters"]["height"]) == (64, 64)


@pytest.mark.parametrize("operation", ["generate", "generate_stream"])
@pytest.mark.asyncio
async def test_generation_requires_explicit_dimensions_instead_of_assuming_square(
    operation: str, tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return json_image_response(image_bytes((1, 2, 3), (2, 2)))

    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    request = paid_operation_request(operation)
    request["parameters"].pop("width")
    request["parameters"].pop("height")

    preview = await service.execute(operation, request)
    assert preview["status"] == "approval_required"
    assert preview["missing_resolution"] is True
    assert preview["width"] is None and preview["height"] is None
    assert (preview["steps"], preview["scale"]) == (28, 6)

    with pytest.raises(NovelAIError, match="需指定明确的 width 和 height"):
        await service.execute(operation, request, cost_approved=True)

    assert requests == []
    assert not settings.output_dir.exists()


@pytest.mark.asyncio
async def test_unapproved_revision_reads_billing_but_does_not_send_or_create_child_record(
    tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []
    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings,
        NovelAIClient(
            settings,
            transport=httpx.MockTransport(
                mocked_service_handler(
                    requests,
                    subscription=subscription_state(active=False),
                    image=image_bytes((31, 63, 95), (2, 2)),
                )
            ),
        ),
    )
    parent = await service.execute(
        "generate",
        paid_operation_request("generate"),
        cost_approved=True,
        high_resolution_approved=True,
    )
    existing_runs = {path.name for path in settings.output_dir.iterdir()}

    result = await service.revise(parent["id"], patch={}, use_output_as_input=True)

    assert result["status"] == "approval_required"
    assert result["cost_status"] in {"paid", "unknown"}
    assert len([item for item in requests if item.url.path == "/ai/generate-image"]) == 1
    assert_billing_only([item for item in requests if item.url.path.endswith("/user/subscription")])
    assert {path.name for path in settings.output_dir.iterdir()} == existing_runs


@pytest.mark.parametrize("model", ["nai-diffusion-4-5-full", "nai-diffusion-5-full"])
@pytest.mark.parametrize("width,height", [(1024, 1024), (832, 1216), (1216, 832)])
@pytest.mark.asyncio
async def test_opus_standard_canvas_sizes_are_free_without_cost_approval(
    model: str, width: int, height: int, tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []
    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings,
        NovelAIClient(
            settings,
            transport=httpx.MockTransport(
                mocked_service_handler(requests, subscription=subscription_state())
            ),
        ),
    )
    request = paid_operation_request("generate", width, height)
    request["model"] = model

    result = await service.execute("generate", request)

    assert result["status"] == "complete"
    record_text = await asyncio.to_thread(Path(result["record_path"]).read_text, "utf-8")
    record = json.loads(record_text)
    assert record["billing"]["cost_status"] == "free"
    assert record["billing"]["estimated_anlas"] == 0
    assert len([item for item in requests if item.url.path == "/user/subscription"]) == 1
    generation = [item for item in requests if item.url.path == "/ai/generate-image"]
    assert len(generation) == 1
    sent = json.loads(generation[0].content)
    assert (sent["parameters"]["width"], sent["parameters"]["height"]) == (width, height)


@pytest.mark.parametrize("operation", ["generate", "generate_stream", "augment"])
@pytest.mark.asyncio
async def test_large_output_requires_both_approvals(
    operation: str, tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return json_image_response(image_bytes((15, 30, 45), (2, 2)))

    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    request = paid_operation_request(operation, 1025, 1024)

    without_cost = await service.execute(operation, request)
    assert without_cost["status"] == "approval_required"
    assert without_cost["needs_high_resolution_approval"] is True
    assert not settings.output_dir.exists()

    with_cost = await service.execute(operation, request, cost_approved=True)
    assert with_cost["status"] == "high_resolution_approval_required"
    assert all(
        (item.method == "GET" and item.url.path.endswith("/user/subscription"))
        or (item.method == "POST" and item.url.path.endswith("/api/quote"))
        for item in requests
    )
    assert not settings.output_dir.exists()

    approved = await service.execute(
        operation, request, cost_approved=True, high_resolution_approved=True
    )
    assert approved["status"] == "complete"
    assert len([item for item in requests if item.url.path.startswith("/ai/")]) == 1


@pytest.mark.asyncio
async def test_upscale_with_unknown_output_size_requires_both_approvals(
    tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return json_image_response(image_bytes((18, 36, 54), (2, 2)))

    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    request = paid_operation_request("upscale")

    without_cost = await service.execute("upscale", request)
    assert without_cost["status"] == "approval_required"
    assert without_cost["output_resolution_unknown"] is True
    assert not settings.output_dir.exists()

    with_cost = await service.execute("upscale", request, cost_approved=True)
    assert with_cost["status"] == "high_resolution_approval_required"
    assert requests == []
    assert not settings.output_dir.exists()

    approved = await service.execute(
        "upscale", request, cost_approved=True, high_resolution_approved=True
    )
    assert approved["status"] == "complete"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_large_revision_requires_both_approvals(
    tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return json_image_response(image_bytes((27, 54, 81), (2, 2)))

    settings = make_test_settings(tmp_path)
    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    parent = await service.execute(
        "generate",
        paid_operation_request("generate", 1025, 1024),
        cost_approved=True,
        high_resolution_approved=True,
    )
    existing_runs = {path.name for path in settings.output_dir.iterdir()}

    needs_resolution_approval = await service.revise(
        parent["id"], patch={}, use_output_as_input=True, cost_approved=True
    )
    assert needs_resolution_approval["status"] == "high_resolution_approval_required"
    assert len(requests) == 1
    assert {path.name for path in settings.output_dir.iterdir()} == existing_runs

    revised = await service.revise(
        parent["id"],
        patch={},
        use_output_as_input=True,
        cost_approved=True,
        high_resolution_approved=True,
    )
    assert revised["status"] == "complete"
    assert len(requests) == 2


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def file_exists(path: Path) -> bool:
    return path.is_file()


def result_data(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    for item in result.content:
        if getattr(item, "type", None) == "text":
            return json.loads(item.text)
    raise AssertionError("MCP tool result did not include structured data")


@pytest.mark.asyncio
async def test_real_stdio_mcp_initialize_list_and_local_image_tools(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    source_path = assets / "reference.png"
    source_path.write_bytes(image_bytes((199, 42, 18), (80, 48)))
    config_path = tmp_path / "mcp-config.toml"
    config_path.write_text(
        'output_dir = "outputs"\n'
        'token_file = "missing-token.txt"\n'
        'allowed_image_roots = ["assets"]\n'
        "preview_max_side = 64\n",
        encoding="utf-8",
    )

    # Construct a minimal child environment; in particular, never inherit or use a real API token.
    child_env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
        if key in os.environ
    }
    child_env.update(
        PYTHONPATH=str(PROJECT_ROOT / "src"),
        PYTHONIOENCODING="utf-8",
        PYTHONDONTWRITEBYTECODE="1",
        NOVELAI_PRIVATE_CONFIG=str(config_path),
    )
    assert "NOVELAI_API_TOKEN" not in child_env
    parameters = StdioServerParameters(
        command=str(VENV_PYTHON),
        args=["-m", "novelai_private.server"],
        cwd=PROJECT_ROOT,
        env=child_env,
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            assert initialized.serverInfo.name == "novelai-private"

            listed = await session.list_tools()
            tool_names = {entry.name for entry in listed.tools}
            tool_schemas = {entry.name: entry.inputSchema for entry in listed.tools}
            assert {
                "novelai_generate",
                "novelai_revise",
                "novelai_encode_vibe",
                "novelai_augment",
                "novelai_upscale",
                "novelai_suggest_tags",
                "novelai_status",
                "novelai_schema",
                "novelai_quote",
                "novelai_view_image",
                "novelai_create_mask",
            } <= tool_names
            paid_tools = {
                "novelai_generate",
                "novelai_revise",
                "novelai_encode_vibe",
                "novelai_augment",
                "novelai_upscale",
            }
            for name in paid_tools:
                properties = tool_schemas[name]["properties"]
                for approval in ("cost_approved", "high_resolution_approved"):
                    assert properties[approval]["type"] == "boolean"
                    assert properties[approval]["default"] is False
            assert "cost_approved" not in tool_schemas["novelai_suggest_tags"]["properties"]
            quote_properties = tool_schemas["novelai_quote"]["properties"]
            assert quote_properties["operation"]["enum"] == [
                "generate",
                "generate_stream",
                "encode_vibe",
                "augment",
                "upscale",
            ]
            assert quote_properties["request"]["type"] == "object"
            assert "cost_approved" not in quote_properties

            status = await session.call_tool("novelai_status", {})
            assert not status.isError
            assert result_data(status)["token_configured"] is False

            schema_result = await session.call_tool("novelai_schema", {"operation": "generate"})
            assert not schema_result.isError
            schema_data = result_data(schema_result)
            parameters_schema = schema_data["schema"]["definitions"]["image.RequestParameters"]
            assert len(parameters_schema["properties"]) == 54

            viewed = await session.call_tool("novelai_view_image", {"path": str(source_path)})
            assert not viewed.isError
            view_data = result_data(viewed)
            assert (view_data["width"], view_data["height"]) == (80, 48)
            image_item = next(item for item in viewed.content if item.type == "image")
            decoded_preview = base64.b64decode(image_item.data, validate=True)
            with Image.open(io.BytesIO(decoded_preview)) as preview_image:
                preview_image.load()
                assert preview_image.size == (64, 38)
                assert preview_image.getpixel((32, 19)) == (199, 42, 18)

            mask_result = await session.call_tool(
                "novelai_create_mask",
                {
                    "image_path": str(source_path),
                    "regions": [{"shape": "rectangle", "box": [12, 8, 30, 24]}],
                },
            )
            assert not mask_result.isError
            mask_data = result_data(mask_result)
            assert (mask_data["width"], mask_data["height"]) == (80, 48)
            assert "白色重绘，黑色保留" in mask_data["semantics"]
            with Image.open(mask_data["path"]) as mask:
                mask.load()
                assert mask.format == "PNG"
                assert mask.mode == "L"
                assert mask.size == (80, 48)
                assert mask.getpixel((20, 16)) == 255
                assert mask.getpixel((2, 2)) == 0
