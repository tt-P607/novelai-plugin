from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft7Validator

from .config import NovelAIError

OPERATIONS = {
    "generate": ("POST", "/ai/generate-image", "image.ImageGenerationRequest"),
    "generate_stream": ("POST", "/ai/generate-image-stream", "image.ImageGenerationRequest"),
    "encode_vibe": ("POST", "/ai/encode-vibe", "image.EncodeVibeRequest"),
    "augment": ("POST", "/ai/augment-image", "image.AugmentImageRequest"),
    "upscale": ("POST", "/ai/upscale", "image.UpscaleRequest"),
    "suggest_tags": ("GET", "/ai/generate-image/suggest-tags", None),
}


@lru_cache
def official_document() -> dict[str, Any]:
    return json.loads(
        files("novelai_private").joinpath("data/novelai-image-openapi.json").read_text("utf-8")
    )


def operation_schema(operation: str) -> dict[str, Any]:
    if operation not in OPERATIONS:
        raise NovelAIError(f"未知图像操作。可用：{', '.join(OPERATIONS)}")
    method, path, definition = OPERATIONS[operation]
    doc = official_document()
    if definition:
        result = copy.deepcopy(doc["definitions"][definition])
        result["definitions"] = doc["definitions"]
    else:
        params = doc["paths"][path][method.lower()]["parameters"]
        query = [p for p in params if p["in"] == "query"]
        result = {
            "type": "object",
            "properties": {
                p["name"]: {k: v for k, v in p.items() if k in {"type", "description", "enum"}}
                for p in query
            },
            "required": [p["name"] for p in query if p.get("required")],
        }
    return result


def describe(operation: str = "catalog", definition: str | None = None) -> dict[str, Any]:
    doc = official_document()
    if definition:
        if definition not in doc["definitions"]:
            raise NovelAIError("找不到该定义；使用 image.RequestParameters 查看所有生成参数。")
        return {"definition": definition, "schema": doc["definitions"][definition]}
    if operation == "catalog":
        return {
            "source": "https://image.novelai.net/docs/doc.json",
            "snapshot_date": "2026-10-20",
            "operations": {k: {"method": v[0], "path": v[1]} for k, v in OPERATIONS.items()},
            "generation_parameters": doc["definitions"]["image.RequestParameters"],
            "definitions": [k for k in doc["definitions"] if k.startswith("image.")],
            "unknown_fields": "Preserved and sent unchanged; upstream decides support.",
        }
    schema = operation_schema(operation)
    schema["definitions"] = {k: v for k, v in doc["definitions"].items() if k.startswith("image.")}
    return {"operation": operation, "schema": schema}


def validate(operation: str, body: dict[str, Any]) -> None:
    if not isinstance(body, dict):
        raise NovelAIError("request 必须是官方格式的 JSON 对象。")
    errors = sorted(Draft7Validator(operation_schema(operation)).iter_errors(body), key=str)
    if errors:
        # Never stringify ValidationError: it contains the original request and image base64.
        err = errors[0]
        field = "/".join(str(p) for p in err.absolute_path) or "request"
        raise NovelAIError(f"参数格式不符合官方 schema：{field}（{err.validator}）。")
    if operation in {"generate", "generate_stream"}:
        if not all(k in body for k in ("input", "model", "action", "parameters")):
            raise NovelAIError("生成 request 需要 input、model、action、parameters。")
