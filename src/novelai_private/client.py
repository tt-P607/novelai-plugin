from __future__ import annotations

import base64
import json
import re
import secrets
import string
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import msgpack

from .config import NovelAIError, Settings
from .schema import OPERATIONS

MAX_RESPONSE_BYTES = 256 * 1024 * 1024


@dataclass
class ApiReply:
    data: bytes
    content_type: str
    correlation_id: str
    stream_images: list[dict] | None = None
    anlas_cost: int | None = None


def stream_frames(raw: bytes) -> list[dict]:
    """Only final frames count as results; intermediates never masquerade as success."""
    results: dict[int, dict] = {}
    for block in raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n\n"):
        event = "message"
        lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                lines.append(line[5:].strip())
        if not lines:
            continue
        text = "\n".join(lines)
        if text == "[DONE]":
            continue
        try:
            frame = json.loads(text)
        except json.JSONDecodeError as e:
            raise NovelAIError("流式响应包含无法解析的 JSON；已停止，未自动重试。") from e
        if not isinstance(frame, dict):
            continue
        kind = str(frame.get("event_type", event)).lower()
        if kind == "error" or event.lower() == "error":
            raise NovelAIError("NovelAI 在流式生成中返回错误；未自动重试。")
        if kind in {"final", "complete", "completed"}:
            images = frame.get("images")
            if images:
                for index, item in enumerate(images):
                    results[index] = {"image": item} if isinstance(item, str) else item
            elif isinstance(frame.get("image"), str):
                index = frame.get("samp_ix", frame.get("index", 0))
                results[int(index)] = {
                    "image": frame["image"],
                    "seed": frame.get("seed"),
                    "index": index,
                }
    if not results:
        raise NovelAIError("流式响应已结束，但没有最终图片。不能将预览帧作为生成成功。")
    return [results[k] for k in sorted(results)]


def msgpack_frames(raw: bytes) -> list[dict]:
    """Decode the length-prefixed frames used by NovelAI's public image frontend."""
    results: dict[int, dict] = {}
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 4:
            raise NovelAIError("MessagePack 流在帧头处截断，未自动重试。")
        size = int.from_bytes(raw[offset : offset + 4], "big")
        offset += 4
        if size <= 0 or offset + size > len(raw):
            raise NovelAIError("MessagePack 流在帧内容处截断，未自动重试。")
        try:
            frame = msgpack.unpackb(raw[offset : offset + size], raw=False, strict_map_key=False)
        except (ValueError, msgpack.UnpackException) as e:
            raise NovelAIError("无法解码 MessagePack 生成事件，未自动重试。") from e
        offset += size
        if not isinstance(frame, dict):
            raise NovelAIError("MessagePack 事件不是对象。")
        kind = frame.get("event_type")
        if kind == "error":
            raise NovelAIError("NovelAI 在 MessagePack 生成中返回错误，未自动重试。")
        if kind == "final":
            image = frame.get("image")
            if isinstance(image, list):
                image = bytes(image)
            if not isinstance(image, bytes):
                raise NovelAIError("MessagePack 最终帧不含二进制图片。")
            index = int(frame.get("samp_ix", 0))
            results[index] = {"index": index, "image": base64.b64encode(image).decode("ascii")}
    if not results:
        raise NovelAIError("MessagePack 流没有最终图片。")
    return [results[k] for k in sorted(results)]


class NovelAIClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    async def subscription(self) -> dict:
        """Read live entitlement only; this endpoint never generates an image."""
        return await self._billing_json("GET", self.settings.base_url + "/user/subscription")

    async def gateway_quote(self, body: dict) -> dict | None:
        """Huishu's same-origin quote, including platform fees and Key source routing."""
        if not urlsplit(self.settings.base_url).path.endswith("/origin"):
            return None
        url = self.settings.base_url.removesuffix("/origin") + "/api/quote"
        result = await self._billing_json("POST", url, body)
        quote = result.get("quote")
        if not isinstance(quote, dict):
            raise NovelAIError("绘枢未返回有效报价；尚未发起生成。")
        return quote

    async def _billing_json(self, method: str, url: str, body: dict | None = None) -> dict:
        token = self.settings.token()
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(20, connect=10),
                proxy=self.settings.proxy,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    method,
                    url,
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                    **({"json": body} if body is not None else {}),
                ) as response:
                    if not response.is_success:
                        raise NovelAIError(
                            f"无法读取计费状态（HTTP {response.status_code}）；尚未发起生成。"
                        )
                    chunks = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > 1024 * 1024:
                            raise NovelAIError("计费状态响应过大；尚未发起生成。")
                        chunks.append(chunk)
                    result = json.loads(b"".join(chunks))
                    if not isinstance(result, dict):
                        raise ValueError("object required")
                    return result
        except (httpx.RequestError, ValueError) as e:
            raise NovelAIError("无法确认当前计费状态；尚未发起生成，也未自动重试。") from e

    async def request(self, operation: str, body: dict[str, Any]) -> ApiReply:
        method, path, _ = OPERATIONS[operation]
        token = self.settings.token()
        correlation = "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(6)
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "x-correlation-id": correlation,
            "User-Agent": "NovelAIPrivate/0.1",
            "Accept": "text/event-stream" if operation == "generate_stream" else "application/zip",
        }
        if operation == "suggest_tags":
            headers["Accept"] = "application/json"
        kwargs = {"params": body} if method == "GET" else {"json": body}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.timeout_seconds, connect=30),
                proxy=self.settings.proxy,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                async with client.stream(
                    method, self.settings.base_url + path, headers=headers, **kwargs
                ) as response:
                    chunks = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_RESPONSE_BYTES:
                            raise NovelAIError("服务响应超过 256 MiB，已停止读取；未自动重试。")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    if not response.is_success:
                        message = self._error(response.status_code, data, token)
                        raise NovelAIError(f"{message}（关联编号 {correlation}）。未自动重试。")
                    content_type = response.headers.get("content-type", "").split(";")[0]
                    raw_cost = response.headers.get("x-anlas-cost", "")
                    actual_cost = int(raw_cost) if raw_cost.isdigit() else None
        except httpx.TimeoutException as e:
            raise NovelAIError(
                f"NovelAI 请求超时（关联编号 {correlation}）。服务端可能仍在处理，未自动重试。"
            ) from e
        except httpx.RequestError as e:
            raise NovelAIError(
                f"无法连接 NovelAI（{type(e).__name__}，关联编号 {correlation}）。未自动重试。"
            ) from e
        images = None
        if content_type == "text/event-stream":
            images = stream_frames(data)
        elif (
            operation == "generate_stream" and body.get("parameters", {}).get("stream") == "msgpack"
        ):
            images = msgpack_frames(data)
        if images is not None and len(images) < body.get("parameters", {}).get("n_samples", 1):
            raise NovelAIError("流式响应缺少部分样本的最终图片，未自动重试。")
        return ApiReply(data, content_type, correlation, images, actual_cost)

    @staticmethod
    def _error(status: int, data: bytes, token: str) -> str:
        labels = {
            400: "请求参数被 NovelAI 拒绝",
            401: "Token 无效或已过期",
            402: "NovelAI 额度不足",
            403: "NovelAI 拒绝访问",
            429: "NovelAI 正在限流，请稍后由用户决定是否再次生成",
        }
        label = labels.get(status, f"NovelAI 返回 HTTP {status}")
        try:
            details = json.loads(data)
            detail = details.get("message", details.get("error", ""))
            if isinstance(detail, str):
                detail = detail.replace(token, "[已隐藏]")
                detail = re.sub(r"[A-Za-z0-9+/=_-]{80,}", "[长数据已隐藏]", detail)
                if detail:
                    return f"{label}：{detail[:500]}"
        except (ValueError, AttributeError):
            pass
        return label
