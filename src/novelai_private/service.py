from __future__ import annotations

import base64
import binascii
import io
import json
import re
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .billing import quote_cost, unknown
from .client import ApiReply, NovelAIClient
from .config import NovelAIError, Settings
from .media import bind_files, make_mask, merge_patch, open_image, preview
from .metadata import extract_metadata
from .schema import validate


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ImageService:
    def __init__(self, settings: Settings, client: NovelAIClient | None = None):
        self.settings = settings
        self.client = client or NovelAIClient(settings)

    def status(self) -> dict:
        try:
            self.settings.token()
            configured = True
        except NovelAIError:
            configured = False
        return {
            "token_configured": configured,
            "config_path": str(self.settings.path),
            "token_file": str(self.settings.token_file),
            "output_dir": str(self.settings.output_dir),
            "endpoint_host_configured": True,
            "remote_verified": False,
            "message": "此检查只读本地配置，不发送网络请求。",
        }

    def new_run(self) -> tuple[str, Path]:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        directory = self.settings.output_dir / run_id
        directory.mkdir(parents=True, exist_ok=False)
        return run_id, directory

    def load_record(self, run_id: str) -> tuple[Path, dict]:
        if not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[a-f0-9]{12}", run_id):
            raise NovelAIError("生成记录 ID 无效。")
        directory = self.settings.output_dir / run_id
        path = directory / "record.json"
        if not path.is_file():
            raise NovelAIError("找不到该生成记录。")
        return directory, json.loads(path.read_text("utf-8"))

    async def execute(
        self,
        operation: str,
        request: dict,
        files: list[dict] | None = None,
        parent_id: str | None = None,
        cost_approved: bool = False,
        high_resolution_approved: bool = False,
    ) -> dict:
        payload = self.prepare_request(operation, request, files)
        plan = None
        if operation != "suggest_tags":
            plan = self.cost_preview(operation, payload)
            if not cost_approved:
                billing = (
                    unknown("先指定明确的 width 和 height，才能判断像素量和点数。")
                    if plan["missing_resolution"]
                    else await quote_cost(self.client, operation, payload)
                )
                plan.update(billing)
                if billing["cost_status"] != "free":
                    return {"status": "approval_required", **plan}
            else:
                plan.update(unknown("本次点数消耗已由用户明确批准。"))
            if plan["needs_high_resolution_approval"] and not high_resolution_approved:
                return {"status": "high_resolution_approval_required", **plan}
            if plan.get("missing_resolution"):
                raise NovelAIError("生成前需指定明确的 width 和 height，以确认像素量和点数风险。")
        # Fail before creating a run if credentials are missing.
        self.settings.token()
        run_id, directory = self.new_run()
        record = {
            "id": run_id,
            "operation": operation,
            "parent_id": parent_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "running",
            "artifacts": [],
        }
        if plan is not None:
            record["billing"] = plan
        # This is the exact provider body, including base64 assets, with no HTTP credentials.
        write_json(directory / "request.json", payload)
        write_json(directory / "record.json", record)
        try:
            reply = await self.client.request(operation, payload)
            record["correlation_id"] = reply.correlation_id
            if reply.anlas_cost is not None:
                record["actual_anlas"] = reply.anlas_cost
            artifacts = self.save_reply(directory, operation, reply)
            record.update(status="complete", artifacts=artifacts)
        except BaseException as e:
            record["status"] = "interrupted" if not isinstance(e, Exception) else "failed"
            record["error"] = str(e) if isinstance(e, NovelAIError) else type(e).__name__
            write_json(directory / "record.json", record)
            raise
        write_json(directory / "record.json", record)
        return self.summary(directory, record)

    def prepare_request(
        self, operation: str, request: dict, files: list[dict] | None = None
    ) -> dict:
        payload = bind_files(request, files, self.settings)
        if operation in {"generate", "generate_stream"}:
            parameters = payload.get("parameters")
            if isinstance(parameters, dict):
                parameters.setdefault("steps", 28)
                parameters.setdefault("scale", 6)
                parameters.setdefault("n_samples", 1)
            # Huishu/native NovelAI places this flag at the top level. Omitting it
            # on a private source can forfeit the user's free Opus generation.
            payload.setdefault("use_new_shared_trial", True)
        validate(operation, payload)
        return payload

    async def quote(self, operation: str, request: dict, files: list[dict] | None = None) -> dict:
        payload = self.prepare_request(operation, request, files)
        plan = self.cost_preview(operation, payload)
        billing = (
            unknown("先指定明确的 width 和 height，才能判断像素量和点数。")
            if plan["missing_resolution"]
            else await quote_cost(self.client, operation, payload)
        )
        return {"status": "quote", **plan, **billing}

    @staticmethod
    def cost_preview(operation: str, payload: dict) -> dict:
        parameters = (
            payload.get("parameters", {})
            if operation in {"generate", "generate_stream"}
            else payload
        )
        width = parameters.get("width")
        height = parameters.get("height")
        generates_image = operation in {"generate", "generate_stream", "augment", "upscale"}
        missing = (
            generates_image
            and operation != "upscale"
            and not (
                isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0
            )
        )
        pixels = width * height if isinstance(width, int) and isinstance(height, int) else None
        unknown_output = operation == "upscale"
        exceeds = generates_image and pixels is not None and pixels > 1024 * 1024
        return {
            "operation": operation,
            "model": payload.get("model"),
            "action": payload.get("action"),
            "width": width,
            "height": height,
            "pixels_per_image": pixels,
            "pixel_limit_without_approval": 1024 * 1024,
            "n_samples": parameters.get("n_samples", 1),
            "steps": parameters.get("steps"),
            "scale": parameters.get("scale"),
            "missing_resolution": missing,
            "output_resolution_unknown": unknown_output,
            "needs_high_resolution_approval": exceeds or unknown_output,
            "message": "报价/订阅查询不生图。免费请求按用户生图指令执行；收费或未知先申请本次同意。"
            "超过像素上限或输出尺寸未知的放大还需额外同意。",
        }

    def save_reply(self, directory: Path, operation: str, reply: ApiReply) -> list[dict]:
        if operation == "encode_vibe":
            (directory / "vibe.bin").write_bytes(reply.data)
            return [{"path": str(directory / "vibe.bin"), "kind": "vibe"}]
        if operation == "suggest_tags":
            data = json.loads(reply.data)
            write_json(directory / "response.json", data)
            return [{"path": str(directory / "response.json"), "kind": "json", "data": data}]
        images: list[tuple[str, bytes, dict]] = []
        if reply.stream_images is not None:
            suffix = ".sse" if reply.content_type == "text/event-stream" else ".msgpack"
            (directory / ("response" + suffix)).write_bytes(reply.data)
            items = reply.stream_images
        elif zipfile.is_zipfile(io.BytesIO(reply.data)):
            (directory / "response.zip").write_bytes(reply.data)
            with zipfile.ZipFile(io.BytesIO(reply.data)) as archive:
                infos = archive.infolist()
                total_size = sum(entry.file_size for entry in infos)
                if total_size > 256 * 1024 * 1024 or len(infos) > 256:
                    raise NovelAIError("返回的压缩包过大，已保留原包。")
                for entry in infos:
                    if not entry.is_dir() and Path(entry.filename).suffix.lower() in {
                        ".png",
                        ".webp",
                        ".jpg",
                        ".jpeg",
                    }:
                        images.append((Path(entry.filename).name, archive.read(entry), {}))
            items = []
        elif reply.content_type.startswith("image/"):
            suffix = ".webp" if reply.content_type == "image/webp" else ".png"
            images.append(("image" + suffix, reply.data, {}))
            items = []
        else:
            (directory / "response.json").write_bytes(reply.data)
            try:
                data = json.loads(reply.data)
                items = data.get("images", []) if isinstance(data, dict) else data
            except ValueError as e:
                raise NovelAIError("NovelAI 返回了无法识别的图片响应；已保留原始响应。") from e
        for index, item in enumerate(items):
            value = item if isinstance(item, str) else item.get("image", "")
            try:
                encoded = value.split(",", 1)[-1] if value.startswith("data:") else value
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error, AttributeError) as e:
                raise NovelAIError("返回的图片 base64 无效；已保留原始响应。") from e
            image = open_image(data)
            suffix = ".webp" if image.format == "WEBP" else ".png"
            metadata = (
                {k: item[k] for k in ("seed", "index") if k in item}
                if isinstance(item, dict)
                else {}
            )
            images.append((f"image_{index}{suffix}", data, metadata))
        if not images:
            raise NovelAIError("响应中没有最终图片；已保留原始响应。")
        artifacts = []
        for index, (name, data, metadata) in enumerate(images):
            thumbnail, dimensions = preview(data, self.settings.preview_max_side)
            # Never use archive paths; prefixes also prevent duplicate filenames overwriting.
            destination = directory / f"{index:02d}_{name}"
            destination.write_bytes(data)
            thumbnail_path = directory / f"preview_{index:02d}.png"
            thumbnail_path.write_bytes(thumbnail)
            artifacts.append(
                {
                    "path": str(destination),
                    "preview_path": str(thumbnail_path),
                    "kind": "image",
                    **dimensions,
                    **metadata,
                }
            )
        return artifacts

    @staticmethod
    def summary(directory: Path, record: dict) -> dict:
        return {
            **record,
            "record_path": str(directory / "record.json"),
            "request_path": str(directory / "request.json"),
        }

    async def revise(
        self,
        run_id: str,
        patch: dict,
        files: list[dict] | None = None,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        use_output_as_input: bool = False,
        image_index: int = 0,
        cost_approved: bool = False,
        high_resolution_approved: bool = False,
    ) -> dict:
        directory, record = self.load_record(run_id)
        if record["operation"] not in {"generate", "generate_stream"}:
            raise NovelAIError("连续调整仅适用于生成记录。其他操作请创建新的官方请求。")
        request = merge_patch(json.loads((directory / "request.json").read_text("utf-8")), patch)
        params = request.setdefault("parameters", {})
        if prompt is not None:
            request["input"] = prompt
            if "v4_prompt" in params:
                params["v4_prompt"].setdefault("caption", {})["base_caption"] = prompt
        if negative_prompt is not None:
            params["negative_prompt"] = negative_prompt
            if "v4_negative_prompt" in params:
                params["v4_negative_prompt"].setdefault("caption", {})["base_caption"] = (
                    negative_prompt
                )
        bindings = list(files or [])
        if use_output_as_input:
            outputs = [a for a in record["artifacts"] if a["kind"] == "image"]
            if not 0 <= image_index < len(outputs):
                raise NovelAIError("image_index 超出已生成图片范围。")
            request["action"] = "img2img"
            request["model"] = request["model"].removesuffix("-inpainting")
            params.pop("mask", None)
            params.pop("img2img", None)
            params.setdefault("strength", 0.5)
            params.setdefault("noise", 0)
            bindings.append({"pointer": "/parameters/image", "path": outputs[image_index]["path"]})
        return await self.execute(
            record["operation"],
            request,
            bindings,
            parent_id=run_id,
            cost_approved=cost_approved,
            high_resolution_approved=high_resolution_approved,
        )

    def image_path(
        self, path: str | None = None, run_id: str | None = None, image_index: int = 0
    ) -> Path:
        if not path and run_id:
            _, record = self.load_record(run_id)
            outputs = [a for a in record["artifacts"] if a["kind"] == "image"]
            if not 0 <= image_index < len(outputs):
                raise NovelAIError("该记录没有对应图片。")
            path = outputs[image_index]["path"]
        if not path:
            raise NovelAIError("需要 path 或 run_id。")
        return self.settings.check_file(path)

    def metadata(
        self,
        path: str | None = None,
        run_id: str | None = None,
        image_index: int = 0,
        include_raw: bool = True,
    ) -> dict:
        return extract_metadata(self.image_path(path, run_id, image_index), include_raw=include_raw)

    def history(self, limit: int = 10, run_id: str | None = None) -> dict:
        if run_id:
            directory, record = self.load_record(run_id)
            return self.summary(directory, record)
        records = []
        for path in sorted(self.settings.output_dir.glob("*/record.json"), reverse=True)[:limit]:
            try:
                records.append(self.summary(path.parent, json.loads(path.read_text("utf-8"))))
            except (OSError, ValueError):
                continue
        return {"records": records}

    def create_mask(self, image_path: str, regions: list[dict], feather: float = 0) -> dict:
        image = open_image(self.settings.check_file(image_path).read_bytes())
        data = make_mask(image.size, regions, feather)
        run_id, directory = self.new_run()
        path = directory / "mask.png"
        path.write_bytes(data)
        return {
            "id": run_id,
            "path": str(path),
            "width": image.width,
            "height": image.height,
            "semantics": "白色重绘，黑色保留；像素坐标与原图一致。",
        }
