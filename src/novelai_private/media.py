from __future__ import annotations

import base64
import copy
import io
import json
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .config import NovelAIError, Settings


def open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        if image.format not in {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}:
            raise NovelAIError("素材不是支持的图像格式。")
        image.load()
        normalized = ImageOps.exif_transpose(image)
        normalized.format = image.format
        return normalized
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise NovelAIError("无法读取该图片。") from e


def png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def precise_reference(image: Image.Image) -> Image.Image:
    """Official reference canvases, fitted without cropping and padded black."""
    ratio = image.width / image.height
    # Local choice: choose the official canvas with the least aspect-ratio padding.
    size = min(
        [(1536, 1024), (1024, 1536), (1472, 1472)],
        key=lambda s: max(ratio / (s[0] / s[1]), (s[0] / s[1]) / ratio),
    )
    canvas = Image.new("RGB", size, "black")
    fitted = ImageOps.contain(image.convert("RGBA"), size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2), fitted)
    return canvas


def set_pointer(document: dict, pointer: str, value: Any) -> None:
    if not pointer.startswith("/"):
        raise NovelAIError("文件绑定使用 JSON Pointer，例如 /parameters/image。")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]
    node: Any = document
    try:
        for i, key in enumerate(parts):
            last = i == len(parts) - 1
            if isinstance(node, list):
                index = int(key)
                if index < 0 or index > len(node):
                    raise ValueError()
                if index == len(node):
                    node.append(None if last else ([] if parts[i + 1].isdigit() else {}))
                if last:
                    node[index] = value
                else:
                    node = node[index]
            elif isinstance(node, dict):
                if last:
                    node[key] = value
                else:
                    node = node.setdefault(key, [] if parts[i + 1].isdigit() else {})
            else:
                raise ValueError()
    except (ValueError, IndexError, TypeError) as e:
        raise NovelAIError(f"无法应用文件绑定位置：{pointer}") from e


def bind_files(body: dict, bindings: list[dict] | None, settings: Settings) -> dict:
    body = copy.deepcopy(body)
    for binding in bindings or []:
        path = settings.check_file(binding["path"])
        mode = binding.get("mode", "image")
        data = path.read_bytes()
        if mode == "image":
            image = open_image(data)
            if binding.get("prepare") == "precise_reference":
                image = precise_reference(image)
            elif binding.get("prepare", "none") != "none":
                raise NovelAIError("未知的图片预处理方式。")
            data = png_bytes(image)
        elif mode == "vibe":
            if path.suffix.lower() in {".naiv4vibe", ".naiv4vibebundle"}:
                try:
                    container = json.loads(data)
                    if container.get("identifier") == "novelai-vibe-transfer-bundle":
                        container = container["vibes"][binding.get("vibe_index", 0)]
                    if container.get("identifier") != "novelai-vibe-transfer":
                        raise NovelAIError("不是可识别的 NovelAI Vibe 文件。")
                    model = body.get("model", "").removesuffix("-inpainting")
                    family = {
                        "nai-diffusion-4-curated-preview": "v4curated",
                        "nai-diffusion-4-full": "v4full",
                        "nai-diffusion-4-5-curated": "v4-5curated",
                        "nai-diffusion-4-5-full": "v4-5full",
                    }.get(model, model)
                    encodings = container.get("encodings", {})
                    values = encodings.get(family, encodings.get(model, {}))
                    wanted = binding.get("information_extracted")
                    entries = [v for v in values.values() if isinstance(v, dict)]
                    if wanted is not None:
                        entries = [
                            v
                            for v in entries
                            if abs(v.get("params", {}).get("information_extracted", -1) - wanted)
                            < 1e-6
                        ]
                    if len(entries) > 1:
                        raise NovelAIError("Vibe 文件有多个编码，请指定 information_extracted。")
                    entry = entries[0] if entries else None
                    if not entry:
                        raise NovelAIError("该 .naiv4vibe 没有匹配的模型和信息提取量编码。")
                    encoded = entry["encoding"]
                    data = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
                    raise NovelAIError(
                        "无法读取 .naiv4vibe 编码；可用 encode_vibe 输出的 .bin。"
                    ) from e
            elif path.suffix.lower() not in {".bin", ".vibe"}:
                raise NovelAIError("vibe 绑定仅接受编码后的 .bin、.vibe 或 .naiv4vibe。")
        else:
            raise NovelAIError("文件绑定 mode 必须是 image 或 vibe。")
        set_pointer(body, binding["pointer"], base64.b64encode(data).decode("ascii"))
    return body


def merge_patch(original: Any, patch: Any) -> Any:
    """RFC 7396: dictionaries merge, arrays replace, null removes a key."""
    if not isinstance(patch, dict):
        return copy.deepcopy(patch)
    result = copy.deepcopy(original) if isinstance(original, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def preview(data: bytes, max_side: int) -> tuple[bytes, dict]:
    image = open_image(data)
    info = {"width": image.width, "height": image.height, "mode": image.mode}
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return png_bytes(image), info


def make_mask(size: tuple[int, int], regions: list[dict], feather: float = 0) -> bytes:
    from PIL import ImageFilter

    if min(size) <= 0 or size[0] * size[1] > 64_000_000:
        raise NovelAIError("遮罩尺寸无效或过大。")
    if feather < 0:
        raise NovelAIError("羽化半径不能小于 0。")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        shape = region.get("shape", "rectangle")
        if shape == "polygon":
            points = [tuple(p) for p in region["points"]]
            if len(points) < 3:
                raise NovelAIError("多边形至少需要三个点。")
            draw.polygon(points, fill=255)
        elif shape in {"rectangle", "ellipse"}:
            box = tuple(region["box"])
            if len(box) != 4 or box[2] < box[0] or box[3] < box[1]:
                raise NovelAIError("box 应为 [左, 上, 右, 下] 像素坐标。")
            getattr(draw, shape)(box, fill=255)
        else:
            raise NovelAIError("遮罩形状支持 rectangle、ellipse、polygon。")
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return png_bytes(mask)
