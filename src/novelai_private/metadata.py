# SPDX-License-Identifier: AGPL-3.0-or-later
# Adapted Huishu implementation; see NOTICE for source attribution.
"""Read NovelAI generation metadata from an image without modifying it.

The parser extracts metadata only. It does not authenticate signed metadata.
The signature_verification result field is always not_performed.
"""

from __future__ import annotations

import gzip
import io
import json
import zlib
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STEALTH_COMPRESSED_MAGIC = b"stealth_pngcomp"
STEALTH_PLAIN_MAGIC = b"stealth_pnginfo"
STEALTH_HEADER_BYTES = len(STEALTH_COMPRESSED_MAGIC) + 4
MAX_PNG_CHUNK_BYTES = 2 * 1024 * 1024
MAX_PNG_TEXT_BYTES = 4 * 1024 * 1024
MAX_PNG_TEXT_VALUE_BYTES = 1 * 1024 * 1024
MAX_STEALTH_COMPRESSED_BYTES = 2 * 1024 * 1024
MAX_STEALTH_PLAIN_BYTES = 4 * 1024 * 1024
MAX_STEALTH_DECOMPRESSED_BYTES = 4 * 1024 * 1024
MAX_COMMENT_JSON_BYTES = 4 * 1024 * 1024

_SOURCE_PRIORITY = ("png_text", "alpha_stealth", "exif")
_COMMENT_MARKERS = {
    "actual_prompts",
    "model",
    "model_name",
    "model_hash",
    "request_type",
    "seed",
    "signed_hash",
    "v4_negative_prompt",
    "v4_prompt",
}
_EXIF_TAGS = (
    (ExifTags.Base.DocumentName, "Title"),
    (ExifTags.Base.ImageDescription, "Description"),
    (ExifTags.Base.Software, "Source"),
    (ExifTags.Base.Artist, "Artist"),
    (ExifTags.Base.Copyright, "Copyright"),
)


def extract_metadata(path: Path, include_raw: bool = True) -> dict[str, Any]:
    """Extract image metadata from one image path; never reads sidecar files."""
    if not isinstance(include_raw, bool):
        raise TypeError("include_raw must be a boolean")
    image_path = Path(path)
    warnings: list[str] = []
    raw_sources: dict[str, dict[str, Any]] = {}

    png_text, png_warnings = _read_png_text(image_path)
    warnings.extend(png_warnings)

    with Image.open(image_path) as image:
        image_info = {
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }

        if image.format == "PNG" and png_text:
            raw_sources["png_text"] = png_text

        if image.format in {"PNG", "WEBP"}:
            try:
                stealth = _read_alpha_stealth(image, warnings)
            except (OSError, ValueError, zlib.error, gzip.BadGzipFile):
                _append_warning(warnings, "Alpha-channel metadata could not be decoded.")
                stealth = None
            if stealth is not None:
                raw_sources["alpha_stealth"] = stealth

        if image.format in {"WEBP", "JPEG"}:
            try:
                exif_metadata = _read_exif_metadata(image)
            except (OSError, ValueError, TypeError, zlib.error):
                _append_warning(warnings, "EXIF metadata could not be decoded.")
                exif_metadata = None
            if exif_metadata:
                raw_sources["exif"] = exif_metadata

        candidates: dict[str, dict[str, Any]] = {}
        for source_id, raw in raw_sources.items():
            parameters = _comment_parameters(raw, source_id, warnings)
            if parameters is not None and _looks_like_novelai(raw, parameters, source_id):
                candidates[source_id] = parameters

        primary_source = next(
            (source_id for source_id in _SOURCE_PRIORITY if source_id in candidates),
            None,
        )
        parameters = candidates.get(primary_source, {}) if primary_source else {}
        primary_raw = raw_sources.get(primary_source, {}) if primary_source else {}

        if primary_source:
            for source_id, candidate in candidates.items():
                if source_id != primary_source and candidate != parameters:
                    _append_warning(
                        warnings,
                        f"Metadata sources disagree; using {primary_source} as the primary source.",
                    )
                    break

        result: dict[str, Any] = {
            "path": str(image_path),
            "image": image_info,
            "found": primary_source is not None,
            "primary_source": primary_source,
            "sources": list(raw_sources),
            "generation": _generation_fields(parameters, primary_raw),
            "parameters": parameters,
            "warnings": warnings,
            "signature_verification": "not_performed",
        }
        if include_raw:
            result["raw"] = raw_sources
        return result


def _read_png_text(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    warnings: list[str] = []
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            if stream.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                return values, warnings
            metadata_bytes = 0
            while True:
                chunk_start = stream.tell()
                header = stream.read(8)
                if len(header) != 8:
                    if header:
                        _append_warning(warnings, "PNG metadata ended unexpectedly.")
                    break
                length = int.from_bytes(header[:4], "big")
                chunk_type = header[4:]
                chunk_end = chunk_start + 12 + length
                if chunk_end > file_size:
                    _append_warning(warnings, "PNG chunk length exceeds the file boundary.")
                    break

                is_text = chunk_type in {b"tEXt", b"iTXt", b"zTXt"}
                if is_text:
                    metadata_bytes += length
                if is_text and (
                    length > MAX_PNG_CHUNK_BYTES or metadata_bytes > MAX_PNG_TEXT_BYTES
                ):
                    _append_warning(warnings, "PNG text metadata exceeds the extraction limit.")
                    stream.seek(chunk_end)
                elif is_text:
                    chunk_data = stream.read(length)
                    crc_bytes = stream.read(4)
                    expected_crc = int.from_bytes(crc_bytes, "big")
                    actual_crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
                    if actual_crc != expected_crc:
                        _append_warning(warnings, "A PNG text chunk has an invalid checksum.")
                    else:
                        try:
                            key, value = _decode_png_text_chunk(chunk_type, chunk_data)
                        except (ValueError, UnicodeError, zlib.error):
                            _append_warning(warnings, "A PNG text chunk could not be decoded.")
                        else:
                            if key in values and values[key] != value:
                                _append_warning(warnings, "PNG text chunks contain duplicate keys.")
                            values[key] = value
                else:
                    stream.seek(chunk_end)

                if chunk_type == b"IEND":
                    break
    except OSError:
        _append_warning(warnings, "PNG text metadata could not be read.")
    return values, warnings


def _decode_png_text_chunk(chunk_type: bytes, data: bytes) -> tuple[str, str]:
    if chunk_type == b"tEXt":
        keyword, separator, text = data.partition(b"\x00")
        if not separator or not keyword:
            raise ValueError("invalid tEXt chunk")
        return keyword.decode("latin-1"), text.decode("latin-1")

    keyword, separator, rest = data.partition(b"\x00")
    if not separator or not keyword:
        raise ValueError("invalid PNG text keyword")
    if chunk_type == b"zTXt":
        if len(rest) < 2 or rest[0] != 0:
            raise ValueError("unsupported zTXt compression method")
        text = _decompress_zlib_limited(rest[1:], MAX_PNG_TEXT_VALUE_BYTES)
        return keyword.decode("latin-1"), text.decode("latin-1")

    if len(rest) < 2:
        raise ValueError("invalid iTXt chunk")
    compression_flag, compression_method = rest[0], rest[1]
    if compression_flag not in {0, 1} or compression_method != 0:
        raise ValueError("unsupported iTXt compression")
    language_end = rest.find(b"\x00", 2)
    if language_end < 0:
        raise ValueError("invalid iTXt language tag")
    translated_end = rest.find(b"\x00", language_end + 1)
    if translated_end < 0:
        raise ValueError("invalid iTXt translated keyword")
    text = rest[translated_end + 1 :]
    if compression_flag:
        text = _decompress_zlib_limited(text, MAX_PNG_TEXT_VALUE_BYTES)
    return keyword.decode("latin-1"), text.decode("utf-8")


def _decompress_zlib_limited(data: bytes, limit: int) -> bytes:
    decoder = zlib.decompressobj()
    output = decoder.decompress(data, limit + 1)
    if len(output) > limit or decoder.unconsumed_tail:
        raise ValueError("decompressed metadata exceeds the extraction limit")
    output += decoder.flush()
    if len(output) > limit or not decoder.eof:
        raise ValueError("invalid or oversized compressed metadata")
    return output


def _read_alpha_stealth(image: Image.Image, warnings: list[str]) -> dict[str, Any] | None:
    width, height = image.size
    capacity_bytes = width * height // 8
    if capacity_bytes < STEALTH_HEADER_BYTES:
        return None

    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A").tobytes()
    header = _read_alpha_bytes(alpha, width, height, STEALTH_HEADER_BYTES)
    magic = header[: len(STEALTH_COMPRESSED_MAGIC)]
    if magic not in {STEALTH_COMPRESSED_MAGIC, STEALTH_PLAIN_MAGIC}:
        return None

    declared_bits = int.from_bytes(header[len(magic) :], "big")
    payload_length = declared_bits // 8
    if declared_bits % 8:
        _append_warning(warnings, "Alpha-channel metadata has a non-byte-aligned length.")
        return None
    compressed = magic == STEALTH_COMPRESSED_MAGIC
    limit = MAX_STEALTH_COMPRESSED_BYTES if compressed else MAX_STEALTH_PLAIN_BYTES
    if payload_length <= 0 or payload_length > limit:
        _append_warning(warnings, "Alpha-channel metadata length exceeds the extraction limit.")
        return None
    if STEALTH_HEADER_BYTES + payload_length > capacity_bytes:
        _append_warning(warnings, "Alpha-channel metadata length exceeds image capacity.")
        return None

    packet = _read_alpha_bytes(alpha, width, height, STEALTH_HEADER_BYTES + payload_length)
    payload = packet[STEALTH_HEADER_BYTES:]
    if compressed:
        try:
            payload = _decompress_gzip_limited(payload, MAX_STEALTH_DECOMPRESSED_BYTES)
        except (OSError, EOFError, zlib.error, ValueError):
            _append_warning(warnings, "Compressed alpha-channel metadata is invalid or oversized.")
            return None
    elif len(payload) > MAX_STEALTH_DECOMPRESSED_BYTES:
        _append_warning(warnings, "Alpha-channel metadata exceeds the extraction limit.")
        return None

    metadata = _json_object(payload)
    if metadata is None:
        _append_warning(warnings, "Alpha-channel metadata is not a JSON object.")
    return metadata


def _read_alpha_bytes(alpha: bytes, width: int, height: int, byte_count: int) -> bytes:
    if byte_count * 8 > width * height:
        raise ValueError("requested alpha payload exceeds image capacity")
    output = bytearray(byte_count)
    bit_index = 0
    for x in range(width):
        column_start = x
        for y in range(height):
            if bit_index >= byte_count * 8:
                return bytes(output)
            if alpha[y * width + column_start] & 1:
                output[bit_index // 8] |= 1 << (7 - bit_index % 8)
            bit_index += 1
    return bytes(output)


def _decompress_gzip_limited(data: bytes, limit: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as stream:
        output = stream.read(limit + 1)
        if len(output) > limit or stream.read(1):
            raise ValueError("gzip metadata exceeds the extraction limit")
        return output


def _read_exif_metadata(image: Image.Image) -> dict[str, Any] | None:
    exif = image.getexif()
    if not exif:
        return None

    result: dict[str, Any] = {}
    for tag, key in _EXIF_TAGS:
        if tag in exif:
            result[key] = _json_safe(exif[tag])

    try:
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except (KeyError, TypeError, ValueError):
        exif_ifd = {}
    user_comment = exif_ifd.get(ExifTags.Base.UserComment)
    if user_comment is None:
        user_comment = exif.get(ExifTags.Base.UserComment)
    comment = _decode_user_comment(user_comment)
    if comment:
        result["Comment"] = comment
    return result or None


def _decode_user_comment(value: Any) -> str | None:
    if isinstance(value, bytes):
        prefix = b"ASCII\x00\x00\x00"
        if not value.startswith(prefix):
            return None
        return value[len(prefix) :].rstrip(b"\x00").decode("ascii", errors="replace")
    return value if isinstance(value, str) and value else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _json_object(data: bytes | str) -> dict[str, Any] | None:
    if isinstance(data, bytes):
        if len(data) > MAX_COMMENT_JSON_BYTES:
            return None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        if len(data.encode("utf-8", errors="replace")) > MAX_COMMENT_JSON_BYTES:
            return None
        text = data
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _comment_parameters(
    raw: dict[str, Any], source_id: str, warnings: list[str]
) -> dict[str, Any] | None:
    value: Any = raw.get("Comment")
    if value is None:
        return raw if _COMMENT_MARKERS.intersection(raw) else None

    for _ in range(3):
        if isinstance(value, dict):
            nested = value.get("Comment")
            if isinstance(nested, str):
                parsed_nested = _json_object(nested)
                if parsed_nested is not None:
                    value = parsed_nested
                    continue
            return value
        if isinstance(value, str):
            parsed = _json_object(value)
            if parsed is not None:
                value = parsed
                continue
        _append_warning(warnings, f"{source_id} Comment is not a valid JSON object.")
        return None
    _append_warning(warnings, f"{source_id} Comment nesting exceeds the extraction limit.")
    return None


def _looks_like_novelai(raw: dict[str, Any], parameters: dict[str, Any], source_id: str) -> bool:
    software = _text(raw.get("Software"))
    if software is None and source_id == "exif":
        # The official EXIF mapping stores the Software tag under Source.
        software = _text(raw.get("Source"))
    if software is not None and "NovelAI" not in software:
        return False

    model = (
        _text(parameters.get("model"))
        or _text(parameters.get("model_name"))
        or _text(parameters.get("Source"))
        or _text(raw.get("Source"))
    )
    looks_like = (
        (model is not None and model.startswith("nai-diffusion-"))
        or "v4_prompt" in parameters
        or "actual_prompts" in parameters
    )
    return ("NovelAI" in software) if software is not None else looks_like


def _generation_fields(parameters: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    generation: dict[str, Any] = {}
    model = _text(parameters.get("model")) or _text(parameters.get("model_name"))
    source = _text(parameters.get("Source")) or _text(raw.get("Source"))
    if source is None:
        source = _text(raw.get("Software"))
    if model is not None:
        generation["model"] = model
    if source is not None:
        generation["Source"] = source

    prompt_caption = _read_caption(parameters.get("v4_prompt"))
    negative_caption = _read_caption(parameters.get("v4_negative_prompt"))
    prompt = (
        _text(prompt_caption.get("base_caption") if prompt_caption else None)
        or _text(parameters.get("prompt"))
        or _text(raw.get("Description"))
    )
    negative_prompt = (
        _text(negative_caption.get("base_caption") if negative_caption else None)
        or _text(parameters.get("uc"))
        or _text(parameters.get("negative_prompt"))
    )
    if prompt is not None:
        generation["prompt"] = prompt
    if negative_prompt is not None:
        generation["negative_prompt"] = negative_prompt

    for key in (
        "seed",
        "steps",
        "scale",
        "sampler",
        "noise_schedule",
        "cfg_rescale",
        "strength",
        "width",
        "height",
        "n_samples",
        "model_hash",
        "request_type",
        "version",
    ):
        if key in parameters:
            generation[key] = parameters[key]

    v4_prompt = parameters.get("v4_prompt")
    v4_negative_prompt = parameters.get("v4_negative_prompt")
    if isinstance(v4_prompt, dict):
        generation["v4_prompt"] = v4_prompt
        generation["use_character_positions"] = v4_prompt.get("use_coords") is True
    if isinstance(v4_negative_prompt, dict):
        generation["v4_negative_prompt"] = v4_negative_prompt

    characters = _read_characters(prompt_caption, negative_caption)
    if characters:
        generation["characters"] = characters

    actual_prompts = parameters.get("actual_prompts")
    if isinstance(actual_prompts, dict):
        generation["actual_prompts"] = actual_prompts
        actual_prompt_caption = _read_caption(actual_prompts.get("prompt"))
        actual_negative_caption = _read_caption(actual_prompts.get("negative_prompt"))
        actual_prompt: dict[str, Any] = {}
        actual_text = _text(
            actual_prompt_caption.get("base_caption") if actual_prompt_caption else None
        )
        actual_negative = _text(
            actual_negative_caption.get("base_caption") if actual_negative_caption else None
        )
        if actual_text is not None:
            actual_prompt["prompt"] = actual_text
        if actual_negative is not None:
            actual_prompt["negative_prompt"] = actual_negative
        if isinstance(v4_prompt, dict):
            actual_prompt["use_character_positions"] = v4_prompt.get("use_coords") is True
        actual_characters = _read_characters(actual_prompt_caption, actual_negative_caption)
        if actual_characters:
            actual_prompt["characters"] = actual_characters
        generation["actual_prompt"] = actual_prompt

    reference_images = parameters.get("reference_image_multiple")
    if isinstance(reference_images, list):
        strengths = parameters.get("reference_strength_multiple")
        information = parameters.get("reference_information_extracted_multiple")
        vibes = []
        for index, encoded in enumerate(reference_images):
            if not isinstance(encoded, str) or not encoded:
                continue
            vibes.append(
                {
                    "encodedData": encoded,
                    "strength": _array_number(strengths, index),
                    "informationExtracted": _array_number(information, index),
                }
            )
        if vibes:
            generation["vibes"] = vibes
            generation["vibe_transfer"] = True
    return generation


def _read_caption(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    caption = value.get("caption")
    if isinstance(caption, dict):
        return caption
    if "base_caption" in value or "char_captions" in value:
        return value
    return None


def _read_characters(
    prompt_caption: dict[str, Any] | None,
    negative_caption: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if prompt_caption is None or not isinstance(prompt_caption.get("char_captions"), list):
        return None
    negative_characters = (
        negative_caption.get("char_captions")
        if negative_caption is not None and isinstance(negative_caption.get("char_captions"), list)
        else []
    )
    result = []
    for index, value in enumerate(prompt_caption["char_captions"]):
        prompt_value = value if isinstance(value, dict) else {}
        negative_value = (
            negative_characters[index]
            if index < len(negative_characters) and isinstance(negative_characters[index], dict)
            else {}
        )
        centers = prompt_value.get("centers")
        first_center = centers[0] if isinstance(centers, list) and centers else {}
        if not isinstance(first_center, dict):
            first_center = {}
        result.append(
            {
                "prompt": _text(prompt_value.get("char_caption")) or "",
                "negative_prompt": _text(negative_value.get("char_caption")) or "",
                "center": {
                    "x": _default_coordinate(first_center.get("x")),
                    "y": _default_coordinate(first_center.get("y")),
                },
            }
        )
    return result or None


def _default_coordinate(value: Any) -> int | float:
    coordinate = _number(value)
    return 0.5 if coordinate is None else coordinate


def _array_number(value: Any, index: int) -> int | float | None:
    if not isinstance(value, list) or index >= len(value):
        return None
    return _number(value[index])


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _append_warning(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)
