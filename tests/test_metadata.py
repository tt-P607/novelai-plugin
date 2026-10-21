from __future__ import annotations

import gzip
import hashlib
import json
import struct
from pathlib import Path

import pytest
from PIL import ExifTags, Image, PngImagePlugin

from novelai_private.metadata import (
    MAX_STEALTH_DECOMPRESSED_BYTES,
    STEALTH_COMPRESSED_MAGIC,
    STEALTH_PLAIN_MAGIC,
    extract_metadata,
)


def sample_parameters() -> dict:
    return {
        "model_name": "NovelAI Diffusion V5",
        "model_hash": "0ADF9AB7",
        "request_type": "PromptGenerateRequest",
        "prompt": "fallback prompt",
        "uc": "fallback negative",
        "seed": 2026102102,
        "steps": 28,
        "scale": 6,
        "sampler": "k_euler_ancestral",
        "noise_schedule": "karras",
        "cfg_rescale": 0.15,
        "strength": 0.72,
        "width": 832,
        "height": 1216,
        "v4_prompt": {
            "caption": {
                "base_caption": "雨の窓辺，猫 🐈",
                "char_captions": [
                    {
                        "char_caption": "青い髪の人物",
                        "centers": [{"x": 0.0, "y": 0.75}],
                        "vendor_character_key": "keep-me",
                    }
                ],
            },
            "use_coords": True,
            "vendor_prompt_key": {"keep": "this"},
        },
        "v4_negative_prompt": {
            "caption": {
                "base_caption": "ぼやけ、低品質",
                "char_captions": [{"char_caption": "帽子"}],
            },
            "vendor_negative_key": True,
        },
        "actual_prompts": {
            "prompt": {
                "caption": {
                    "base_caption": "actual: 雨の窓辺",
                    "char_captions": [{"char_caption": "人物"}],
                }
            },
            "negative_prompt": {
                "caption": {
                    "base_caption": "actual negative",
                    "char_captions": [{"char_caption": "帽子"}],
                }
            },
            "vendor_actual_key": ["preserve", 1],
        },
        "reference_image_multiple": ["", "AA=="],
        "reference_strength_multiple": [0.1, 0.4],
        "reference_information_extracted_multiple": [0.2, 0.8],
        "vendor_unknown": {"nested": ["preserved", 7]},
    }


def _png_info(parameters: dict, *, comment_format: str = "text") -> PngImagePlugin.PngInfo:
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "NovelAI")
    info.add_text("Source", "NovelAI Diffusion V5 0ADF9AB7")
    comment = json.dumps(parameters, ensure_ascii=False)
    if comment_format == "itxt":
        info.add_itxt("Comment", comment)
    elif comment_format == "ztxt":
        info.add_text("Comment", comment, zip=True)
    else:
        info.add_text("Comment", comment)
    return info


def _write_png(
    path: Path,
    *,
    info: PngImagePlugin.PngInfo | None = None,
    size: tuple[int, int] = (128, 128),
    mode: str = "RGB",
) -> None:
    image = Image.new(mode, size, (45, 73, 91) if mode == "RGB" else (45, 73, 91, 255))
    image.save(path, format="PNG", pnginfo=info)


def _stealth_packet(metadata: dict, *, compressed: bool) -> bytes:
    raw = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    if compressed:
        payload = gzip.compress(raw)
        magic = STEALTH_COMPRESSED_MAGIC
    else:
        payload = raw
        magic = STEALTH_PLAIN_MAGIC
    return magic + struct.pack(">I", len(payload) * 8) + payload


def _write_alpha_png(
    path: Path,
    packet: bytes,
    *,
    info: PngImagePlugin.PngInfo | None = None,
    size: tuple[int, int] = (512, 512),
) -> None:
    width, height = size
    if len(packet) * 8 > width * height:
        raise ValueError("test stealth packet does not fit")
    image = Image.new("RGBA", size, (40, 60, 80, 255))
    pixels = image.load()
    for bit_index in range(len(packet) * 8):
        x, y = divmod(bit_index, height)
        bit = (packet[bit_index // 8] >> (7 - bit_index % 8)) & 1
        red, green, blue, alpha = pixels[x, y]
        pixels[x, y] = (red, green, blue, (alpha & 0xFE) | bit)
    image.save(path, format="PNG", pnginfo=info)


@pytest.mark.parametrize("comment_format", ["text", "itxt"])
def test_png_text_metadata_maps_unicode_v4_and_actual_prompts(
    tmp_path: Path, comment_format: str
) -> None:
    path = tmp_path / "novelai.png"
    parameters = sample_parameters()
    _write_png(path, info=_png_info(parameters, comment_format=comment_format))
    before = hashlib.sha256(path.read_bytes()).digest()

    result = extract_metadata(path)
    compact = extract_metadata(path, include_raw=False)

    assert result["found"] is True
    assert result["primary_source"] == "png_text"
    assert result["sources"] == ["png_text"]
    assert result["image"] == {
        "format": "PNG",
        "width": 128,
        "height": 128,
        "mode": "RGB",
    }
    assert result["generation"]["model"] == "NovelAI Diffusion V5"
    assert result["generation"]["model"] != "nai-diffusion-5-full"
    assert result["generation"]["Source"] == "NovelAI Diffusion V5 0ADF9AB7"
    assert result["generation"]["prompt"] == "雨の窓辺，猫 🐈"
    assert result["generation"]["negative_prompt"] == "ぼやけ、低品質"
    assert result["generation"]["seed"] == 2026102102
    assert result["generation"]["steps"] == 28
    assert result["generation"]["scale"] == 6
    assert result["generation"]["cfg_rescale"] == 0.15
    assert result["generation"]["strength"] == 0.72
    assert result["generation"]["v4_prompt"] == parameters["v4_prompt"]
    assert result["generation"]["v4_negative_prompt"] == parameters["v4_negative_prompt"]
    assert result["generation"]["actual_prompts"] == parameters["actual_prompts"]
    assert result["generation"]["actual_prompt"]["prompt"] == "actual: 雨の窓辺"
    assert result["generation"]["characters"][0]["center"] == {"x": 0.0, "y": 0.75}
    assert result["generation"]["vibes"][0]["encodedData"] == "AA=="
    assert result["generation"]["vibes"][0]["strength"] == 0.4
    assert result["generation"]["vibes"][0]["informationExtracted"] == 0.8
    assert result["parameters"] == parameters
    assert result["raw"]["png_text"]["Comment"] == json.dumps(parameters, ensure_ascii=False)
    assert compact["parameters"] == parameters
    assert "raw" not in compact
    assert result["signature_verification"] == "not_performed"
    assert not any("signature" in warning.lower() for warning in result["warnings"])
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_png_ztxt_comment_is_bounded_and_parsed(tmp_path: Path) -> None:
    path = tmp_path / "compressed-text.png"
    parameters = sample_parameters()
    _write_png(path, info=_png_info(parameters, comment_format="ztxt"))

    result = extract_metadata(path)

    assert result["found"] is True
    assert result["primary_source"] == "png_text"
    assert result["parameters"] == parameters


@pytest.mark.parametrize("compressed", [False, True])
def test_alpha_stealth_plain_and_gzip_formats(tmp_path: Path, compressed: bool) -> None:
    path = tmp_path / f"stealth-{compressed}.png"
    parameters = sample_parameters()
    wrapper = {
        "Software": "NovelAI",
        "Source": "NovelAI Diffusion V5 0ADF9AB7",
        "Comment": json.dumps(parameters, ensure_ascii=False),
    }
    _write_alpha_png(path, _stealth_packet(wrapper, compressed=compressed))

    result = extract_metadata(path)

    assert result["found"] is True
    assert result["primary_source"] == "alpha_stealth"
    assert result["sources"] == ["alpha_stealth"]
    assert result["parameters"] == parameters
    assert result["generation"]["model"] == "NovelAI Diffusion V5"
    assert result["generation"]["prompt"] == "雨の窓辺，猫 🐈"


def test_corrupt_alpha_stealth_does_not_hide_valid_png_text(tmp_path: Path) -> None:
    path = tmp_path / "text-plus-bad-stealth.png"
    parameters = sample_parameters()
    info = _png_info(parameters)
    bad_payload = b"not a gzip stream"
    packet = STEALTH_COMPRESSED_MAGIC + struct.pack(">I", len(bad_payload) * 8) + bad_payload
    _write_alpha_png(path, packet, info=info)

    result = extract_metadata(path)

    assert result["found"] is True
    assert result["primary_source"] == "png_text"
    assert result["parameters"] == parameters
    assert "alpha_stealth" not in result["sources"]
    assert any("Compressed alpha-channel metadata" in item for item in result["warnings"])


def test_conflicting_channels_remain_separate_and_primary_is_not_merged(
    tmp_path: Path,
) -> None:
    path = tmp_path / "conflict.png"
    text_parameters = sample_parameters()
    stealth_parameters = sample_parameters()
    stealth_parameters["seed"] = 17
    info = _png_info(text_parameters)
    wrapper = {
        "Software": "NovelAI",
        "Comment": json.dumps(stealth_parameters, ensure_ascii=False),
    }
    _write_alpha_png(
        path,
        _stealth_packet(wrapper, compressed=True),
        info=info,
    )

    result = extract_metadata(path)

    assert result["primary_source"] == "png_text"
    assert result["parameters"] == text_parameters
    assert result["raw"]["png_text"]["Comment"] != result["raw"]["alpha_stealth"]["Comment"]
    assert any("sources disagree" in item for item in result["warnings"])


def test_oversized_alpha_length_is_rejected_before_payload_read(tmp_path: Path) -> None:
    path = tmp_path / "bad-length.png"
    packet = STEALTH_COMPRESSED_MAGIC + struct.pack(">I", 0xFFFFFFF8)
    _write_alpha_png(path, packet, size=(64, 64))

    result = extract_metadata(path)

    assert result["found"] is False
    assert any("length exceeds" in item for item in result["warnings"])


def test_gzip_expansion_limit_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "gzip-bomb.png"
    payload = gzip.compress(b"x" * (MAX_STEALTH_DECOMPRESSED_BYTES + 1))
    packet = STEALTH_COMPRESSED_MAGIC + struct.pack(">I", len(payload) * 8) + payload
    _write_alpha_png(path, packet, size=(512, 512))

    result = extract_metadata(path)

    assert result["found"] is False
    assert any("Compressed alpha-channel metadata" in item for item in result["warnings"])


@pytest.mark.parametrize(("image_format", "suffix"), [("JPEG", ".jpg"), ("WEBP", ".webp")])
def test_jpeg_and_webp_exif_use_official_tag_mapping(
    tmp_path: Path, image_format: str, suffix: str
) -> None:
    path = tmp_path / f"exif{suffix}"
    parameters = sample_parameters()
    exif = Image.Exif()
    exif[ExifTags.Base.DocumentName] = "Test title"
    exif[ExifTags.Base.ImageDescription] = "metadata description"
    exif[ExifTags.Base.Software] = "NovelAI"
    exif[ExifTags.Base.Artist] = "artist"
    exif[ExifTags.Base.Copyright] = "copyright"
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    exif_ifd[ExifTags.Base.UserComment] = b"ASCII\x00\x00\x00" + json.dumps(
        parameters, ensure_ascii=True
    ).encode("ascii")
    save_options = {"exif": exif.tobytes()}
    if image_format == "WEBP":
        save_options["lossless"] = True
    Image.new("RGB", (96, 64), (70, 80, 90)).save(path, format=image_format, **save_options)

    result = extract_metadata(path)

    assert result["found"] is True
    assert result["primary_source"] == "exif"
    assert result["sources"] == ["exif"]
    assert result["raw"]["exif"]["Title"] == "Test title"
    assert result["raw"]["exif"]["Description"] == "metadata description"
    assert result["raw"]["exif"]["Source"] == "NovelAI"
    assert result["parameters"] == parameters
    assert result["generation"]["model"] == "NovelAI Diffusion V5"
    assert result["generation"]["prompt"] == "雨の窓辺，猫 🐈"


def test_missing_metadata_stays_missing_and_sidecar_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "seed-2026102102.png"
    _write_png(path)
    (tmp_path / "request.json").write_text(
        json.dumps({"Comment": json.dumps(sample_parameters())}),
        encoding="utf-8",
    )

    result = extract_metadata(path)

    assert result["found"] is False
    assert result["primary_source"] is None
    assert result["sources"] == []
    assert result["parameters"] == {}
    assert "seed" not in result["generation"]
    assert result["warnings"] == []
    assert result["signature_verification"] == "not_performed"


def test_exif_comment_requires_official_ascii_prefix(tmp_path: Path) -> None:
    path = tmp_path / "bad-user-comment.jpg"
    exif = Image.Exif()
    exif[ExifTags.Base.Software] = "NovelAI"
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.UserComment] = b"UNDEFINED" + json.dumps(
        sample_parameters(), ensure_ascii=True
    ).encode("ascii")
    Image.new("RGB", (32, 32), (1, 2, 3)).save(path, format="JPEG", exif=exif.tobytes())

    result = extract_metadata(path)

    assert result["found"] is False
    assert result["primary_source"] is None
    assert result["generation"] == {}
