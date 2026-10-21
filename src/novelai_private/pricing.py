# SPDX-License-Identifier: AGPL-3.0-or-later
# Adapted Huishu implementation; see NOTICE for source attribution.
"""Local mirror of NovelAI Anlas quote calculations used by Huishu."""

from __future__ import annotations

import base64
import binascii
import io
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from PIL import Image

COEFFICIENT_SCALE = 1_000_000_000_000_000
AREA_COEFFICIENT = 2_951_823_175
AREA_STEPS_COEFFICIENT = 575_329_823
FREE_AREA_LIMIT = 1_048_576
FREE_STEPS_LIMIT = 28
MIN_ANLAS_PER_IMAGE = 2
MAX_ANLAS_PER_IMAGE = 140
UPSCALE_ANLAS = 1
VIBE_ENCODE_ANLAS = 2
CHARACTER_REFERENCE_ANLAS = 5
FREE_VIBE_COUNT = 4
EXTRA_VIBE_ANLAS = 2
DIRECTOR_STEPS = 28
DIRECTOR_MIN_AREA = 1_048_576
DIRECTOR_MAX_AREA = 3_145_728
BACKGROUND_REMOVAL_MULTIPLIER = 3
BACKGROUND_REMOVAL_FLAT_ANLAS = 5
MODEL_ANCHOR = "nai-diffusion-4-5-full"

_SOURCE_FIELDS = (
    "source",
    "sourceId",
    "source_id",
    "ownAccountId",
    "own_account_id",
    "account_source",
    "accountSource",
    "private_account_id",
    "privateAccountId",
)
_GENERATION_OPERATIONS = {"generate", "img2img", "inpaint"}
_OPERATION_ALIASES = {
    "generate": "generate",
    "generate-image": "generate",
    "generate-stream": "generate",
    "generate-image-stream": "generate",
    "stream": "generate",
    "text-to-image": "generate",
    "txt2img": "generate",
    "img2img": "img2img",
    "image-to-image": "img2img",
    "inpaint": "inpaint",
    "infill": "inpaint",
    "upscale": "upscale",
    "encode-vibe": "encode-vibe",
    "augment": "director",
    "augment-image": "director",
    "director": "director",
}
_DIRECTOR_TOOLS = {
    "declutter",
    "declutter-keep-bubbles",
    "bg-removal",
    "lineart",
    "sketch",
    "colorize",
    "emotion",
}
_MODEL_FAMILIES = {
    "nai-diffusion-5-full": "v5",
    "nai-diffusion-5-curated": "v5",
    "nai-diffusion-4-5-full": "v4.5",
    "nai-diffusion-4-5-curated": "v4.5",
    "nai-diffusion-4-full": "v4",
    "nai-diffusion-4-curated-preview": "v4",
    "nai-diffusion-3": "v3",
    "nai-diffusion-furry-3": "v3",
}
_INPAINTING_SUFFIX = "-inpainting"


def build_request(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Map a raw NovelAI payload to Huishu /api/quote fields without mutating it."""
    outer = _mapping(payload, "payload")
    native_value = outer.get("request")
    native = _mapping(native_value, "request") if native_value is not None else outer
    route_operation = _normalize_operation(operation)

    if route_operation in _GENERATION_OPERATIONS:
        action = native.get("action")
        quote_operation = _normalize_operation(action) if action is not None else route_operation
        if quote_operation not in _GENERATION_OPERATIONS:
            raise ValueError("unsupported generation action")
        parameters = _mapping(native.get("parameters"), "parameters")
        model = _canonical_model(_required_model(native.get("model", outer.get("model"))))
        _model_family(model)
        width = _required_int(parameters, "width")
        height = _required_int(parameters, "height")
        steps = _default_int(parameters.get("steps"), 28, "steps")
        quantity = _default_int(parameters.get("n_samples"), 1, "n_samples")
        strength_bps, has_strength = _native_strength(parameters, quote_operation)
        smea = _optional_bool(parameters, "sm") or _optional_bool(parameters, "autoSmea")
        sm_dynamic = _optional_bool(parameters, "sm_dyn")
        result: dict[str, Any] = {
            "model": model,
            "operation": quote_operation,
            "width": width,
            "height": height,
            "steps": steps,
            "quantity": quantity,
            "strengthBps": strength_bps,
            "hasStrength": has_strength,
            "quality": _optional_string(parameters, "quality", default="standard"),
            "smea": smea,
            "smDynamic": sm_dynamic,
            "vibeCount": _array_length(parameters, "reference_image_multiple"),
            # Native generation references are encoded already. New encoding is
            # billed by the separate encode-vibe operation.
            "uncachedVibeCount": 0,
            "characterReferenceCount": _array_length(parameters, "director_reference_images"),
        }
    elif route_operation in {"upscale", "encode-vibe", "director"}:
        request = _mapping(native.get("request"), "request") if "request" in native else native
        if route_operation in {"upscale", "director"}:
            width, height = _tool_image_dimensions(native, outer)
        else:
            width, height = 0, 0
        result = {
            "model": _tool_quote_model(request.get("model", outer.get("model"))),
            "operation": route_operation,
            "width": width,
            "height": height,
            "steps": 0,
            "quantity": 1,
            "strengthBps": 10_000,
            "hasStrength": False,
            "quality": "standard",
            "smea": False,
            "smDynamic": False,
            "vibeCount": 0,
            "uncachedVibeCount": 0,
            "characterReferenceCount": 0,
        }
        if route_operation == "director":
            tool = _normalize_director_tool(
                request.get("req_type", request.get("directorTool", ""))
            )
            if tool not in _DIRECTOR_TOOLS:
                raise ValueError("unsupported Director tool")
            result["directorTool"] = tool
    else:
        raise ValueError(f"unsupported operation: {operation!r}")

    for field in _SOURCE_FIELDS:
        values = []
        for source in (outer, native):
            if field in source:
                value = source[field]
                if not isinstance(value, str):
                    raise ValueError(f"{field} must be a string")
                values.append(value)
        if values:
            if len(set(values)) > 1:
                raise ValueError(f"conflicting {field} values")
            result[field] = values[0]
    return result


def calculate_anlas(quote_request: dict[str, Any], opus: bool) -> dict[str, Any]:
    """Return provider-side Anlas and the component breakdown."""
    request = _mapping(quote_request, "quote_request")
    if not isinstance(opus, bool):
        raise ValueError("opus must be a boolean")

    operation = _normalize_operation(request.get("operation"))
    if operation in _GENERATION_OPERATIONS:
        family = _model_family(_required_model(request.get("model")))
        width = _positive_int(request.get("width"), "width")
        height = _positive_int(request.get("height"), "height")
        steps = _default_int(request.get("steps"), 28, "steps")
        quantity = _default_int(request.get("quantity"), 1, "quantity")
        if steps < 1 or quantity < 1:
            raise ValueError("steps and quantity must be positive")
        strength_bps = _quote_strength(request)
        smea = _quote_bool(request, "smea")
        sm_dynamic = _quote_bool(request, "smDynamic")
        vibe_count = _nonnegative_int(request.get("vibeCount", 0), "vibeCount")
        uncached_vibe_count = _nonnegative_int(
            request.get("uncachedVibeCount", 0), "uncachedVibeCount"
        )
        character_count = _nonnegative_int(
            request.get("characterReferenceCount", 0), "characterReferenceCount"
        )

        per_image = _generation_per_image(
            width,
            height,
            steps,
            family,
            strength_bps,
            operation in {"img2img", "inpaint"},
            smea,
            sm_dynamic,
        )
        free = (
            opus
            and width * height <= FREE_AREA_LIMIT
            and steps <= FREE_STEPS_LIMIT
            and quantity == 1
            and character_count == 0
        )
        billable_quantity = quantity - 1 if free else quantity
        components = [
            {
                "kind": "generation",
                "providerAnlas": per_image * billable_quantity,
                "description": f"{billable_quantity} billable image(s) at {per_image} Anlas",
            }
        ]
        if free:
            components.append(
                {
                    "kind": "opus-free",
                    "providerAnlas": 0,
                    "description": "one qualifying image is free",
                }
            )
        character_cost = character_count * CHARACTER_REFERENCE_ANLAS * quantity
        if character_cost:
            components.append(
                {
                    "kind": "character-reference",
                    "providerAnlas": character_cost,
                    "description": "character reference surcharge",
                }
            )
        vibe_encode_cost = uncached_vibe_count * VIBE_ENCODE_ANLAS
        if vibe_encode_cost:
            components.append(
                {
                    "kind": "vibe-encode",
                    "providerAnlas": vibe_encode_cost,
                    "description": "uncached vibe encoding",
                }
            )
        vibe_overage_cost = max(vibe_count - FREE_VIBE_COUNT, 0) * EXTRA_VIBE_ANLAS
        if vibe_overage_cost:
            components.append(
                {
                    "kind": "vibe-overage",
                    "providerAnlas": vibe_overage_cost,
                    "description": "vibe references beyond the free allowance",
                }
            )
        return _quote_result(components)

    if operation == "upscale":
        area = _positive_int(request.get("width"), "width") * _positive_int(
            request.get("height"), "height"
        )
        cost = UPSCALE_ANLAS
        if area > 1_048_576:
            cost = 2
        if area > 1_747_627:
            cost = 3
        if area > 2_446_678:
            cost = 4
        return _quote_result(
            [
                {
                    "kind": "upscale",
                    "providerAnlas": cost,
                    "description": "fixed 2x upscale",
                }
            ]
        )

    if operation == "encode-vibe":
        return _quote_result(
            [
                {
                    "kind": "encode-vibe",
                    "providerAnlas": VIBE_ENCODE_ANLAS,
                    "description": "vibe encoding",
                }
            ]
        )

    if operation == "director":
        width = _positive_int(request.get("width"), "width")
        height = _positive_int(request.get("height"), "height")
        area = width * height
        tool = _normalize_director_tool(request.get("directorTool", ""))
        if tool not in _DIRECTOR_TOOLS:
            raise ValueError("unsupported Director tool")
        base = _director_base(area)
        if tool == "bg-removal":
            cost = base * BACKGROUND_REMOVAL_MULTIPLIER + BACKGROUND_REMOVAL_FLAT_ANLAS
            return _quote_result([{"kind": "director", "providerAnlas": cost, "description": tool}])
        if opus and area <= DIRECTOR_MIN_AREA:
            return _quote_result(
                [
                    {
                        "kind": "director-free",
                        "providerAnlas": 0,
                        "description": tool,
                    }
                ]
            )
        return _quote_result([{"kind": "director", "providerAnlas": base, "description": tool}])

    raise ValueError(f"unsupported operation: {operation!r}")


def opus_eligible(subscription: dict[str, Any], model: str) -> bool | None:
    """Return None when the subscription or V5 quota is not sufficiently known."""
    if not isinstance(subscription, Mapping):
        return None
    nested = subscription.get("subscription")
    if isinstance(nested, Mapping):
        subscription = nested
    try:
        family = _model_family(model)
    except ValueError:
        return None

    active = subscription.get("active")
    tier = _tier_name(subscription.get("tier"))
    if active is False or (tier is not None and tier not in {"3", "opus"}):
        return False
    if active is not True or tier is None:
        return None
    if family != "v5":
        return True

    usage = subscription.get("usage")
    if not isinstance(usage, Mapping):
        return None
    percent = usage.get("percent")
    if isinstance(percent, bool) or not isinstance(percent, (int, float, Decimal)):
        return None
    try:
        quota = Decimal(str(percent))
    except InvalidOperation:
        return None
    if not quota.is_finite():
        return None
    negative = usage.get("isNegative", False)
    if not isinstance(negative, bool):
        return None
    return not negative and quota > 0


def _quote_result(components: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "providerAnlas": sum(component["providerAnlas"] for component in components),
        "components": components,
    }


def _generation_per_image(
    width: int,
    height: int,
    steps: int,
    family: str,
    strength_bps: int,
    apply_strength: bool,
    smea: bool,
    sm_dynamic: bool,
) -> int:
    area = width * height
    base = _ceil_div(
        area * (AREA_COEFFICIENT + steps * AREA_STEPS_COEFFICIENT),
        COEFFICIENT_SCALE,
    )
    multiplier_numerator = 1
    multiplier_denominator = 1
    if smea:
        multiplier_numerator *= 7 if sm_dynamic else 6
        multiplier_denominator *= 5
    if family == "v5":
        multiplier_numerator *= 3
        multiplier_denominator *= 2
    if apply_strength:
        multiplier_numerator *= strength_bps
        multiplier_denominator *= 10_000
    cost = _ceil_div(base * multiplier_numerator, multiplier_denominator)
    return max(MIN_ANLAS_PER_IMAGE, min(MAX_ANLAS_PER_IMAGE, cost))


def _director_base(area: int) -> int:
    normalized_area = max(DIRECTOR_MIN_AREA, min(DIRECTOR_MAX_AREA, area))
    base = _ceil_div(
        normalized_area * (AREA_COEFFICIENT + DIRECTOR_STEPS * AREA_STEPS_COEFFICIENT),
        COEFFICIENT_SCALE,
    )
    return max(MIN_ANLAS_PER_IMAGE, min(MAX_ANLAS_PER_IMAGE, base))


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("invalid ceil division")
    return (numerator + denominator - 1) // denominator


def _normalize_operation(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("operation must be a string")
    name = value.strip().lower().replace("_", "-")
    if name.startswith("novelai-"):
        name = name[len("novelai-") :]
    normalized = _OPERATION_ALIASES.get(name)
    if normalized is None:
        raise ValueError(f"unsupported operation: {value!r}")
    return normalized


def _model_family(model: Any) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    name = _canonical_model(model)
    family = _MODEL_FAMILIES.get(name)
    if family is None:
        raise ValueError("unknown model family")
    return family


def _canonical_model(model: str) -> str:
    name = model.strip()
    if name == "nai-diffusion-4-curated-inpainting":
        return "nai-diffusion-4-curated-preview"
    if name.endswith(_INPAINTING_SUFFIX):
        name = name[: -len(_INPAINTING_SUFFIX)]
    if name not in _MODEL_FAMILIES:
        raise ValueError("unknown model family")
    return name


def _required_model(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model is required")
    return value.strip()


def _tool_quote_model(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        try:
            return _canonical_model(value)
        except ValueError:
            pass
    # Tool prices do not depend on model; use the native Director quote anchor.
    return MODEL_ANCHOR


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _required_int(values: Mapping[str, Any], key: str) -> int:
    return _nonnegative_int(values.get(key), key)


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _default_int(value: Any, default: int, field: str) -> int:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool) and value == 0:
        return default
    return _nonnegative_int(value, field)


def _optional_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _quote_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_string(values: Mapping[str, Any], key: str, *, default: str) -> str:
    value = values.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _array_length(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key, [])
    if value is None:
        return 0
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be an array")
    return len(value)


def _native_strength(parameters: Mapping[str, Any], operation: str) -> tuple[int, bool]:
    if operation == "generate":
        return 0, False
    candidates = ["strength"]
    if operation == "inpaint":
        candidates = ["inpaintImg2ImgStrength", "img2img.strength", "strength"]
    for key in candidates:
        if key == "img2img.strength":
            nested = parameters.get("img2img")
            if nested is None:
                continue
            nested = _mapping(nested, "parameters.img2img")
            if "strength" not in nested:
                continue
            value = nested["strength"]
        else:
            if key not in parameters:
                continue
            value = parameters[key]
        return _strength_basis_points(value), True
    return 0, False


def _strength_basis_points(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("strength must be a number between 0 and 1")
    try:
        strength = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("strength must be finite") from exc
    if not strength.is_finite() or strength < 0 or strength > 1:
        raise ValueError("strength must be finite and between 0 and 1")
    return int((strength * 10_000).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _quote_strength(request: Mapping[str, Any]) -> int:
    has_strength = request.get("hasStrength", False)
    if not isinstance(has_strength, bool):
        raise ValueError("hasStrength must be a boolean")
    if not has_strength:
        return 10_000
    value = _nonnegative_int(request.get("strengthBps"), "strengthBps")
    if value > 10_000:
        raise ValueError("strengthBps must be at most 10000")
    return value


def _normalize_director_tool(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("directorTool must be a string")
    tool = value.strip().lower()
    if tool.startswith("director-"):
        tool = tool[len("director-") :]
    return "bg-removal" if tool == "bg-remover" else tool


def _tool_image_dimensions(native: Mapping[str, Any], outer: Mapping[str, Any]) -> tuple[int, int]:
    image = native.get("image")
    if image is None:
        parameters = native.get("parameters")
        if isinstance(parameters, Mapping):
            image = parameters.get("image")
    if image is not None:
        try:
            with Image.open(io.BytesIO(_image_bytes(image))) as opened:
                width, height = opened.size
            return _positive_int(width, "image width"), _positive_int(height, "image height")
        except (OSError, ValueError, binascii.Error) as exc:
            raise ValueError("unable to read tool input image dimensions") from exc
    width = native.get("width", outer.get("width"))
    height = native.get("height", outer.get("height"))
    if width is None or height is None:
        raise ValueError("tool image dimensions are required")
    return _positive_int(width, "width"), _positive_int(height, "height")


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if not isinstance(value, str) or not value:
        raise ValueError("image must be base64 data")
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    encoded = value.strip()
    encoded += "=" * ((4 - len(encoded) % 4) % 4)
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image must be base64 data") from exc


def _tier_name(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return str(value).strip().lower() or None
