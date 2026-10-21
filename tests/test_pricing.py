from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from novelai_private.pricing import build_request, calculate_anlas, opus_eligible


def quote_request(
    *,
    model: str = "nai-diffusion-4-5-full",
    operation: str = "generate",
    width: int = 1024,
    height: int = 1024,
    steps: int = 28,
    quantity: int = 1,
    **values,
):
    return {
        "model": model,
        "operation": operation,
        "width": width,
        "height": height,
        "steps": steps,
        "quantity": quantity,
        **values,
    }


def image_base64(width: int, height: int) -> str:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def test_generation_base_and_v5_multiplier_match_huishu_golden_quotes():
    v45 = calculate_anlas(quote_request(), opus=False)
    v5 = calculate_anlas(quote_request(model="nai-diffusion-5-full"), opus=False)

    assert v45["providerAnlas"] == 20
    assert v5["providerAnlas"] == 30
    assert v5["components"][0]["kind"] == "generation"


@pytest.mark.parametrize(
    ("model", "expected_family_cost"),
    [
        ("nai-diffusion-5-full", 30),
        ("nai-diffusion-5-curated", 30),
        ("nai-diffusion-4-5-full", 20),
        ("nai-diffusion-4-5-curated", 20),
        ("nai-diffusion-4-full", 20),
        ("nai-diffusion-4-curated-preview", 20),
        ("nai-diffusion-3", 20),
        ("nai-diffusion-furry-3", 20),
        ("nai-diffusion-5-full-inpainting", 30),
        ("nai-diffusion-4-5-full-inpainting", 20),
    ],
)
def test_only_huishu_catalog_models_and_inpainting_aliases_are_accepted(
    model, expected_family_cost
):
    result = calculate_anlas(quote_request(model=model), opus=False)

    assert result["providerAnlas"] == expected_family_cost


def test_unknown_model_suffix_is_not_accepted_as_known_family():
    with pytest.raises(ValueError):
        calculate_anlas(quote_request(model="nai-diffusion-5-future-model"), opus=False)


def test_exact_rounding_applies_base_ceiling_then_multipliers():
    small_v45 = calculate_anlas(quote_request(width=512, height=512), opus=False)
    small_v5 = calculate_anlas(
        quote_request(model="nai-diffusion-5-full", width=512, height=512),
        opus=False,
    )

    # The V4.5 base rounds up to 5 first; V5 then computes ceil(5 * 3/2).
    assert small_v45["providerAnlas"] == 5
    assert small_v5["providerAnlas"] == 8


def test_opus_free_generation_and_reference_surcharges_match_golden_quote():
    free = calculate_anlas(quote_request(), opus=True)
    batch = calculate_anlas(
        quote_request(model="nai-diffusion-5-full", quantity=2),
        opus=True,
    )
    referenced = calculate_anlas(
        quote_request(
            operation="img2img",
            steps=50,
            strengthBps=5000,
            hasStrength=True,
            characterReferenceCount=1,
            vibeCount=6,
            uncachedVibeCount=1,
        ),
        opus=False,
    )
    free_with_vibes = calculate_anlas(
        quote_request(vibeCount=5, uncachedVibeCount=1),
        opus=True,
    )

    assert free["providerAnlas"] == 0
    assert [part["kind"] for part in free["components"]] == ["generation", "opus-free"]
    assert batch["providerAnlas"] == 60
    assert referenced["providerAnlas"] == 28
    assert free_with_vibes["providerAnlas"] == 4
    assert [part["kind"] for part in free_with_vibes["components"]] == [
        "generation",
        "opus-free",
        "vibe-encode",
        "vibe-overage",
    ]


def test_character_reference_disqualifies_free_image_and_is_per_sample():
    one = calculate_anlas(quote_request(characterReferenceCount=1), opus=True)
    batch = calculate_anlas(
        quote_request(quantity=2, characterReferenceCount=1),
        opus=True,
    )

    assert one["providerAnlas"] == 25
    assert batch["providerAnlas"] == 50
    assert all(part["kind"] != "opus-free" for part in one["components"])


def test_smea_and_dynamic_multipliers_use_exact_integer_ceiling():
    v3_smea = calculate_anlas(
        quote_request(model="nai-diffusion-3", smea=True),
        opus=False,
    )
    v3_dynamic = calculate_anlas(
        quote_request(model="nai-diffusion-3", smea=True, smDynamic=True),
        opus=False,
    )
    dynamic_without_smea = calculate_anlas(
        quote_request(model="nai-diffusion-3", smDynamic=True),
        opus=False,
    )

    assert v3_smea["providerAnlas"] == 24
    assert v3_dynamic["providerAnlas"] == 28
    assert dynamic_without_smea["providerAnlas"] == 20


def test_stream_and_native_payload_fields_map_without_mutation():
    payload = {
        "request": {
            "input": "keep this provider prompt",
            "model": "nai-diffusion-4-5-full",
            "action": "infill",
            "use_new_shared_trial": True,
            "parameters": {
                "width": 1024,
                "height": 1024,
                "steps": 50,
                "n_samples": 2,
                "inpaintImg2ImgStrength": 0.12345,
                "img2img": {"strength": 0.5},
                "autoSmea": True,
                "sm_dyn": True,
                "reference_image_multiple": ["one", "two"],
                "director_reference_images": ["character"],
            },
            "source": "own",
            "ownAccountId": "account-1",
        }
    }
    original = {
        "request": {
            **payload["request"],
            "parameters": dict(payload["request"]["parameters"]),
        }
    }

    mapped = build_request("generate_stream", payload)

    assert mapped["operation"] == "inpaint"
    assert mapped["strengthBps"] == 1235
    assert mapped["hasStrength"] is True
    assert mapped["quantity"] == 2
    assert mapped["smea"] is True
    assert mapped["smDynamic"] is True
    assert mapped["vibeCount"] == 2
    assert mapped["characterReferenceCount"] == 1
    assert mapped["uncachedVibeCount"] == 0
    assert mapped["source"] == "own"
    assert mapped["ownAccountId"] == "account-1"
    assert "opus" not in mapped
    assert payload == original


def test_inpaint_strength_falls_back_to_nested_then_top_level_strength():
    nested = build_request(
        "generate",
        {
            "model": "nai-diffusion-4-5-full-inpainting",
            "action": "infill",
            "parameters": {
                "width": 1024,
                "height": 1024,
                "img2img": {"strength": 0.5},
            },
        },
    )
    top_level = build_request(
        "generate",
        {
            "model": "nai-diffusion-4-5-full",
            "action": "infill",
            "parameters": {"width": 1024, "height": 1024, "strength": 0.25},
        },
    )

    assert nested["strengthBps"] == 5000
    assert nested["model"] == "nai-diffusion-4-5-full"
    assert top_level["strengthBps"] == 2500


def test_upscale_tiers_use_input_image_area():
    cases = [
        (1024, 1024, 1),
        (1024, 1088, 2),
        (1536, 1280, 3),
        (1536, 2048, 4),
    ]
    for width, height, expected in cases:
        result = calculate_anlas(
            quote_request(operation="upscale", width=width, height=height),
            opus=False,
        )
        assert result["providerAnlas"] == expected


def test_tool_mapping_reads_source_image_dimensions_and_uses_quote_anchor():
    payload = {
        "request": {
            "image": image_base64(1024, 1088),
            "model": "SwinIR_4x",
        }
    }

    mapped = build_request("upscale", payload)

    assert mapped["operation"] == "upscale"
    assert (mapped["width"], mapped["height"]) == (1024, 1088)
    assert mapped["model"] == "nai-diffusion-4-5-full"
    assert calculate_anlas(mapped, opus=False)["providerAnlas"] == 2


def test_tool_mapping_does_not_read_file_binding_paths():
    mapped = build_request(
        "upscale",
        {
            "request": {"width": 512, "height": 512, "model": "SwinIR_4x"},
            "files": [{"pointer": "/image", "path": "C:\\not-read\\image.png"}],
        },
    )

    assert (mapped["width"], mapped["height"]) == (512, 512)


def test_director_formula_and_opus_free_exception_match_golden_quotes():
    lineart = quote_request(operation="director", directorTool="director-lineart")
    background = quote_request(operation="director", directorTool="bg-remover")

    assert calculate_anlas(lineart, opus=False)["providerAnlas"] == 20
    assert calculate_anlas(lineart, opus=True)["providerAnlas"] == 0
    assert calculate_anlas(background, opus=True)["providerAnlas"] == 65

    mapped = build_request(
        "augment",
        {
            "request": {
                "image": image_base64(1024, 1024),
                "req_type": "bg-remover",
                "model": "nai-diffusion-4-5-full",
            }
        },
    )
    assert mapped["directorTool"] == "bg-removal"
    assert calculate_anlas(mapped, opus=True)["providerAnlas"] == 65


def test_encode_vibe_is_fixed():
    result = calculate_anlas(
        quote_request(operation="encode-vibe", model="nai-diffusion-5-full"),
        opus=False,
    )

    assert result["providerAnlas"] == 2
    assert result["components"][0]["kind"] == "encode-vibe"


def test_opus_eligibility_uses_active_tier_and_v5_usage():
    opus_v4 = {"tier": 3, "active": True}
    opus_v5 = {
        "tier": 3,
        "active": True,
        "usage": {"percent": 100, "isNegative": False},
    }

    assert opus_eligible(opus_v4, "nai-diffusion-4-5-full") is True
    assert opus_eligible(opus_v4, "nai-diffusion-5-full") is None
    assert opus_eligible(opus_v5, "nai-diffusion-5-full") is True
    assert (
        opus_eligible(
            {"tier": 3, "active": True, "usage": {"percent": 0, "isNegative": False}},
            "nai-diffusion-5-full",
        )
        is False
    )
    assert (
        opus_eligible(
            {"tier": 3, "active": True, "usage": {"percent": 100, "isNegative": True}},
            "nai-diffusion-5-full",
        )
        is False
    )
    assert (
        opus_eligible(
            {"tier": 2, "active": True, "usage": {"percent": 100}},
            "nai-diffusion-4-5-full",
        )
        is False
    )
    assert opus_eligible({"tier": 3, "active": False}, "nai-diffusion-4-5-full") is False
    assert opus_eligible({"tier": 3}, "nai-diffusion-4-5-full") is None


@pytest.mark.parametrize(
    "quote_request",
    [
        {"operation": "generate", "model": "custom-model", "width": 1024, "height": 1024},
        {"operation": "mystery", "model": "nai-diffusion-4-5-full"},
        {"operation": "generate", "model": "nai-diffusion-4-5-full", "width": 0, "height": 1024},
    ],
)
def test_insufficient_or_unsupported_quote_data_is_unknown(quote_request):
    with pytest.raises(ValueError):
        calculate_anlas(quote_request, opus=False)


def test_v4_curated_inpainting_alias_matches_huishu_canonical_model():
    request = build_request(
        "generate",
        {
            "model": "nai-diffusion-4-curated-inpainting",
            "action": "infill",
            "parameters": {"width": 832, "height": 1216, "steps": 28, "n_samples": 1},
        },
    )
    assert request["model"] == "nai-diffusion-4-curated-preview"
    assert calculate_anlas(request, opus=True)["providerAnlas"] == 0
