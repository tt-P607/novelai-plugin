from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_contracts import (
    SYNTHETIC_TOKEN,
    make_test_settings,
    paid_operation_request,
    subscription_state,
)

from novelai_private.client import NovelAIClient
from novelai_private.service import ImageService


@pytest.fixture
def synthetic_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOVELAI_API_TOKEN", SYNTHETIC_TOKEN)
    monkeypatch.delenv("NOVELAI_PRIVATE_CONFIG", raising=False)


async def run_quote(
    tmp_path: Path,
    *,
    request: dict[str, Any],
    subscription: dict[str, Any] | None = None,
    gateway_quote: dict[str, Any] | None = None,
    base_url: str = "https://image.novelai.net",
    status: int | None = None,
) -> tuple[dict[str, Any], list[httpx.Request], ImageService]:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/user/subscription"):
            if status is not None:
                return httpx.Response(status, json={"message": "synthetic subscription failure"})
            return httpx.Response(200, json=subscription or {})
        if request.url.path.endswith("/api/quote"):
            if status is not None:
                return httpx.Response(status, json={"message": "synthetic quote failure"})
            return httpx.Response(200, json={"quote": gateway_quote})
        raise AssertionError(
            f"Unexpected image endpoint reached: {request.method} {request.url.path}"
        )

    settings = replace(make_test_settings(tmp_path), base_url=base_url)
    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    result = await service.quote("generate", request)
    return result, requests, service


def assert_no_image_call(requests: list[httpx.Request]) -> None:
    assert all(not request.url.path.startswith("/ai/") for request in requests)
    for request in requests:
        assert request.headers.get("authorization") == f"Bearer {SYNTHETIC_TOKEN}"
        assert SYNTHETIC_TOKEN.encode() not in request.content
        assert all(
            SYNTHETIC_TOKEN not in value
            for name, value in request.headers.items()
            if name.lower() != "authorization"
        )


@pytest.mark.asyncio
async def test_private_gateway_stale_quote_uses_live_opus_entitlement(
    tmp_path: Path, synthetic_token: None
) -> None:
    request = paid_operation_request("generate", 1024, 1024)
    result, requests, service = await run_quote(
        tmp_path,
        request=request,
        subscription=subscription_state(),
        gateway_quote={
            "source": "private",
            "currency": "anlas",
            "providerAnlas": 30,
            "credits": None,
            "estimate": True,
            "billingNote": "stale database quote",
        },
        base_url="https://gateway.invalid/origin",
    )
    assert result["cost_status"] == "free"
    assert result["estimated_anlas"] == 0
    assert result["billing_source"] == "huishu-private-live"
    assert [request.url.path for request in requests] == ["/api/quote", "/origin/user/subscription"]
    assert_no_image_call(requests)
    assert not service.settings.output_dir.exists()


@pytest.mark.asyncio
async def test_platform_credits_remain_paid_when_provider_anlas_is_zero(
    tmp_path: Path, synthetic_token: None
) -> None:
    result, requests, service = await run_quote(
        tmp_path,
        request=paid_operation_request("generate", 1024, 1024),
        gateway_quote={
            "source": "platform",
            "currency": "credits",
            "credits": 60,
            "providerAnlas": 0,
        },
        base_url="https://gateway.invalid/origin",
    )
    assert result["cost_status"] == "paid"
    assert result["estimated_credits"] == 60
    assert result["estimated_anlas"] == 0
    assert [request.url.path for request in requests] == ["/api/quote"]
    assert_no_image_call(requests)
    assert not service.settings.output_dir.exists()


@pytest.mark.asyncio
async def test_free_platform_quote_still_requires_high_resolution_approval(
    tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/quote":
            return httpx.Response(
                200,
                json={"quote": {"source": "platform", "currency": "credits", "credits": 0}},
            )
        raise AssertionError(f"High-resolution request escaped approval: {request.url.path}")

    settings = replace(make_test_settings(tmp_path), base_url="https://gateway.invalid/origin")
    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    result = await service.execute("generate", paid_operation_request("generate", 1025, 1024))

    assert result["status"] == "high_resolution_approval_required"
    assert result["cost_status"] == "free"
    assert result["needs_high_resolution_approval"] is True
    assert [request.url.path for request in requests] == ["/api/quote"]
    assert not settings.output_dir.exists()


@pytest.mark.parametrize(
    "subscription",
    [
        subscription_state(include_usage=False),
        subscription_state(percent=0),
        subscription_state(percent=-1),
        subscription_state(is_negative=True),
        subscription_state(active=False),
    ],
    ids=["missing-v5-usage", "zero-quota", "negative-quota", "negative-flag", "inactive-opus"],
)
@pytest.mark.asyncio
async def test_v5_without_valid_opus_free_entitlement_is_never_treated_free(
    subscription: dict[str, Any], tmp_path: Path, synthetic_token: None
) -> None:
    result, requests, service = await run_quote(
        tmp_path,
        request={
            **paid_operation_request("generate", 1024, 1024),
            "model": "nai-diffusion-5-full",
        },
        subscription=subscription,
    )
    assert result["cost_status"] in {"paid", "unknown"}
    assert result["may_spend_credits"] is True
    gated = await service.execute(
        "generate",
        {**paid_operation_request("generate", 1024, 1024), "model": "nai-diffusion-5-full"},
    )
    assert gated["status"] == "approval_required"
    assert gated["cost_status"] in {"paid", "unknown"}
    assert_no_image_call(requests)
    assert not service.settings.output_dir.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda request: request["parameters"].update(steps=29),
        lambda request: request["parameters"].update(n_samples=2),
        lambda request: request["parameters"].update(
            director_reference_images=["synthetic-reference"]
        ),
        lambda request: request.update(use_new_shared_trial=False),
    ],
    ids=[
        "over-28-steps",
        "multiple-samples",
        "precise-reference-addon",
        "trial-explicitly-disabled",
    ],
)
@pytest.mark.asyncio
async def test_paid_generation_features_are_gated_even_with_active_opus(
    mutate, tmp_path: Path, synthetic_token: None
) -> None:
    requests: list[httpx.Request] = []
    settings = make_test_settings(tmp_path)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/user/subscription"):
            return httpx.Response(200, json=subscription_state())
        raise AssertionError(f"Unapproved request reached image endpoint: {request.url.path}")

    service = ImageService(
        settings, NovelAIClient(settings, transport=httpx.MockTransport(handler))
    )
    request = paid_operation_request("generate", 1024, 1024)
    mutate(request)

    result = await service.execute("generate", request)

    assert result["status"] == "approval_required"
    assert result["cost_status"] == "paid"
    assert result["estimated_anlas"] > 0
    assert_no_image_call(requests)
    assert not settings.output_dir.exists()


@pytest.mark.parametrize("case", ["subscription-failure", "unknown-model", "gateway-quote-failure"])
@pytest.mark.asyncio
async def test_missing_or_unknown_price_is_not_free(
    case: str, tmp_path: Path, synthetic_token: None
) -> None:
    request = paid_operation_request("generate", 1024, 1024)
    if case == "unknown-model":
        request["model"] = "future-unknown-model"
    result, requests, service = await run_quote(
        tmp_path,
        request=request,
        subscription=subscription_state(),
        gateway_quote={"source": "private", "currency": "anlas", "providerAnlas": 30},
        base_url="https://gateway.invalid/origin"
        if case == "gateway-quote-failure"
        else "https://image.novelai.net",
        status=503 if case in {"subscription-failure", "gateway-quote-failure"} else None,
    )
    assert result["cost_status"] == "unknown"
    assert result["may_spend_credits"] is True
    assert_no_image_call(requests)
    assert not service.settings.output_dir.exists()


@pytest.mark.asyncio
async def test_quote_calls_do_not_leak_credentials_or_create_output_records(
    tmp_path: Path, synthetic_token: None
) -> None:
    result, requests, service = await run_quote(
        tmp_path,
        request=paid_operation_request("generate", 1024, 1024),
        subscription=subscription_state(),
    )
    assert result["status"] == "quote"
    assert SYNTHETIC_TOKEN not in json.dumps(result)
    assert all(
        request.headers.get("authorization") == f"Bearer {SYNTHETIC_TOKEN}" for request in requests
    )
    assert_no_image_call(requests)
    assert not service.settings.output_dir.exists()
