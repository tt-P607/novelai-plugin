"""Apply Huishu's migrated Anlas rules with fresh account state and gateway fees."""

from __future__ import annotations

from urllib.parse import urlsplit

from .client import NovelAIClient
from .config import NovelAIError
from .pricing import build_request, calculate_anlas, opus_eligible

ROUTING_FIELDS = {
    "source",
    "sourceId",
    "source_id",
    "ownAccountId",
    "own_account_id",
    "account_source",
    "accountSource",
    "private_account_id",
    "privateAccountId",
}


def unknown(reason: str) -> dict:
    return {
        "cost_status": "unknown",
        "may_spend_credits": True,
        "estimated_credits": None,
        "estimated_anlas": None,
        "currency": None,
        "billing_reason": reason,
    }


def _amount(value: object) -> bool:
    return type(value) is int and value >= 0


async def quote_cost(client: NovelAIClient, operation: str, payload: dict) -> dict:
    """No billable endpoint is used here. Unknown prices require human consent."""
    try:
        request = build_request(operation, payload)
        gateway = await client.gateway_quote(request)
        if gateway is not None:
            if gateway.get("source") == "platform" and gateway.get("currency") == "credits":
                credits = gateway.get("credits")
                if not _amount(credits):
                    return unknown("绘枢没有返回有效的平台绘点报价。")
                return {
                    "cost_status": "free" if credits == 0 else "paid",
                    "may_spend_credits": credits != 0,
                    "estimated_credits": credits,
                    "estimated_anlas": gateway.get("providerAnlas"),
                    "currency": "credits",
                    "billing_source": "huishu-platform",
                    "billing_reason": "使用绘枢服务端报价；上游免费不免除平台绘点费用。",
                }
            if gateway.get("source") != "private" or gateway.get("currency") != "anlas":
                return unknown("绘枢未确认当前 Key 的账号来源及计费单位。")
            if ROUTING_FIELDS.intersection(payload):
                return unknown("请求指定了其他账号来源，不能用 Key 默认账号的订阅判断免费。")
            source = "huishu-private-live"
        else:
            if urlsplit(client.settings.base_url).hostname not in {
                "image.novelai.net",
                "api.novelai.net",
            }:
                return unknown("自定义服务未提供绘枢报价，无法确认是否存在额外收费。")
            source = "novelai-live"

        paid = calculate_anlas(request, False)
        free = calculate_anlas(request, True)
        eligible = False
        account: dict = {}
        free_flag = operation not in {"generate", "generate_stream"} or payload.get(
            "use_new_shared_trial", True
        )
        if free_flag and paid["providerAnlas"] != free["providerAnlas"]:
            subscription = await client.subscription()
            eligible = opus_eligible(subscription, request["model"])
            usage = subscription.get("usage")
            account = {
                "tier": subscription.get("tier"),
                "active": subscription.get("active"),
            }
            if isinstance(usage, dict):
                account["v5_usage_percent"] = usage.get("percent")
                account["v5_usage_negative"] = usage.get("isNegative")
            if eligible is None:
                return unknown("当前订阅或 V5 免费额度不完整，不能确认是否免费。")
        estimate = calculate_anlas(request, eligible)
        anlas = estimate["providerAnlas"]
        return {
            "cost_status": "free" if anlas == 0 else "paid",
            "may_spend_credits": anlas != 0,
            "estimated_credits": anlas,
            "estimated_anlas": anlas,
            "currency": "anlas",
            "billing_source": source,
            "billing_reason": "按绘枢原公式和当前实时订阅计算；免费时无需点数确认。",
            "components": estimate["components"],
            "account": account,
            "consumes_v5_allowance": bool(
                anlas == 0 and str(request["model"]).startswith("nai-diffusion-5-")
            ),
        }
    except NovelAIError:
        return unknown("无法读取当前订阅或绘枢报价；未发起生成，不能把未知当成免费或已收费。")
    except (ValueError, KeyError, TypeError):
        return unknown("计费参数缺失或当前模型/参数尚未纳入移植公式；未发起生成。")
