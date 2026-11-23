"""Metered, cached LLM access.

The supplied Azure key carries a hard ~$5 ceiling, so this app treats a model
call as a budgeted resource rather than something to sprinkle around:

  * TWO purposes only -- batch feedback routing, and the rep-facing narrative
    (the latter via D5's own `explain()`).
  * Every response is CACHED in SQLite on a hash of its exact inputs, so
    re-running the same demo costs nothing the second time.
  * A guard refuses to call once the configured call/spend ceiling is hit, and
    that ceiling sits BELOW the real $5 so the app stops itself before the
    provider does.
  * Every path has a deterministic fallback that is always built and always
    correct. The LLM improves the wording and the intent parsing; it is never
    load-bearing for a number.

Client construction is delegated to D5's own `_llm_client()` rather than
reimplemented. That function already encodes three fixes established against
these exact deployments -- `model=` must be the DEPLOYMENT name, it is
`max_completion_tokens` not `max_tokens`, and the GPT-5 family rejects a
non-default `temperature` -- and duplicating it here would mean maintaining two
copies of hard-won knowledge that can drift apart.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Tuple

from app import config
from app.db import store


@dataclass(frozen=True)
class LLMResult:
    text: str
    source: str          # "llm" | "cache" | "unavailable" | "refused" | "error"
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0

    @property
    def ok(self) -> bool:
        return self.source in {"llm", "cache"} and bool(self.text.strip())


def _cache_key(purpose: str, system: str, user: str, model_hint: str) -> str:
    blob = json.dumps([purpose, system, user, model_hint], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def tier_for(purpose: str) -> str:
    """Unlisted purposes default to the CHEAP tier, so a new feature has to opt
    in to the expensive one rather than silently inheriting it."""
    return config.LLM_PURPOSE_TIER.get(purpose, config.TIER_NANO)


def _estimate_cost(tier: str, prompt_tokens: int, completion_tokens: int) -> float:
    p_rate, c_rate = config.LLM_RATES.get(tier, config.LLM_RATES[config.TIER_MINI])
    return (prompt_tokens / 1_000_000.0) * p_rate + \
           (completion_tokens / 1_000_000.0) * c_rate


def engine_spend_usd() -> float:
    """What the ENGINE-side fence has recorded, at its own pessimistic rate.

    The app and the engines keep separate ledgers -- the app's calls go through
    this module into SQLite, the engines' go through `llm_agent.py`'s four-axis
    fence into its own persisted ledger. Both are guarded, but neither could
    answer the only question that matters operationally: what has this TEAM
    spent against one shared $5 key. Reading the engine ledger here is the
    cheap half of the answer. Read-only, and it never raises -- a missing or
    unreadable ledger must not take a screen down.
    """
    try:
        import llm_agent
        return float(llm_agent.usage().get("usd") or 0.0)
    except Exception:      # noqa: BLE001 -- absent, unreadable, changed shape
        return 0.0


def budget_status() -> dict:
    """What the sidebar meter renders."""
    summary = store.llm_spend_summary()
    spend = float(summary.get("spend_usd") or 0.0)
    billable = int(summary.get("n_billable") or 0)
    engine = engine_spend_usd()
    nano = config.deployment_for(config.TIER_NANO) or ""
    mini = config.deployment_for(config.TIER_MINI) or ""
    return {
        "configured": config.azure_configured(),
        "spend_usd": spend,
        "max_spend_usd": config.LLM_MAX_SPEND_USD,
        "billable_calls": billable,
        "max_calls": config.LLM_MAX_CALLS_PER_SESSION,
        "cached_hits": int(summary.get("n_cached") or 0),
        "prompt_tokens": int(summary.get("prompt_tokens") or 0),
        "completion_tokens": int(summary.get("completion_tokens") or 0),
        "spend_pct": (spend / config.LLM_MAX_SPEND_USD) if config.LLM_MAX_SPEND_USD else 0.0,
        "exhausted": spend >= config.LLM_MAX_SPEND_USD
                     or billable >= config.LLM_MAX_CALLS_PER_SESSION,
        # Deployment names are not secrets; the API KEY never leaves this module.
        "nano_deployment": nano,
        "mini_deployment": mini,
        "router_tier": tier_for("feedback_router"),
        "distinct_tiers": (nano != mini),
        # Engine-side spend, and the two added together. Both are PROTECTION
        # estimates at a deliberately pessimistic rate -- not an invoice, and
        # never to be quoted to a mentor as actual spend.
        "engine_spend_usd": engine,
        "total_spend_usd": spend + engine,
    }


def _guard_reason() -> Optional[str]:
    st = budget_status()
    if not st["configured"]:
        return ("no AZURE_OPENAI_API_KEY / _ENDPOINT / _DEPLOYMENT in the "
                "environment")
    if st["spend_usd"] >= config.LLM_MAX_SPEND_USD:
        return (f"app-side budget guard: estimated ${st['spend_usd']:.3f} spent "
                f"of the ${config.LLM_MAX_SPEND_USD:.2f} ceiling")
    if st["billable_calls"] >= config.LLM_MAX_CALLS_PER_SESSION:
        return (f"app-side call guard: {st['billable_calls']} billable calls "
                f"of {config.LLM_MAX_CALLS_PER_SESSION}")
    return None


def chat(*, purpose: str, system: str, user: str,
         max_completion_tokens: int = 600,
         use_cache: bool = True) -> LLMResult:
    """One metered completion. Never raises -- every failure returns a result
    whose `.source` says what happened, so callers fall back explicitly instead
    of catching exceptions."""
    try:
        from build_d5_orchestrator import _llm_client
    except Exception as exc:  # noqa: BLE001
        return LLMResult("", "unavailable", f"could not import D5's LLM client: {exc}")

    # D5's helper builds the client AND reports the deployment it would use
    # (mini). The client itself is deployment-agnostic -- `model=` is a per-call
    # argument -- so we keep its client and substitute the cheaper deployment
    # this purpose actually needs.
    conn = _llm_client()
    tier = tier_for(purpose)
    deployment = config.deployment_for(tier) or (conn[1] if conn else None)
    key = _cache_key(purpose, system, user, deployment or "none")

    if use_cache:
        cached = store.llm_cache_get(key)
        if cached is not None:
            store.log_llm_call(purpose=purpose, cache_key=key, cached=True,
                               deployment=deployment, prompt_tokens=0,
                               completion_tokens=0, est_cost_usd=0.0, ok=True)
            return LLMResult(cached, "cache", "served from the response cache")

    refuse = _guard_reason()
    if refuse:
        return LLMResult("", "refused", refuse)
    if conn is None:
        return LLMResult("", "unavailable",
                         "Azure client could not be constructed (see stderr)")

    client, _d5_deployment = conn
    try:
        resp = client.chat.completions.create(
            model=deployment,          # the tier-selected deployment, not D5's
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_completion_tokens=max_completion_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = _estimate_cost(tier, pt, ct)
        store.log_llm_call(purpose=purpose, cache_key=key, cached=False,
                           deployment=deployment, prompt_tokens=pt,
                           completion_tokens=ct, est_cost_usd=cost, ok=bool(text),
                           error=None if text else "empty completion")
        if not text:
            return LLMResult("", "error", "the model returned an empty completion",
                             pt, ct, cost)
        if use_cache:
            store.llm_cache_put(key, purpose, text)
        return LLMResult(text, "llm", "", pt, ct, cost)

    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        # Same diagnosis D5's explain() makes, so the app names the real cause
        # instead of showing a raw traceback.
        if "403" in msg:
            why = "403 from Azure -- off the VPN/office network, not a bad key"
        elif "401" in msg:
            why = "401 from Azure -- the key itself looks wrong"
        elif "429" in msg:
            why = "429 from Azure -- rate ceiling is shared across the team"
        elif "insufficient_quota" in msg or "quota" in msg.lower():
            why = "Azure reports the quota/credit is exhausted"
        else:
            why = f"call failed ({type(exc).__name__}: {exc})"
        store.log_llm_call(purpose=purpose, cache_key=key, cached=False,
                           deployment=deployment, prompt_tokens=None,
                           completion_tokens=None, est_cost_usd=0.0, ok=False,
                           error=why)
        return LLMResult("", "error", why)


def parse_json_object(text: str) -> Tuple[Optional[dict], str]:
    """Pull the first JSON object out of a completion.

    Models wrap JSON in prose or a ```json fence often enough that a bare
    json.loads is not a reliable contract. Returns (obj, error).
    """
    if not text:
        return None, "empty response"
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        return json.loads(cleaned), ""
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None, "no JSON object found in the response"
    depth = 0
    for i, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1]), ""
                except json.JSONDecodeError as exc:
                    return None, f"malformed JSON: {exc}"
    return None, "unbalanced JSON braces in the response"
