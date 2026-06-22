"""
LLM wrapper — DeepSeek API for all LLM calls in the CRAG pipeline.

DeepSeek is OpenAI-compatible. The base URL is https://api.deepseek.com.

Available models (verified working):
  - deepseek-v4-flash      (fast, default — recommended)
  - deepseek-v4-pro        (better quality, slower)
  - deepseek-chat          (legacy, deprecated 2026/07/24)
  - deepseek-reasoner      (legacy reasoning model, deprecated 2026/07/24)

Why DeepSeek:
  - No aggressive rate limiting (unlike free OpenRouter tier)
  - Native OpenAI-compatible API (drop-in replacement)
  - Multiple model tiers (flash for speed, pro for quality)
  - Supports reasoning via dedicated 'deepseek-reasoner' model

Default model: deepseek-v4-flash (fast, reliable for the 5+ LLM calls per CRAG query).
Switch via DEEPSEEK_MODEL in .env.
"""
import os
import json
import re
import time
from typing import Optional, Dict, Any
import requests
from dotenv import load_dotenv

load_dotenv()

_DEEPSEEK_API_KEY = None
_DEEPSEEK_MODEL = None
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
# Reasoning models (deepseek-reasoner) can take 30-90s per call.
_DEFAULT_TIMEOUT = 180  # seconds


def _get_api_key() -> str:
    """Lazily read the DeepSeek API key from env."""
    global _DEEPSEEK_API_KEY
    if _DEEPSEEK_API_KEY is None:
        _DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    if not _DEEPSEEK_API_KEY or _DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. "
            "Get a key from https://platform.deepseek.com and add it to .env"
        )
    return _DEEPSEEK_API_KEY


def _get_model() -> str:
    """Lazily read the DeepSeek model name from env."""
    global _DEEPSEEK_MODEL
    if _DEEPSEEK_MODEL is None:
        _DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    return _DEEPSEEK_MODEL


def _get_fallback_model() -> Optional[str]:
    """Return a fallback model to use when the primary fails.

    Default: deepseek-chat (legacy but reliable, always available).
    """
    return os.getenv("DEEPSEEK_FALLBACK_MODEL", "deepseek-chat")


def _use_reasoning() -> bool:
    """Whether to route reasoning-capable queries through deepseek-reasoner.

    When True, the wrapper uses 'deepseek-reasoner' for generation calls
    (where reasoning helps) and the configured DEEPSEEK_MODEL for fast
    grading calls (evaluator, hallucination check).

    Default: False (use the configured model for everything).
    """
    return os.getenv("DEEPSEEK_USE_REASONING", "false").lower() == "true"


def _call_deepseek(
    payload: Dict[str, Any],
    headers: Dict[str, str],
    primary_model: str,
    fallback_model: Optional[str],
) -> Dict[str, Any]:
    """Make a single DeepSeek call with retry + automatic model fallback.

    Tries the primary model first. If it errors after retries, switches to
    the fallback model.
    """
    models_to_try = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models_to_try.append(fallback_model)

    last_err = None
    for model_idx, model in enumerate(models_to_try):
        is_fallback = model_idx > 0
        if is_fallback:
            print(f"[llm_invoke] Primary model '{primary_model}' failed. "
                  f"Falling back to '{model}'.")

        call_payload = dict(payload)
        call_payload["model"] = model
        # deepseek-reasoner doesn't support temperature/system prompt the same way
        if model == "deepseek-reasoner":
            call_payload.pop("temperature", None)

        max_retries = 2 if is_fallback else 3
        backoff = 3
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    url=_DEEPSEEK_BASE_URL,
                    headers=headers,
                    data=json.dumps(call_payload),
                    timeout=_DEFAULT_TIMEOUT,
                )
                if response.status_code == 429:
                    last_err = f"HTTP 429 from {model}"
                    if attempt < max_retries:
                        print(f"[llm_invoke] {model}: HTTP 429 "
                              f"(attempt {attempt+1}/{max_retries+1}), "
                              f"retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    break  # try next model
                if response.status_code >= 500:
                    last_err = f"HTTP {response.status_code} from {model}"
                    if attempt < max_retries:
                        print(f"[llm_invoke] {model}: HTTP {response.status_code} "
                              f"(attempt {attempt+1}/{max_retries+1}), "
                              f"retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    break
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                last_err = f"HTTP {e.response.status_code} from {model}"
                if e.response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    print(f"[llm_invoke] {model}: HTTP {e.response.status_code} "
                          f"(attempt {attempt+1}/{max_retries+1}), "
                          f"retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[llm_invoke] {model}: HTTP {e.response.status_code}, "
                      f"trying next model...")
                break
            except requests.exceptions.Timeout:
                last_err = f"Timeout from {model}"
                break

    raise RuntimeError(f"All models failed. Last error: {last_err}")


def llm_invoke(
    prompt: str,
    system: Optional[str] = None,
    use_reasoning: Optional[bool] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """Invoke the LLM via DeepSeek with a prompt (and optional system message).

    Args:
        prompt: User message.
        system: Optional system message.
        use_reasoning: Override the env default for reasoning. None = use env.
            When True, uses 'deepseek-reasoner' model.
        temperature: 0-1, lower = more deterministic (good for grading).
        max_tokens: Max tokens to generate.

    Returns:
        The text content of the assistant's response.
    """
    api_key = _get_api_key()
    model = _get_model()
    fallback_model = _get_fallback_model()

    # If reasoning is enabled, route to deepseek-reasoner
    enable_reasoning = _use_reasoning() if use_reasoning is None else use_reasoning
    if enable_reasoning:
        # deepseek-reasoner is a dedicated reasoning model — no need for a flag
        primary_model = "deepseek-reasoner"
        fallback_for_reasoning = model  # fall back to the configured model
    else:
        primary_model = model
        fallback_for_reasoning = fallback_model

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": primary_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        data = _call_deepseek(
            payload=payload,
            headers=headers,
            primary_model=primary_model,
            fallback_model=fallback_for_reasoning,
        )

        # Extract assistant message
        message = data["choices"][0]["message"]
        content = message.get("content") or ""

        # Some reasoning models return content in different fields
        if not content and message.get("reasoning"):
            content = message["reasoning"]

        return content.strip()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"DeepSeek request timed out after {_DEFAULT_TIMEOUT}s."
        )
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"DeepSeek request failed: {e}")


def llm_invoke_json(
    prompt: str,
    system: Optional[str] = None,
    use_reasoning: Optional[bool] = None,
) -> Dict[str, Any]:
    """Invoke the LLM and parse JSON from the response.

    Robust against markdown code fences and trailing text.
    """
    raw = llm_invoke(prompt, system, use_reasoning=use_reasoning)
    return _extract_json(raw)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first valid JSON object from a text that may contain
    markdown fences, prose, or trailing commentary."""
    if not text:
        return {"raw_response": "", "parse_error": True}

    # Strip code fences
    fence_pattern = r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```"
    fence_match = re.search(fence_pattern, text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    # Try direct JSON parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block (handles one level of nesting)
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: return empty dict with the raw text
    return {"raw_response": text, "parse_error": True}


def get_model_name() -> str:
    """Return the configured DeepSeek model name."""
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")


def get_provider() -> str:
    """Return the LLM provider name (for UI display)."""
    return "DeepSeek"
