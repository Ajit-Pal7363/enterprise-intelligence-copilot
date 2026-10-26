"""Optional LLM (Anthropic Messages API via plain HTTP). Without ANTHROPIC_API_KEY the whole app
still works in deterministic mode. A process-wide call cap protects a public demo's budget."""
import os
import requests

_calls = 0


def cap():
    return int(os.getenv("LLM_CALL_CAP", "200"))


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and _calls < cap()


def complete(system: str, user: str, max_tokens: int = 600, timeout: int = 30) -> str:
    global _calls
    _calls += 1
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": os.getenv("CLAUDE_MODEL", "claude-haiku-5-5"), "max_tokens": max_tokens,
              "system": system, "messages": [{"role": "user", "content": user}]},
        timeout=timeout)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]).strip()
