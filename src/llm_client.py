"""Simple OpenAI-compatible LLM client with local SQLite caching and JSON extraction."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Callable
import urllib.error
import urllib.request


def load_dotenv_file(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if it exists."""
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def extract_json_object(raw_text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response string."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Try direct parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Scan for the first balanced {...} block
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {raw_text[:200]}")

    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    return json.loads(candidate)

    raise ValueError(f"Unbalanced JSON braces in response: {raw_text[:200]}")


def _resolve_endpoint_and_key(model: str, api_key: str | None, base_url: str | None) -> tuple[str, str]:
    """Resolve the chat completions URL and API key based on model or environment variables."""
    load_dotenv_file()

    url = base_url or os.environ.get("LLM_BASE_URL")
    key = api_key or os.environ.get("LLM_API_KEY")

    if not url:
        if model.startswith("gemini"):
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            key = key or os.environ.get("GEMINI_API_KEY")
        elif "llama" in model.lower() or "mixtral" in model.lower() or os.environ.get("GROQ_API_KEY"):
            url = "https://api.groq.com/openai/v1/chat/completions"
            key = key or os.environ.get("GROQ_API_KEY")
        elif model.startswith("ollama/"):
            url = "http://localhost:11434/v1/chat/completions"
            key = key or "ollama"
        else:
            url = "https://api.openai.com/v1/chat/completions"
            key = key or os.environ.get("OPENAI_API_KEY")

    if url and not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"

    return url, (key or "")


class LLMClient:
    """Lightweight LLM client with deterministic SQLite response caching."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_path: str = "cache/llm_cache.db",
        temperature: float = 0.0,
        transport: Callable[[list[dict[str, str]], str, float], str] | None = None,
    ) -> None:
        load_dotenv_file()
        self.model = model or os.environ.get("LLM_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.transport = transport
        self.endpoint, self.api_key = _resolve_endpoint_and_key(self.model, api_key, base_url)

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT,
                    response_text TEXT,
                    created_at TEXT
                )
                """
            )

    def _make_cache_key(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps(
            {"model": self.model, "temperature": self.temperature, "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text response from cache or LLM API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cache_key = self._make_cache_key(messages)

        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute(
                "SELECT response_text FROM responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is not None:
                return row[0]

        if self.transport is not None:
            response_text = self.transport(messages, self.model, self.temperature)
        else:
            response_text = self._http_call(messages)

        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (cache_key, model, response_text, created_at) VALUES (?, ?, ?, ?)",
                (cache_key, self.model, response_text, datetime.now(timezone.utc).isoformat()),
            )

        return response_text

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Return parsed JSON dictionary from cache or LLM API."""
        raw_text = self.complete_text(system_prompt, user_prompt)
        return extract_json_object(raw_text)

    def _http_call(self, messages: list[dict[str, str]]) -> str:
        """Send standard OpenAI-compatible /chat/completions HTTP POST request."""
        if not self.api_key:
            raise RuntimeError(
                f"No API key found for model '{self.model}'. "
                "Set GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, or LLM_API_KEY in .env."
            )

        model_name = self.model.removeprefix("ollama/")
        body = json.dumps(
            {
                "model": model_name,
                "messages": messages,
                "temperature": self.temperature,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "vbc-fallacy-research/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as err:
            error_body = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API error ({err.code}): {error_body}") from err
