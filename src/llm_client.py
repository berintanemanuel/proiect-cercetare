"""Simple OpenAI-compatible LLM client with local SQLite caching and JSON extraction."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request


def load_dotenv_file(env_path: str | None = None) -> None:
    """Load simple KEY=VALUE pairs from the project .env file if it exists."""
    default_root_env = Path(__file__).resolve().parent.parent / ".env"
    path = Path(env_path) if env_path else (Path(".env") if Path(".env").exists() else default_root_env)
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


def _repair_json_candidate(text: str) -> str:
    """Repair common small-LLM JSON glitches (missing opening quote on string values, trailing commas)."""
    repaired = re.sub(
        r'(:\s*)(?!\s*["\[\{\d\-]|\s*(?:true|false|null)\b)([^"\s][^}]*?)"(?=\s*(?:,\s*"[^"]+"\s*:|\}))',
        lambda m: m.group(1) + json.dumps(m.group(2).strip()),
        text,
        flags=re.DOTALL,
    )
    return re.sub(r",\s*([}\]])", r"\1", repaired)


def extract_json_object(raw_text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response string."""
    cleaned = re.sub(r"<(?:thought|think)>.*?</(?:thought|think)>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
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
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        parsed = json.loads(_repair_json_candidate(candidate))
                    if isinstance(parsed, dict):
                        return parsed
                    raise ValueError(f"Expected JSON object in response: {raw_text[:200]}")

    # Fallback repair when an unmatched quote prevented balanced-brace scanning
    end = cleaned.rfind("}")
    if end > start:
        try:
            parsed = json.loads(_repair_json_candidate(cleaned[start : end + 1]))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Unbalanced JSON braces in response: {raw_text[:200]}")


def _is_local_endpoint(url: str) -> bool:
    """Return True if the endpoint targets a local loopback host (e.g. LM Studio or Ollama)."""
    hostname = urllib.parse.urlparse(url).hostname
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _resolve_endpoint_and_key(model: str, api_key: str | None, base_url: str | None) -> tuple[str, str]:
    """Resolve the chat completions URL and API key based on model or environment variables."""
    load_dotenv_file()

    url = base_url or os.environ.get("LLM_BASE_URL")
    key = api_key or os.environ.get("LLM_API_KEY")

    if not url:
        if model.startswith("lmstudio/"):
            url = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1/chat/completions")
            key = key or "lm-studio"
        elif model.startswith("ollama/"):
            url = "http://localhost:11434/v1/chat/completions"
            key = key or "ollama"
        elif model.startswith("gemini"):
            url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
            key = key or os.environ.get("GEMINI_API_KEY")
        elif "llama" in model.lower() or "mixtral" in model.lower() or os.environ.get("GROQ_API_KEY"):
            url = "https://api.groq.com/openai/v1/chat/completions"
            key = key or os.environ.get("GROQ_API_KEY")
        else:
            url = "https://api.openai.com/v1/chat/completions"
            key = key or os.environ.get("OPENAI_API_KEY")

    if url and not url.endswith("/chat/completions"):
        parsed = urllib.parse.urlparse(url)
        suffix = "/v1/chat/completions" if not parsed.path.strip("/") else "/chat/completions"
        url = url.rstrip("/") + suffix

    if url and not key and _is_local_endpoint(url):
        key = "local"

    return url, (key or "")


GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
)


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
        self.model = model or os.environ.get("LLM_MODEL", "gemini-3-flash-preview")
        self.temperature = temperature
        self.transport = transport
        self.endpoint, self.api_key = _resolve_endpoint_and_key(self.model, api_key, base_url)
        self._exhausted_models: set[str] = set()

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

    def _make_cache_key(self, messages: list[dict[str, str]], model_override: str | None = None) -> str:
        target_model = model_override or self.model
        payload = json.dumps(
            {"model": target_model, "temperature": self.temperature, "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _candidate_cache_keys(self, messages: list[dict[str, str]]) -> list[str]:
        candidate_keys = [self._make_cache_key(messages, self.model)]
        if self.transport is None and "generativelanguage.googleapis.com" in self.endpoint:
            for m in GEMINI_FALLBACK_MODELS:
                k = self._make_cache_key(messages, m)
                if k not in candidate_keys:
                    candidate_keys.append(k)
        return candidate_keys

    def _get_cached_response(self, messages: list[dict[str, str]]) -> str | None:
        with sqlite3.connect(self.cache_path) as conn:
            for key in self._candidate_cache_keys(messages):
                row = conn.execute(
                    "SELECT response_text FROM responses WHERE cache_key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    return row[0]
        return None

    def _save_cached_response(self, messages: list[dict[str, str]], response_text: str) -> None:
        cache_key = self._make_cache_key(messages, self.model)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (cache_key, model, response_text, created_at) VALUES (?, ?, ?, ?)",
                (cache_key, self.model, response_text, datetime.now(timezone.utc).isoformat()),
            )

    def _delete_cached_responses(self, messages: list[dict[str, str]]) -> None:
        keys = self._candidate_cache_keys(messages)
        placeholders = ",".join("?" for _ in keys)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(f"DELETE FROM responses WHERE cache_key IN ({placeholders})", keys)

    def _invoke_model(self, messages: list[dict[str, str]]) -> str:
        if self.transport is not None:
            return self.transport(messages, self.model, self.temperature)
        return self._http_call(messages)

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text response from cache or LLM API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cached = self._get_cached_response(messages)
        if cached is not None:
            return cached

        response_text = self._invoke_model(messages)
        self._save_cached_response(messages, response_text)
        return response_text

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Return parsed JSON dictionary from cache or LLM API, retrying on invalid JSON."""
        base_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        cached = self._get_cached_response(base_messages)
        if cached is not None:
            try:
                return extract_json_object(cached)
            except (ValueError, json.JSONDecodeError):
                self._delete_cached_responses(base_messages)

        last_err: Exception | None = None
        for attempt in range(3):
            if attempt == 0:
                messages = base_messages
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            "IMPORTANT: Return ONLY one valid JSON object with all keys and string values properly double-quoted."
                        ),
                    },
                ]
            raw_text = self._invoke_model(messages)
            try:
                parsed = extract_json_object(raw_text)
                self._save_cached_response(base_messages, raw_text)
                return parsed
            except (ValueError, json.JSONDecodeError) as err:
                last_err = err

        raise ValueError(f"Failed to parse JSON after retries: {last_err}") from last_err

    def _switch_to_next_gemini_model(self, reason: str) -> bool:
        """Rotate to the next available Gemini free-tier model when quota/503 is hit."""
        if "generativelanguage.googleapis.com" not in self.endpoint:
            return False
        self._exhausted_models.add(self.model)
        for candidate in GEMINI_FALLBACK_MODELS:
            if candidate not in self._exhausted_models:
                print(f"  [Model Switch] '{self.model}' unavailable ({reason}) -> switching to '{candidate}'")
                self.model = candidate
                return True
        return False

    def _http_call(self, messages: list[dict[str, str]]) -> str:
        """Send standard OpenAI-compatible /chat/completions HTTP POST request."""
        if not self.api_key:
            raise RuntimeError(
                f"No API key found for model '{self.model}'. "
                "Set GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY, or LLM_API_KEY in .env."
            )

        is_local = _is_local_endpoint(self.endpoint)
        is_gemini = "generativelanguage.googleapis.com" in self.endpoint
        timeout_sec = 180 if is_local else 60

        for attempt in range(10):
            model_name = self.model.removeprefix("ollama/").removeprefix("lmstudio/")
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
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if is_gemini:
                        time.sleep(3.0)  # Smoothly pace live requests under free-tier 15 RPM limit
                    return data["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as err:
                error_body = err.read().decode("utf-8", errors="replace")
                # Only switch model permanently if its DAILY quota (GenerateRequestsPerDay) is exhausted
                if err.code == 429 and "GenerateRequestsPerDay" in error_body:
                    if self._switch_to_next_gemini_model("daily quota reached"):
                        continue
                if err.code in (429, 500, 503):
                    wait_sec = 12 if err.code == 429 else 6 * (attempt + 1)
                    print(f"  [HTTP {err.code} on {self.model}] Waiting {wait_sec}s before retry...")
                    time.sleep(wait_sec)
                    continue
                raise RuntimeError(f"LLM API error ({err.code}): {error_body}") from err
            except (TimeoutError, urllib.error.URLError) as err:
                wait_sec = 6 * (attempt + 1)
                print(f"  [Timeout/Network error on {self.model}: {err}] Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
                continue
        raise RuntimeError("LLM API failed after maximum retries.")
