from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_MODEL, OLLAMA_URL


class LLMService:
    """Small provider adapter. Returns None when no provider is available.

    The project is intentionally designed so retrieval/database answers still work
    even if an external LLM API is unavailable during a classroom demo.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER

    def generate(self, prompt: str, system: str | None = None) -> str | None:
        if self.provider == "openai":
            return self._openai(prompt, system)
        if self.provider == "ollama":
            return self._ollama(prompt, system)
        return None

    def _openai(self, prompt: str, system: str | None) -> str | None:
        if not OPENAI_API_KEY:
            return None
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            input_items = []
            if system:
                input_items.append({"role": "developer", "content": system})
            input_items.append({"role": "user", "content": prompt})
            response = client.responses.create(
                model=OPENAI_MODEL,
                input=input_items,
            )
            text = getattr(response, "output_text", None)
            return text.strip() if text else None
        except Exception:
            return None

    def _ollama(self, prompt: str, system: str | None) -> str | None:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        body = json.dumps({"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload.get("response")
            return text.strip() if text else None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
