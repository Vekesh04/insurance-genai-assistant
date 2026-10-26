from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OllamaLLM:
    """Optional local LLM adapter. No Python package is required."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")

    def generate(self, prompt: str, timeout: float = 8.0) -> str | None:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return (data.get("response") or "").strip() or None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
