""" Minimal, free Ollama client with JSON-schema structured output """
from __future__ import annotations

import json
import logging

import httpx

log = logging.getLogger(__name__)


class Ollama:
    def __init__(self, url: str, model: str, num_ctx: int = 16384, timeout: float = 600,
                 max_tokens: int = 3000):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        # Cap on generated tokens: at temperature 0 a model can loop forever inside a JSON list
        self.max_tokens = max_tokens
        self.client = httpx.Client(timeout=timeout)

    def available(self) -> bool:
        try:
            tags = self.client.get(f"{self.url}/api/tags").json()
        except httpx.HTTPError:
            return False
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            log.error("Model %s is not pulled. Run: ollama pull %s", self.model, self.model)
            return False
        return True

    def chat_json(self, system: str, user: str, schema: dict, retries: int = 2) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            "think": False,  # Qwen3 etc.: skip the reasoning trace, we want JSON fast
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
        }
        last_err = None
        for attempt in range(retries + 1):
            try:
                r = self.client.post(f"{self.url}/api/chat", json=payload)
                r.raise_for_status()
                body = r.json()
                if body.get("done_reason") == "length":
                    # Deterministic sampling: retrying would loop the same way.
                    raise RuntimeError(f"LLM output hit the {self.max_tokens}-token cap (runaway generation)")
                return json.loads(body["message"]["content"])
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
                last_err = e
                log.warning("LLM call failed (attempt %d): %s", attempt + 1, e)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")
