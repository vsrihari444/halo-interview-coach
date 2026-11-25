"""Minimal OpenAI-compatible chat client (stdlib only). Works with OpenRouter, vLLM, or any compatible server."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def chat(messages, model, base_url=None, api_key=None, temperature=0.7, max_tokens=1500, retries=4, timeout=120, json_mode=False):
    base_url = (base_url or os.environ.get("LLM_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
    api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "EMPTY"
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if "openrouter.ai" in base_url:
        headers["HTTP-Referer"] = "https://github.com/vsrihari444/halo-interview-coach"
        headers["X-Title"] = "Halo Interview Coach"
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(f"{base_url}/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = e
            code = getattr(e, "code", None)
            if code in (400, 401, 403, 404):
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")
