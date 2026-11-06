"""llm.py -- thin, defensive wrapper around the Ollama Python client.

Set OLLAMA_HOST to talk to a remote machine, e.g.
    OLLAMA_HOST=http://192.168.1.50:11434 streamlit run dashboard.py
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import ollama

HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("TRUST_RADAR_MODEL", "qwen2.5:3b")

_client = None


def client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=HOST)
    return _client


def list_models() -> list:
    """Names of locally available models; [] if the server is unreachable."""
    try:
        resp = client().list()
        models = resp.get("models", []) if isinstance(resp, dict) else resp.models
        names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            names.append(name or m.get("name"))
        return sorted(n for n in names if n)
    except Exception:                                   # noqa: BLE001
        return []


def server_ok() -> bool:
    try:
        client().list()
        return True
    except Exception:                                   # noqa: BLE001
        return False


@dataclass
class Generation:
    text: str
    tokens: list = field(default_factory=list)   # [{token, p, top:[{token,p}]}]
    seconds: float = 0.0


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _strip_thinking(text: str, tokens: list):
    """Drop a <think>...</think> preamble that some models emit anyway."""
    if "</think>" not in text:
        return text, tokens
    cut = text.index("</think>") + len("</think>")
    answer = text[cut:]
    lead = len(answer) - len(answer.lstrip())
    answer = answer.strip()
    # drop tokens up to the end of the think block (+ leading whitespace)
    acc = ""
    keep_from = len(tokens)
    for i, tok in enumerate(tokens):
        acc += tok["token"]
        if len(acc) >= cut + lead:
            keep_from = i + 1
            break
    return answer, tokens[keep_from:]


_THINK_SUPPORTED: dict = {}


def generate(model: str, prompt: str, system: str = "", temperature: float = 0.0,
             seed: int = 0, num_predict: int = 64, logprobs: bool = False,
             top_logprobs: int = 5) -> Generation:
    """One completion; with logprobs=True every token carries p and top-k."""
    kwargs = dict(
        model=model,
        prompt=prompt,
        options={"temperature": temperature, "seed": seed, "num_predict": num_predict},
    )
    if system:
        kwargs["system"] = system
    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs

    # Thinking models (qwen3, deepseek-r1...) would spend the token budget on
    # hidden reasoning; switch it off where supported, quietly retry otherwise.
    use_think = _THINK_SUPPORTED.get(model, True)
    start = time.time()
    try:
        resp = client().generate(**kwargs, **({"think": False} if use_think else {}))
    except Exception as exc:                            # noqa: BLE001
        if use_think and "think" in str(exc).lower():
            _THINK_SUPPORTED[model] = False
            resp = client().generate(**kwargs)
        else:
            raise
    elapsed = time.time() - start

    text = _field(resp, "response", "") or ""
    tokens = []
    for item in _field(resp, "logprobs", None) or []:
        lp = _field(item, "logprob")
        if lp is None:
            continue
        top = []
        for alt in _field(item, "top_logprobs", None) or []:
            alp = _field(alt, "logprob")
            if alp is not None:
                top.append({"token": _field(alt, "token", ""), "p": math.exp(float(alp))})
        tokens.append({"token": _field(item, "token", ""), "p": math.exp(float(lp)), "top": top})

    text, tokens = _strip_thinking(text, tokens)
    return Generation(text=text.strip() if not tokens else text, tokens=tokens, seconds=elapsed)
