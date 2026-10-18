"""Thin client for the local LLM/VLM (Qwen2.5-VL behind an OpenAI-compatible API).

Works with Ollama (`ollama serve`) or vLLM (`vllm serve ...`). One model handles
both text (witness statements) and images (CCTV frames, generated images).
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any, List, Optional, Union

from config import load_config

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        cfg = load_config()["llm"]
        _client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=cfg.get("timeout_sec", 180))
    return _client


def image_to_data_url(image) -> str:
    """Accept a file path or PIL image, return a base64 data URL."""
    from PIL import Image

    if isinstance(image, str):
        image = Image.open(image)
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(prompt: str, system: str = "", images: Optional[List[Any]] = None,
         json_mode: bool = False, temperature: Optional[float] = None) -> str:
    cfg = load_config()["llm"]
    content: Union[str, list] = prompt
    if images:
        content = [{"type": "image_url", "image_url": {"url": image_to_data_url(im)}}
                   for im in images]
        content.append({"type": "text", "text": prompt})
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    kwargs = dict(model=cfg["model"], messages=messages,
                  temperature=cfg["temperature"] if temperature is None else temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = _get_client().chat.completions.create(**kwargs)
    except Exception:
        if not json_mode:
            raise
        # Some servers don't support response_format; the prompt still asks for JSON.
        kwargs.pop("response_format")
        resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> Any:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"No JSON found in model output: {text[:200]}")


def chat_json(prompt: str, system: str = "", images: Optional[List[Any]] = None,
              retries: int = 2, temperature: Optional[float] = None) -> Any:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            return parse_json(chat(prompt, system=system, images=images, json_mode=True,
                                   temperature=temperature))
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
    raise RuntimeError(f"Model did not return valid JSON: {last_err}")


def yes_no(question: str, image) -> Optional[bool]:
    """Ask the VLM a yes/no question about an image (used for verification)."""
    reply = chat(f"{question}\nAnswer with exactly one word: yes or no.",
                 images=[image], temperature=0.0).strip().lower()
    if reply.startswith("yes"):
        return True
    if reply.startswith("no"):
        return False
    return None
