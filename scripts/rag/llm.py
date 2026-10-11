"""LLM client with Qwen3 thinking on/off support.

  vllm / openai : OpenAI-compatible API (vLLM controls thinking via
                  extra_body.chat_template_kwargs.enable_thinking)
  ollama        : native /api/chat with an explicit think=true/false field
"""
import time, requests
from . import config


class LLM:
    def __init__(self, base_url=None, model=None, api_key=None, backend=None, thinking=False,
                 temperature=0.0, max_tokens=2000, timeout=600):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.backend = backend or config.LLM_BACKEND
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max(max_tokens, 8000) if thinking else max_tokens  # headroom for reasoning tokens
        self.timeout = timeout
        if self.backend != "ollama":
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=api_key or config.LLM_API_KEY, timeout=timeout)

    def _ollama(self, messages):
        url = self.base_url.replace("/v1", "") + "/api/chat"
        body = dict(model=self.model, messages=messages, stream=False, think=bool(self.thinking),
                    options=dict(temperature=self.temperature, num_predict=self.max_tokens, num_ctx=32768))
        r = requests.post(url, json=body, timeout=self.timeout); r.raise_for_status(); j = r.json()
        return j["message"].get("content", ""), j["message"].get("thinking"), j.get("prompt_eval_count"), j.get("eval_count")

    def _openai(self, messages):
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        if self.backend == "vllm":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(self.thinking)}}
        r = self.client.chat.completions.create(**kwargs); msg = r.choices[0].message; u = getattr(r, "usage", None)
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        return msg.content or "", reasoning, getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)

    def chat(self, system, user, retries=3):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last = None
        for i in range(retries):
            try:
                t0 = time.time()
                text, reasoning, pt, ct = self._ollama(messages) if self.backend == "ollama" else self._openai(messages)
                return dict(text=text, reasoning=reasoning, latency=time.time() - t0, prompt_tokens=pt, completion_tokens=ct)
            except Exception as e:
                last = e; time.sleep(2 * (i + 1))
        raise last
