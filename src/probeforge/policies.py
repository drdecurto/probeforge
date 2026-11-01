import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

from .environment import available_probes
from .io import canonical, digest, write_json
from .oracle import action_values


class BudgetExceeded(RuntimeError):
    pass


class RandomProbeIntervention:
    """Evaluation-only: replace a selected probe by a random available probe of equal cost."""
    def __init__(self, policy):
        self.policy = policy

    def generate(self, messages, seed=0):
        from .environment import parse_action
        text, usage = self.policy.generate(messages, seed)
        try:
            action = parse_action(text)
        except ValueError:
            return text, usage
        if action["action"] != "probe":
            return text, usage
        state = json.loads(messages[-1]["content"])
        probes = available_probes(state)
        original = next((p for p in probes if p["id"] == action["probe_id"]), None)
        if original is None:
            return text, usage
        choices = [p for p in probes if p["cost"] == original["cost"]]
        selected = random.Random(seed).choice(choices)
        return canonical({"action": "probe", "probe_id": selected["id"]}), {**usage, "policy_proposal": text}


class RulePolicy:
    def __init__(self, mode="oracle", cost_penalty=0.02):
        self.mode, self.cost_penalty = mode, cost_penalty

    def generate(self, messages, seed=0):
        state = json.loads(messages[-1]["content"])
        ranked = action_values(state, self.cost_penalty)
        if self.mode == "oracle":
            action = ranked[0]["action"]
        elif self.mode == "no_probe":
            action = next(r["action"] for r in ranked if r["action"]["action"] == "answer")
        elif self.mode == "random":
            probes = available_probes(state)
            action = {"action": "probe", "probe_id": random.Random(seed).choice(probes)["id"]} if probes else next(r["action"] for r in ranked if r["action"]["action"] == "answer")
        else:
            raise ValueError(self.mode)
        return canonical(action), {}  # Symbolic arithmetic, not LLM token measurements.


class APIPolicy:
    """OpenAI-compatible endpoint; credentials stay in the process environment.

    max_requests caps HTTP attempts including retries. Cached calls cost no requests.
    Token accounting comes from server usage, not an invented tokenizer estimate.
    """
    def __init__(self, model, base_url=None, key_env="NEBIUS_API_KEY", cache="data/api-cache",
                 max_requests=1000, max_tokens=256, temperature=0.0, extra=None):
        self.model = model
        self.base_url = (base_url or os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")).rstrip("/")
        self.key = os.getenv(key_env, "")
        if not self.key and not self.base_url.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(f"Set {key_env} in the environment")
        self.cache = Path(cache)
        self.max_requests, self.max_tokens, self.temperature = max_requests, max_tokens, temperature
        self.extra = extra or {}
        if set(self.extra) & {"model", "messages", "max_tokens", "temperature"}:
            raise ValueError("extra may not override model, messages, token cap, or temperature")
        self.requests, self.usage = 0, {"prompt_tokens": 0, "completion_tokens": 0}
        self.cache_hits = 0

    def generate(self, messages, seed=0):
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                   "temperature": self.temperature, **self.extra}
        # Seed is a cache namespace; provider-side reproducibility is not assumed.
        key = digest({"url": self.base_url, "payload": payload, "sample": seed})
        path = self.cache / f"{key}.json"
        if path.exists():
            obj = json.loads(path.read_text())
            self.cache_hits += 1
            return obj["text"], obj["usage"]
        for attempt in range(4):
            if self.requests >= self.max_requests:
                raise BudgetExceeded("API HTTP-attempt budget reached; completed episodes are saved")
            self.requests += 1
            request = urllib.request.Request(self.base_url + "/chat/completions", data=canonical(payload).encode(),
                       headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    result = json.load(response)
                text = result["choices"][0]["message"].get("content")
                if not isinstance(text, str):
                    raise ValueError("Endpoint returned no textual content")
                raw = result.get("usage", {})
                usage = {k: int(raw.get(k, 0)) for k in self.usage}
                for k, v in usage.items():
                    self.usage[k] += v
                usage["token_usage_available"] = all(k in raw for k in ("prompt_tokens", "completion_tokens"))
                write_json(path, {"text": text, "usage": usage, "model": self.model,
                                  "finish_reason": result["choices"][0].get("finish_reason")})
                return text, usage
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f"Endpoint HTTP {exc.code}; check model ID, endpoint and credentials") from None
                time.sleep(min(2**attempt, 8))
            except urllib.error.URLError:
                if attempt == 3:
                    raise RuntimeError("Endpoint network failure after four attempts") from None
                time.sleep(min(2**attempt, 8))


class HFPolicy:
    def __init__(self, model, adapter=None, device="auto", quantize=False, max_tokens=128,
                 temperature=0.0, revision="main"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device, self.max_tokens, self.temperature = device, max_tokens, temperature
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16 if device == "mps" else torch.float32
        kwargs = {"torch_dtype": dtype, "revision": revision, "trust_remote_code": False, "attn_implementation": "sdpa"}
        if quantize:
            if not device.startswith("cuda"):
                raise ValueError("4-bit path requires NVIDIA CUDA; use a smaller model on Mac")
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True), device_map={"": device})
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or model, revision=revision if not adapter else "main", trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        if not quantize:
            self.model.to(device)
        self.model.eval()

    def generate(self, messages, seed=0):
        from .training_data import render_prompt
        torch = self.torch
        text = render_prompt(self.tokenizer, messages)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False).to(self.device)
        if inputs["input_ids"].shape[1] + self.max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds model context; no silent truncation")
        kwargs = {"max_new_tokens": self.max_tokens, "do_sample": self.temperature > 0,
                  "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id}
        if self.temperature > 0:
            kwargs["temperature"] = self.temperature
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": len(tokens)}
