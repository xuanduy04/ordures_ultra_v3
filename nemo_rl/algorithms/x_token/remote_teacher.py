"""Read full-vocabulary scores from an unchanged vLLM completions server."""

import asyncio
import json
import os

import aiohttp
import torch

from nemo_rl.algorithms.opd_outsource import VLLMTeacherLogprobClient


class TeacherCapabilityError(RuntimeError):
    """The existing endpoint cannot provide the required dense distribution."""


class XTokenFullLogprobClient(VLLMTeacherLogprobClient):
    """Reuse the existing HTTP session/worker/future lifecycle for dense scores.

    vLLM returns normalized full log-probabilities, not raw model logits.
    They are valid logits up to a per-position additive constant. Full-vocab
    KL (including temperature scaling) is invariant to that constant. The
    remote recipe therefore requires full-vocab P-KL, rather than silently
    changing upstream raw-logit-based microbatch top-k support selection.
    """

    def __init__(self, config):
        for key in ("model_vocab_size", "max_score_elements", "max_response_bytes", "request_timeout_s", "max_attempts", "normalization_atol"):
            if config[key] <= 0:
                raise ValueError(key + " must be positive")
        if config["retry_backoff_s"] < 0:
            raise ValueError("retry_backoff_s must be nonnegative")
        self.config = config
        super().__init__()

    async def _score_on_loop(self, teacher, input_ids, input_lengths):
        if input_lengths is None:
            raise ValueError("Explicit unpadded teacher lengths are required")
        if input_ids.ndim != 2 or input_lengths.ndim != 1 or len(input_lengths) != input_ids.shape[0]:
            raise ValueError("Expected [batch, sequence] IDs and one length per sample")
        lengths = [int(length) for length in input_lengths]
        if not lengths or any(not 2 <= length <= input_ids.shape[1] for length in lengths):
            raise ValueError("Every teacher sequence must have between two and padded-length tokens")
        prompts = [input_ids[index, :length].tolist() for index, length in enumerate(lengths)]
        vocab = self.config["model_vocab_size"]
        elements = input_ids.numel() * vocab
        if elements > self.config["max_score_elements"]:
            raise ValueError("Dense teacher response exceeds max_score_elements; reduce the microbatch/context")
        payload = {"model": teacher["model"], "prompt": prompts, "echo": True,
                   "max_tokens": 0, "logprobs": None, "prompt_logprobs": -1,
                   "return_token_ids": True, "return_tokens_as_token_ids": True,
                   "temperature": 1.0, "add_special_tokens": False, "stream": False}
        key_name = self.config["api_key_env"]
        headers = {"Authorization": "Bearer " + (os.environ[key_name] if key_name else "EMPTY")}
        response = await self._dense_request(teacher, payload, headers)
        choices = response["choices"]
        if len(choices) != len(prompts):
            raise TeacherCapabilityError("Teacher returned the wrong number of choices")
        result = torch.zeros(len(prompts), input_ids.shape[1], vocab, dtype=torch.float32)
        seen = set()
        for choice in choices:
            index = choice["index"]
            if not isinstance(index, int) or index in seen or not 0 <= index < len(prompts):
                raise TeacherCapabilityError("Teacher returned duplicate/invalid choice indices")
            seen.add(index)
            sent = prompts[index]
            if choice.get("prompt_token_ids") != sent:
                raise TeacherCapabilityError("Teacher did not echo the exact teacher-tokenizer IDs")
            dense = self._parse_dense_choice(choice, len(sent), vocab)
            # API row i scores token i from prefix [:i]. X-Token row i predicts i+1.
            result[index, :len(sent) - 1] = dense[1:]
        print("XTOKEN_TEACHER_SCORE=" + json.dumps({
            "batch_size": len(prompts), "scored_positions": sum(len(row) - 1 for row in prompts),
            "model_vocab_size": vocab, "representation": "full_logprobs",
        }), flush=True)
        return result

    def _parse_dense_choice(self, choice, length, vocab):
        rows = choice.get("prompt_logprobs")
        numeric_keys = rows is not None
        if rows is None:
            logprobs = choice.get("logprobs")
            rows = logprobs.get("top_logprobs") if logprobs is not None else None
        if rows is None or length < 2 or len(rows) != length or rows[0] is not None:
            raise TeacherCapabilityError("Expected full prompt logprobs with a null first position")
        dense = torch.zeros(length, vocab, dtype=torch.float32)
        for position, row in enumerate(rows[1:], start=1):
            if row is None or len(row) != vocab:
                raise TeacherCapabilityError("Teacher returned top-k rather than full vocabulary; server capability is insufficient")
            ids, values = [], []
            for key, value in row.items():
                if numeric_keys:
                    token_id = int(key)
                    score = value["logprob"] if isinstance(value, dict) else value
                else:
                    if not str(key).startswith("token_id:"):
                        raise TeacherCapabilityError("Decoded token strings are ambiguous; numeric token IDs are required")
                    token_id = int(str(key).split(":", 1)[1])
                    score = value
                ids.append(token_id)
                values.append(float(score))
            if set(ids) != set(range(vocab)):
                raise TeacherCapabilityError("Missing/duplicate/out-of-range vocabulary IDs")
            dense[position, ids] = torch.tensor(values)
            scores = dense[position]
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise TeacherCapabilityError("Non-finite teacher distribution")
            if not torch.isclose(scores.exp().sum(), torch.tensor(1.), atol=self.config["normalization_atol"], rtol=0):
                raise TeacherCapabilityError("Teacher scores are not normalized full-vocabulary logprobs")
        return dense

    async def _dense_request(self, teacher, payload, headers):
        url = teacher["url"].rstrip("/") + "/v1/completions"
        for attempt in range(self.config["max_attempts"]):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config["request_timeout_s"])
                async with self._session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status >= 400:
                        if response.status not in {429, 500, 502, 503, 504}:
                            raise TeacherCapabilityError(f"Teacher rejected full-vocab scoring (HTTP {response.status}); server was not modified")
                        raise RuntimeError(f"Transient teacher HTTP {response.status}")
                    limit = self.config["max_response_bytes"]
                    if response.content_length is not None and response.content_length > limit:
                        raise TeacherCapabilityError("Teacher response exceeds configured byte limit")
                    body = bytearray()
                    async for chunk in response.content.iter_chunked(65536):
                        body.extend(chunk)
                        if len(body) > limit:
                            raise TeacherCapabilityError("Teacher response exceeds configured byte limit")
                    return json.loads(body)
            except TeacherCapabilityError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, json.JSONDecodeError):
                if attempt + 1 == self.config["max_attempts"]:
                    raise
                await asyncio.sleep(self.config["retry_backoff_s"])


class RemoteTeacherPolicy:
    """Teacher policy facade with no GPU allocation or server-side mutations."""

    data_parallel_size = 1

    def __init__(self, config):
        self.config = config
        self.uses_topk_scores = config.get("score_mode") == "topk"
        self.client = XTokenTopKLogprobClient(config) if self.uses_topk_scores else XTokenFullLogprobClient(config)

    def get_topk_scores(self, data, micro_batch_size):
        width = self.config["top_k"] + 1
        shape = (*data["input_ids"].shape, width)
        if data["input_ids"].numel() * width > self.config["max_score_elements"]:
            raise ValueError("Global top-k batch exceeds max_score_elements")
        result = {"topk_ids": torch.full(shape, -1, dtype=torch.long),
                  "topk_logprobs": torch.full(shape, -float("inf")),
                  "topk_valid": torch.zeros(shape, dtype=torch.bool)}
        for offset in range(0, data.size, micro_batch_size):
            batch = data.get_batch(batch_idx=offset // micro_batch_size, batch_size=micro_batch_size)
            teacher = {"url": self.config["url"], "model": self.config["model"]}
            scores = asyncio.run(self.client.score_group(teacher, batch["input_ids"], batch["input_lengths"]))
            for key, value in scores.items():
                result[key][offset:offset + value.shape[0]].copy_(value)
        return result

    def get_full_logits(self, data, micro_batch_size):
        shape = (*data["input_ids"].shape, self.config["model_vocab_size"])
        if data["input_ids"].numel() * shape[-1] > self.config["max_score_elements"]:
            raise ValueError("Global dense teacher batch exceeds max_score_elements; reduce global batch/context")
        result = torch.empty(shape, dtype=torch.float32)
        for offset in range(0, data.size, micro_batch_size):
            batch = data.get_batch(batch_idx=offset // micro_batch_size, batch_size=micro_batch_size)
            teacher = {"url": self.config["url"], "model": self.config["model"]}
            scores = asyncio.run(self.client.score_group(teacher, batch["input_ids"], batch["input_lengths"]))
            result[offset:offset + scores.shape[0]].copy_(scores)
        return result

    def prepare_for_lp_inference(self):
        pass

    def offload_after_refit(self):
        pass

    def release_ipc_buffer(self):
        pass

    def close(self):
        asyncio.run(self.client.close())


class XTokenTopKLogprobClient(XTokenFullLogprobClient):
    """Preserve global-normalized top-k probabilities and the optional actual token."""

    def __init__(self, config):
        super().__init__(config)
        if not 1 <= config["top_k"] <= config["model_vocab_size"]:
            raise ValueError("top_k must be between 1 and model_vocab_size")

    async def _score_on_loop(self, teacher, input_ids, input_lengths):
        if input_lengths is None or input_ids.ndim != 2 or input_lengths.ndim != 1 or len(input_lengths) != input_ids.shape[0]:
            raise ValueError("Explicit unpadded teacher lengths are required")
        lengths = [int(length) for length in input_lengths]
        if not lengths or any(not 2 <= length <= input_ids.shape[1] for length in lengths):
            raise ValueError("Invalid teacher sequence length")
        k = self.config["top_k"]
        if input_ids.numel() * (k + 1) > self.config["max_score_elements"]:
            raise ValueError("Top-k response exceeds max_score_elements")
        prompts = [input_ids[index, :length].tolist() for index, length in enumerate(lengths)]
        payload = {"model": teacher["model"], "prompt": prompts, "echo": True,
                   "max_tokens": 0, "logprobs": None, "prompt_logprobs": k,
                   "return_token_ids": True, "return_tokens_as_token_ids": True,
                   "temperature": 1., "add_special_tokens": False, "stream": False}
        key_name = self.config["api_key_env"]
        headers = {"Authorization": "Bearer " + (os.environ[key_name] if key_name else "EMPTY")}
        response = await self._dense_request(teacher, payload, headers)
        choices = response["choices"]
        if len(choices) != len(prompts):
            raise TeacherCapabilityError("Wrong teacher choice count")
        shape = (*input_ids.shape, k + 1)
        result = {"topk_ids": torch.full(shape, -1, dtype=torch.long),
                  "topk_logprobs": torch.full(shape, -float("inf")),
                  "topk_valid": torch.zeros(shape, dtype=torch.bool)}
        seen = set()
        for choice in choices:
            index = choice["index"]
            if not isinstance(index, int) or index in seen or not 0 <= index < len(prompts):
                raise TeacherCapabilityError("Duplicate/invalid teacher choice index")
            seen.add(index)
            if choice.get("prompt_token_ids") != prompts[index]:
                raise TeacherCapabilityError("Teacher did not echo exact teacher-tokenizer IDs")
            rows = choice.get("prompt_logprobs")
            if rows is None or len(rows) != lengths[index] or rows[0] is not None:
                raise TeacherCapabilityError("Expected numeric prompt logprobs with null first position")
            for position, row in enumerate(rows[1:]):
                ids, scores = self._parse_topk_row(row)
                count = len(ids)
                result["topk_ids"][index, position, :count] = ids
                result["topk_logprobs"][index, position, :count] = scores
                result["topk_valid"][index, position, :count] = True
        print("XTOKEN_TEACHER_SCORE=" + json.dumps({"batch_size": len(prompts),
              "scored_positions": sum(length - 1 for length in lengths),
              "model_vocab_size": self.config["model_vocab_size"],
              "representation": "topk_logprobs", "top_k": k}), flush=True)
        return result

    def _parse_topk_row(self, row):
        k, vocab = self.config["top_k"], self.config["model_vocab_size"]
        if row is None or not k <= len(row) <= min(k + 1, vocab):
            raise TeacherCapabilityError("Teacher returned fewer than requested top-k scores or too many scores")
        ids = torch.tensor([int(key) for key in row], dtype=torch.long)
        if len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= vocab).any():
            raise TeacherCapabilityError("Invalid/duplicate top-k token IDs")
        scores = torch.tensor([float(value["logprob"] if isinstance(value, dict) else value) for value in row.values()])
        if torch.isnan(scores).any() or torch.isposinf(scores).any():
            raise TeacherCapabilityError("Invalid top-k log-probabilities")
        if scores.exp().sum() > 1 + self.config["normalization_atol"]:
            raise TeacherCapabilityError("Top-k probability mass exceeds one")
        return ids, scores
