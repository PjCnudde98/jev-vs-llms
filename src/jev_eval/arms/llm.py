"""Arms B and C: LLMs with structured output (JSON schema), called directly.

Both providers get the same system prompt, user message and output schema from
jev_eval.prompts. Claude does not expose log-probabilities, so `confidence` is the
model's own 0-1 estimate, returned as part of the schema.
"""

from __future__ import annotations

import json
import time

from jev_eval import pricing
from jev_eval.arms.base import Arm, RetryableError
from jev_eval.prompts import llm_output_schema, llm_system_prompt, llm_user_message, select_examples
from jev_eval.types import FieldAnswer, Item, Prediction


class _LLMArm(Arm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = None
        self._system: str | None = None
        self._schema = llm_output_schema(self.task)

    def fit(self, train: list[Item]) -> None:
        examples = select_examples(train, self.task, self.params.get("examples_per_label", 0))
        self._system = llm_system_prompt(self.task, examples)

    def _parse(self, text: str) -> tuple[dict[str, FieldAnswer], bool]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}, False
        answers: dict[str, FieldAnswer] = {}
        valid = True
        for field in self.task.fields:
            entry = data.get(field.name) if isinstance(data, dict) else None
            label = entry.get("label") if isinstance(entry, dict) else None
            if label not in field.labels:
                valid = False
                answers[field.name] = FieldAnswer(label=None)
                continue
            confidence = entry.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence = min(max(float(confidence), 0.0), 1.0)
            else:
                confidence = None
            answers[field.name] = FieldAnswer(label=label, confidence=confidence)
        return answers, valid


class AnthropicArm(_LLMArm):
    """Params:
    effort: "low" | "medium" | "high" ... (omit for Haiku 4.5, which rejects it)
    max_tokens: output cap including thinking (default 4096)
    prompt_cache: put a cache breakpoint on the system prompt (default true)
    examples_per_label: few-shot examples per label in the prompt (default 0)
    Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
    """

    kind = "anthropic"

    def _get_client(self):
        if self._client is None:
            import anthropic

            # Retries are off here: the runner retries, so backoff never counts as latency.
            self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=self.params.get("timeout", 120.0))
        return self._client

    async def predict(self, item: Item) -> Prediction:
        import anthropic

        if self._system is None:
            self.fit([])
        system_block = {"type": "text", "text": self._system}
        if self.params.get("prompt_cache", True):
            system_block["cache_control"] = {"type": "ephemeral"}
        output_config: dict = {"format": {"type": "json_schema", "schema": self._schema}}
        if self.params.get("effort"):
            output_config["effort"] = self.params["effort"]

        client = self._get_client()
        start = time.perf_counter()
        try:
            response = await client.messages.create(
                model=self.model,
                max_tokens=self.params.get("max_tokens", 4096),
                system=[system_block],
                messages=[{"role": "user", "content": llm_user_message(item)}],
                output_config=output_config,
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            raise RetryableError(
                f"{type(e).__name__}: {e}", float(retry_after) if retry_after and retry_after.isdigit() else None
            ) from e
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise RetryableError(f"HTTP {e.status_code}: {e}") from e
            return self._invalid(item, f"HTTP {e.status_code}: {e}")
        latency = time.perf_counter() - start

        usage = response.usage
        tokens = {
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", None) or 0,
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", None) or 0,
        }
        cost = pricing.cost_usd(self.model, **tokens)
        text = next((b.text for b in response.content if b.type == "text"), "")
        if response.stop_reason in ("refusal", "max_tokens"):
            return self._invalid(
                item, f"stop_reason={response.stop_reason}", cost_usd=cost, latency_s=latency, raw=text[:2000], **tokens
            )
        answers, valid = self._parse(text)
        return Prediction(
            arm=self.name,
            model=self.model,
            model_version=response.model,
            item_id=item.id,
            answers=answers,
            gold=item.labels,
            valid=valid,
            error=None if valid else "unparseable or out-of-schema output",
            cost_usd=cost,
            latency_s=latency,
            raw=None if valid else text[:2000],
            **tokens,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


class OpenAIArm(_LLMArm):
    """Params:
    reasoning_effort: passed through when set (reasoning models)
    max_completion_tokens: output cap (default 4096)
    Needs OPENAI_API_KEY and `uv sync --extra openai`. Add the model's price to
    configs/pricing.yaml first.
    """

    kind = "openai"

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(max_retries=0, timeout=self.params.get("timeout", 120.0))
        return self._client

    async def predict(self, item: Item) -> Prediction:
        import openai

        if self._system is None:
            self.fit([])
        kwargs: dict = {}
        if self.params.get("reasoning_effort"):
            kwargs["reasoning_effort"] = self.params["reasoning_effort"]
        client = self._get_client()
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system},
                    {"role": "user", "content": llm_user_message(item)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "classification", "schema": self._schema, "strict": True},
                },
                max_completion_tokens=self.params.get("max_completion_tokens", 4096),
                **kwargs,
            )
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise RetryableError(f"HTTP {e.status_code}: {e}") from e
            return self._invalid(item, f"HTTP {e.status_code}: {e}")
        latency = time.perf_counter() - start

        usage = response.usage
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) or 0
        tokens = {
            "input_tokens": (usage.prompt_tokens or 0) - cached,
            "output_tokens": usage.completion_tokens or 0,
            "cache_read_tokens": cached,
        }
        cost = pricing.cost_usd(self.model, **tokens)
        text = response.choices[0].message.content or ""
        answers, valid = self._parse(text)
        return Prediction(
            arm=self.name,
            model=self.model,
            model_version=response.model,
            item_id=item.id,
            answers=answers,
            gold=item.labels,
            valid=valid,
            error=None if valid else "unparseable or out-of-schema output",
            cost_usd=cost,
            latency_s=latency,
            raw=None if valid else text[:2000],
            **tokens,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
