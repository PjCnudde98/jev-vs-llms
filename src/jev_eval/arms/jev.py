"""Arm A: Jev through the Typesafe API (typesafe-sdk).

Every output field becomes one Choice question, and all fields go in one request
(Typesafe evaluates questions in parallel). Needs TYPESAFE_API_KEY.
"""

from __future__ import annotations

import time

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeError,
)

from jev_eval import pricing
from jev_eval.arms.base import Arm, RetryableError
from jev_eval.prompts import jev_criteria, jev_instructions, select_examples
from jev_eval.types import FieldAnswer, Item, Prediction


class JevArm(Arm):
    """Params:
    examples_per_label: few-shot examples added to each Choice option (default 0)
    timeout: seconds per request (default 30)
    """

    kind = "jev"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client: AsyncTypeSafeClient | None = None
        self._questions: dict[str, Choice] | None = None

    def fit(self, train: list[Item]) -> None:
        examples = select_examples(train, self.task, self.params.get("examples_per_label", 0))
        self._questions = {
            field.name: Choice(
                instructions=jev_instructions(self.task, field),
                criteria=jev_criteria(field, examples.get(field.name, {})),
            )
            for field in self.task.fields
        }

    def _get_client(self) -> AsyncTypeSafeClient:
        if self._client is None:
            # Retries are off here: the runner retries, so backoff never counts as latency.
            self._client = AsyncTypeSafeClient(
                model=self.model,
                retry=RetryPolicy(max_retries=0),
                timeout=self.params.get("timeout", 30.0),
            )
        return self._client

    async def predict(self, item: Item) -> Prediction:
        if self._questions is None:
            self.fit([])
        client = self._get_client()
        start = time.perf_counter()
        try:
            response = await client.system_one(item.text, self._questions, model=self.model)
        except (TypeSafeAPIConnectionError, TypeSafeAPITimeoutError) as e:
            raise RetryableError(f"{type(e).__name__}: {e}") from e
        except TypeSafeAPIError as e:
            status = getattr(e, "status", None)
            if status == 429 or (status is not None and status >= 500):
                headers = getattr(e, "headers", None) or {}
                retry_after = headers.get("retry-after")
                raise RetryableError(
                    f"HTTP {status}: {e}", float(retry_after) if retry_after and retry_after.isdigit() else None
                ) from e
            return self._invalid(item, f"HTTP {status}: {e}")
        except TypeSafeError as e:
            return self._invalid(item, f"{type(e).__name__}: {e}")
        latency = time.perf_counter() - start

        answers: dict[str, FieldAnswer] = {}
        valid = True
        for field in self.task.fields:
            answer = response.choices.get(field.name)
            if answer is None or answer.choice not in field.labels:
                valid = False
                answers[field.name] = FieldAnswer(label=None)
                continue
            probs = {k: float(v) for k, v in answer.probabilities.items()}
            answers[field.name] = FieldAnswer(
                label=answer.choice,
                probs=probs,
                confidence=probs.get(answer.choice),
                vendor_confidence=float(answer.confidence),
            )

        usage = response.usage
        input_tokens = usage.input_tokens or 0
        output_tokens = usage.output_tokens or 0
        return Prediction(
            arm=self.name,
            model=self.model,
            model_version=response.model,
            item_id=item.id,
            answers=answers,
            gold=item.labels,
            valid=valid,
            error=None if valid else "answer missing or outside the label set",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=pricing.cost_usd(self.model, input_tokens, output_tokens),
            latency_s=latency,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
