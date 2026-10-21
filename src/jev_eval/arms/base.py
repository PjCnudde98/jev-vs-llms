"""The interface every arm implements.

An arm turns one Item into one Prediction. Arms do not retry: they raise
RetryableError for rate limits, overloads and connection problems, and the runner
retries with backoff. That keeps backoff waits out of the measured latency.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

from jev_eval.types import Item, Prediction, TaskSpec


class RetryableError(Exception):
    """A transient failure (429, 5xx, timeout, connection). The runner retries it."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class Arm(ABC):
    """One system under test: a model plus the way we ask it."""

    #: Short type name used in run configs, e.g. "jev" or "anthropic".
    kind: str = ""

    def __init__(self, name: str, model: str, task: TaskSpec, **params):
        self.name = name
        self.model = model
        self.task = task
        self.params = params

    def fit(self, train: list[Item]) -> None:  # noqa: B027 - optional hook
        """Called once before predictions, with the train split. Used for few-shot
        examples and by the trained baseline."""

    @abstractmethod
    async def predict(self, item: Item) -> Prediction:
        """Return one prediction. Raise RetryableError for transient failures; return a
        Prediction with valid=False for anything else that went wrong."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass

    def fingerprint(self) -> str:
        """Identifies everything that changes the answer, so cached calls are only
        reused when the setup is identical."""
        payload = json.dumps(
            {"kind": self.kind, "model": self.model, "params": self.params, "task": self.task.fingerprint()},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _invalid(self, item: Item, error: str, **extra) -> Prediction:
        return Prediction(
            arm=self.name, model=self.model, item_id=item.id, gold=item.labels, valid=False, error=error, **extra
        )
