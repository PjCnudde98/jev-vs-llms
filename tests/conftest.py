"""Shared test helpers: a tiny task, items, and a fake arm that needs no API."""

from __future__ import annotations

import hashlib

import pytest

from jev_eval.arms import ARMS
from jev_eval.arms.base import Arm, RetryableError
from jev_eval.types import FieldAnswer, FieldSpec, Item, Prediction, TaskSpec

LABELS = ["billing", "shipping", "returns", "oos"]


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec(
        name="toy",
        context="Messages sent to a web shop.",
        fields=[
            FieldSpec(
                name="intent",
                instruction="Which team should handle this?",
                labels=LABELS,
                descriptions={"billing": "Charges and invoices", "oos": "None of the above"},
                out_of_scope_label="oos",
            )
        ],
    )


@pytest.fixture
def items() -> list[Item]:
    return [Item(id=f"i{n:03d}", text=f"message {n}", labels={"intent": LABELS[n % 4]}) for n in range(40)]


def _unit(*parts) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class FakeArm(Arm):
    """Right with probability `accuracy`; confidence is high when right, low when wrong."""

    kind = "fake"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.fail_first = self.params.get("fail_first", 0)

    async def predict(self, item: Item) -> Prediction:
        self.calls += 1
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RetryableError("simulated 429", retry_after=0.0)
        gold = item.labels["intent"]
        right = _unit(self.name, item.id) < self.params.get("accuracy", 0.8)
        label = gold if right else next(lab for lab in LABELS if lab != gold)
        conf = 0.95 if right else 0.4
        return Prediction(
            arm=self.name,
            model=self.model,
            model_version=self.model,
            item_id=item.id,
            gold=item.labels,
            answers={"intent": FieldAnswer(label=label, confidence=conf)},
            cost_usd=self.params.get("cost", 0.001),
            latency_s=self.params.get("latency", 0.1),
        )


@pytest.fixture(autouse=True)
def register_fake():
    ARMS["fake"] = FakeArm
    yield
    ARMS.pop("fake", None)
