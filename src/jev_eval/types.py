"""Core data types shared by every part of the harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Item(BaseModel):
    """One labelled example. `labels` holds the gold answer per output field."""

    id: str
    text: str
    labels: dict[str, str]
    meta: dict = Field(default_factory=dict)


class FieldSpec(BaseModel):
    """One typed output the arms must predict, e.g. `intent` or `priority`."""

    name: str
    instruction: str
    labels: list[str]
    # Optional one-line description per label. Every arm receives the same text.
    descriptions: dict[str, str] = Field(default_factory=dict)
    # Label that means "none of the classes apply", if the task has one.
    out_of_scope_label: str | None = None


class TaskSpec(BaseModel):
    """What every arm is asked to do. Loaded from `tasks/<name>.yaml`."""

    name: str
    context: str = ""
    fields: list[FieldSpec]

    @classmethod
    def load(cls, path: str | Path) -> TaskSpec:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def field(self, name: str) -> FieldSpec:
        return next(f for f in self.fields if f.name == name)

    def restrict(self, field: str, keep: list[str]) -> TaskSpec:
        """A copy with one field's label set cut down to `keep` (stress test)."""
        spec = self.model_copy(deep=True)
        f = spec.field(field)
        f.labels = [label for label in f.labels if label in keep]
        f.descriptions = {k: v for k, v in f.descriptions.items() if k in keep}
        return spec

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


class FieldAnswer(BaseModel):
    label: str | None
    # Probability per label, when the arm returns a distribution (Jev, baseline).
    probs: dict[str, float] | None = None
    # Estimated probability that `label` is correct. Used for calibration and the
    # cascade: top probability for Jev and the baseline, a self-reported 0-1 score
    # for LLMs.
    confidence: float | None = None
    # The vendor's own confidence statistic, when it differs from `confidence`
    # (Typesafe's normalised confidence, see docs/typesafe-notes.md).
    vendor_confidence: float | None = None


class Prediction(BaseModel):
    """One call of one arm on one item. Stored as a line in `<arm>.jsonl`."""

    arm: str
    model: str
    model_version: str | None = None
    item_id: str
    repeat: int = 0
    answers: dict[str, FieldAnswer] = Field(default_factory=dict)
    gold: dict[str, str] = Field(default_factory=dict)
    valid: bool = True
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False
    raw: str | None = None


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[BaseModel] | list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            data = row.model_dump() if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_items(path: str | Path) -> list[Item]:
    return [Item.model_validate(row) for row in read_jsonl(path)]
