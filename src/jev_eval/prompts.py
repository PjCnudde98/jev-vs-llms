"""Build what each arm sees from one TaskSpec, so every arm gets the same information.

LLM arms get a system prompt plus a JSON schema; Jev gets the same instruction,
label names and descriptions as Choice questions. Few-shot examples, when enabled,
come from the train split and are the same items for every arm.
"""

from __future__ import annotations

import random
from collections import defaultdict

from jev_eval.types import FieldSpec, Item, TaskSpec

EXAMPLE_SEED = 7


def select_examples(train: list[Item], task: TaskSpec, per_label: int) -> dict[str, dict[str, list[str]]]:
    """Pick `per_label` training texts per label per field: {field: {label: [text]}}."""
    if per_label <= 0:
        return {}
    rng = random.Random(EXAMPLE_SEED)
    out: dict[str, dict[str, list[str]]] = {}
    for field in task.fields:
        by_label: dict[str, list[str]] = defaultdict(list)
        for item in sorted(train, key=lambda i: i.id):
            by_label[item.labels[field.name]].append(item.text)
        out[field.name] = {
            label: rng.sample(by_label[label], min(per_label, len(by_label[label])))
            for label in field.labels
            if by_label.get(label)
        }
    return out


def _label_line(field: FieldSpec, label: str) -> str:
    description = field.descriptions.get(label)
    return f"- {label}: {description}" if description else f"- {label}"


def llm_system_prompt(task: TaskSpec, examples: dict[str, dict[str, list[str]]]) -> str:
    parts = []
    if task.context:
        parts.append(task.context)
    parts.append("Classify the input. For each field below, choose exactly one label from that field's list.")
    for field in task.fields:
        lines = [f"Field `{field.name}`: {field.instruction}", "Labels:"]
        lines += [_label_line(field, label) for label in field.labels]
        field_examples = examples.get(field.name, {})
        if field_examples:
            lines.append("Examples:")
            for label, texts in field_examples.items():
                lines += [f'- "{t}" -> {label}' for t in texts]
        parts.append("\n".join(lines))
    parts.append(
        "For each field, also return `confidence`: your probability, from 0 to 1, that the label you chose is correct."
    )
    return "\n\n".join(parts)


def llm_user_message(item: Item) -> str:
    return f"Input:\n{item.text}"


def llm_output_schema(task: TaskSpec) -> dict:
    """JSON schema for structured output: {field: {label, confidence}}."""
    properties = {
        field.name: {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": field.labels},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        }
        for field in task.fields
    }
    return {
        "type": "object",
        "properties": properties,
        "required": [f.name for f in task.fields],
        "additionalProperties": False,
    }


def jev_criteria(field: FieldSpec, examples: dict[str, list[str]]) -> dict[str, str | dict | None]:
    """Choice criteria for Jev: the label description, plus examples when enabled.

    Typesafe's docs recommend an object per option when an option needs several kinds
    of guidance; the keys are free-form labels the model reads.
    """
    criteria: dict[str, str | dict | None] = {}
    for label in field.labels:
        description = field.descriptions.get(label)
        label_examples = examples.get(label)
        if label_examples:
            criteria[label] = {"description": description or label, "examples": label_examples}
        else:
            criteria[label] = description
    return criteria


def jev_instructions(task: TaskSpec, field: FieldSpec) -> str:
    return f"{task.context} {field.instruction}".strip()
