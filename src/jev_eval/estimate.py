"""Rough cost estimate for a run config, before spending anything.

Tokens are approximated as characters / 4, and LLM output (including thinking) as a
flat `assumed_output_tokens` per call. Prompt caching is ignored, so LLM numbers are
on the high side. Run a 20-item smoke config first for real per-call costs.
"""

from __future__ import annotations

import json

from jev_eval import pricing, stress
from jev_eval.config import RunConfig
from jev_eval.prompts import (
    jev_criteria,
    jev_instructions,
    llm_output_schema,
    llm_system_prompt,
    llm_user_message,
    select_examples,
)
from jev_eval.runner import select_items
from jev_eval.types import TaskSpec, load_items


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate(config: RunConfig, assumed_output_tokens: int = 150) -> list[dict]:
    base_task = TaskSpec.load(config.task_path())
    train = load_items(config.data_dir() / "train.jsonl")
    _, task = stress.apply([], base_task, config.stress)
    items = []
    for split in config.splits:
        chosen = select_items(load_items(config.data_dir() / f"{split}.jsonl"), config.limit)
        items += stress.apply(chosen, base_task, config.stress)[0]
    calls = len(items) * config.repeats
    mean_input = sum(_tokens(i.text) for i in items) / max(1, len(items))

    rows = []
    for arm in config.arms:
        examples = select_examples(train, task, arm.params.get("examples_per_label", 0))
        if arm.kind == "jev":
            overhead = sum(
                _tokens(jev_instructions(task, f)) + _tokens(json.dumps(jev_criteria(f, examples.get(f.name, {}))))
                for f in task.fields
            )
            in_tok, out_tok = overhead + mean_input, 0
        elif arm.kind in ("anthropic", "openai"):
            overhead = _tokens(llm_system_prompt(task, examples)) + _tokens(json.dumps(llm_output_schema(task)))
            in_tok = (
                overhead
                + mean_input
                + _tokens(llm_user_message(items[0]) if items else "")
                - _tokens(items[0].text if items else "")
            )
            out_tok = assumed_output_tokens
        else:
            in_tok, out_tok = 0, 0
        try:
            per_call = pricing.cost_usd(arm.model, int(in_tok), int(out_tok))
            note = ""
        except pricing.MissingPriceError:
            per_call, note = float("nan"), "no price in configs/pricing.yaml"
        rows.append(
            {
                "arm": arm.name,
                "model": arm.model,
                "calls": calls,
                "input_tokens_per_call": int(in_tok),
                "output_tokens_per_call": int(out_tok),
                "cost_per_1k": per_call * 1000,
                "total_usd": per_call * calls,
                "note": note,
            }
        )
    return rows
