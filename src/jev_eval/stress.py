"""Stress transforms: change one thing about the input and keep the rest fixed.

Each transform is seeded per item, so the stressed set is the same on every rerun
and for every arm. Unseen-class tests are not automated yet; see docs/methodology.md.
Language is not a transform: see the MASSIVE datasets.
"""

from __future__ import annotations

import random

from jev_eval.config import StressConfig
from jev_eval.types import Item, TaskSpec

SEED = 11

_GREETINGS = ["Hi team,", "Hello,", "Good morning,", "Dear support,", "Hey there,"]
_SIGNATURE = (
    "\n\nKind regards,\nJordan Peeters\nOperations Lead | Example Logistics NV\n"
    "Tel: +32 9 123 45 67 | www.example.com\n\n"
    "This e-mail and any attachments are confidential and intended solely for the addressee. "
    "If you received this message in error, please notify the sender and delete it."
)
_QUOTED = (
    "\n\n-----Original Message-----\nFrom: Support <support@example.com>\nSent: Monday 09:12\n"
    "Subject: RE: your request\n\n> Thanks for reaching out. Could you share a bit more detail?\n"
    "> We will get back to you as soon as possible.\n> Best, the support team"
)


def _typos(text: str, rate: float, rng: random.Random) -> str:
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.choice(["drop", "double", "swap"])
            if op == "double":
                out += [c, c]
            elif op == "swap" and i + 1 < len(chars):
                out += [chars[i + 1], c]
                i += 1
            # "drop" adds nothing
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _boilerplate(text: str, rng: random.Random) -> str:
    return f"{rng.choice(_GREETINGS)}\n\n{text}{_SIGNATURE}{_QUOTED}"


def apply(items: list[Item], task: TaskSpec, cfg: StressConfig) -> tuple[list[Item], TaskSpec]:
    if cfg.max_classes:
        field = task.fields[0]
        rng = random.Random(SEED)
        oos = field.out_of_scope_label
        pool = [label for label in field.labels if label != oos]
        keep = sorted(rng.sample(pool, min(cfg.max_classes, len(pool))))
        if oos:
            keep.append(oos)
        task = task.restrict(field.name, keep)
        items = [i for i in items if i.labels[field.name] in keep]

    if cfg.typos or cfg.boilerplate:
        stressed = []
        for item in items:
            rng = random.Random(f"{SEED}-{item.id}")
            text = item.text
            if cfg.typos:
                text = _typos(text, cfg.typos, rng)
            if cfg.boilerplate:
                text = _boilerplate(text, rng)
            stressed.append(item.model_copy(update={"text": text}))
        items = stressed
    return items, task
