"""Download public benchmarks and freeze the train / calibration / eval splits.

Every dataset ends up in the same shape:

    data/processed/<dataset>/train.jsonl        few-shot examples, baseline training
    data/processed/<dataset>/calibration.jsonl  cascade thresholds and prompt tuning only
    data/processed/<dataset>/eval.jsonl         the frozen test set, never tuned on
    tasks/<dataset>.yaml                        the label set every arm receives

Splits are drawn with a fixed seed and stratified by label, so a rerun gives the
same files byte for byte.
"""

from __future__ import annotations

import csv
import io
import json
import random
import tarfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from sklearn.model_selection import train_test_split

from jev_eval.types import Item, write_jsonl

SEED = 20261021
EVAL_SIZE = 1000
CALIBRATION_SIZE = 500

CLINC_PARQUET = "https://huggingface.co/api/datasets/clinc/clinc_oos/parquet/plus/{split}/0.parquet"
CLINC_INFO = "https://datasets-server.huggingface.co/info?dataset=clinc/clinc_oos&config=plus"
BANKING_CSV = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{split}.csv"
MASSIVE_TARBALL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_LANGUAGES = {"en": "en-US", "fr": "fr-FR", "nl": "nl-NL", "ja": "ja-JP"}
MASSIVE_NAMES = {"en": "English", "fr": "French", "nl": "Dutch", "ja": "Japanese"}
MASSIVE_EVAL_SIZE = 300
MASSIVE_CALIBRATION_SIZE = 150


def _download(url: str, dest: Path) -> bytes:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())
    return dest.read_bytes()


def _humanize(label: str) -> str:
    return label.replace("_", " ").capitalize()


def _stratified_sample(items: list[Item], field: str, n: int) -> tuple[list[Item], list[Item]]:
    """Split `items` into (sample of size n, rest), stratified on `field`."""
    if n >= len(items):
        return items, []
    rest, sample = train_test_split(items, test_size=n, random_state=SEED, stratify=[i.labels[field] for i in items])
    return sorted(sample, key=lambda i: i.id), sorted(rest, key=lambda i: i.id)


def _balanced_sample(items: list[Item], field: str, n: int) -> list[Item]:
    """About n items spread evenly over the labels (seeded round robin).

    Unlike `_stratified_sample` this tolerates labels with a single item, which MASSIVE
    has in its dev and test partitions (e.g. `general_greet`).
    """
    rng = random.Random(SEED)
    pools: dict[str, list[Item]] = {}
    for item in sorted(items, key=lambda i: i.id):
        pools.setdefault(item.labels[field], []).append(item)
    for pool in pools.values():
        rng.shuffle(pool)
    sample: list[Item] = []
    while len(sample) < n and any(pools.values()):
        for label in sorted(pools):
            if pools[label] and len(sample) < n:
                sample.append(pools[label].pop())
    return sorted(sample, key=lambda i: i.id)


def _write(dataset: str, root: Path, splits: dict[str, list[Item]], task: dict) -> None:
    out = root / "data" / "processed" / dataset
    for name, items in splits.items():
        write_jsonl(out / f"{name}.jsonl", items)
    task_path = root / "tasks" / f"{dataset}.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(yaml.safe_dump(task, sort_keys=False, allow_unicode=True))
    sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    print(f"{dataset}: {sizes} -> {out}")


def prepare_clinc150(root: Path) -> None:
    raw = root / "data" / "raw" / "clinc150"
    info = json.loads(_download(CLINC_INFO, raw / "info.json"))
    names = info["dataset_info"]["features"]["intent"]["names"]

    def load(split: str) -> list[Item]:
        table = pq.read_table(io.BytesIO(_download(CLINC_PARQUET.format(split=split), raw / f"{split}.parquet")))
        rows = table.to_pylist()
        return [
            Item(id=f"clinc-{split}-{i}", text=r["text"], labels={"intent": names[r["intent"]]})
            for i, r in enumerate(rows)
        ]

    eval_items, _ = _stratified_sample(load("test"), "intent", EVAL_SIZE)
    calibration, _ = _stratified_sample(load("validation"), "intent", CALIBRATION_SIZE)
    labels = sorted(names)
    descriptions = {n: _humanize(n) for n in labels}
    descriptions["oos"] = "Out of scope: the request matches none of the other intents"
    task = {
        "name": "clinc150",
        "context": "Messages a user sent to a general-purpose virtual assistant.",
        "fields": [
            {
                "name": "intent",
                "instruction": "Which intent does the user's message express?",
                "labels": labels,
                "descriptions": descriptions,
                "out_of_scope_label": "oos",
            }
        ],
    }
    _write("clinc150", root, {"train": load("train"), "calibration": calibration, "eval": eval_items}, task)


def prepare_banking77(root: Path) -> None:
    raw = root / "data" / "raw" / "banking77"

    def load(split: str) -> list[Item]:
        text = _download(BANKING_CSV.format(split=split), raw / f"{split}.csv").decode()
        rows = list(csv.DictReader(io.StringIO(text)))
        return [
            Item(id=f"banking-{split}-{i}", text=r["text"], labels={"intent": r["category"]})
            for i, r in enumerate(rows)
        ]

    eval_items, _ = _stratified_sample(load("test"), "intent", EVAL_SIZE)
    # Banking77 has no validation split, so the calibration set comes out of train.
    calibration, train = _stratified_sample(load("train"), "intent", CALIBRATION_SIZE)
    labels = sorted({i.labels["intent"] for i in train})
    task = {
        "name": "banking77",
        "context": "Messages a customer sent to an online bank's support chat.",
        "fields": [
            {
                "name": "intent",
                "instruction": "Which intent does the customer's message express?",
                "labels": labels,
                "descriptions": {n: _humanize(n) for n in labels},
            }
        ],
    }
    _write("banking77", root, {"train": train, "calibration": calibration, "eval": eval_items}, task)


def prepare_massive(root: Path, languages: dict[str, str] | None = None) -> None:
    """Multilingual intent detection from Amazon MASSIVE (human-translated, CC BY 4.0).

    The same utterances exist in every locale, so the eval and calibration ids are drawn
    once, spread evenly over the intents, from English and reused for every language. English is the
    control: any gap between languages is a language effect, not a different item mix.
    Labels and descriptions stay in English; only the user's text changes language.
    """
    languages = languages or MASSIVE_LANGUAGES
    raw = root / "data" / "raw" / "massive"
    archive = raw / "massive-1.1.tar.gz"
    _download(MASSIVE_TARBALL, archive)

    def load(locale: str) -> list[dict]:
        with tarfile.open(archive) as tar:
            member = tar.extractfile(f"1.1/data/{locale}.jsonl")
            assert member is not None, f"{locale} not in {archive}"
            return [json.loads(line) for line in member]

    by_locale = {code: {r["id"]: r for r in load(locale)} for code, locale in languages.items()}
    english = by_locale["en"]

    def items(code: str, partition: str) -> list[Item]:
        return [
            Item(
                id=f"massive-{r['id']}",
                text=r["utt"],
                labels={"intent": r["intent"]},
                meta={"language": code, "massive_id": r["id"]},
            )
            for r in by_locale[code].values()
            if r["partition"] == partition
        ]

    eval_ids = {i.id for i in _balanced_sample(items("en", "test"), "intent", MASSIVE_EVAL_SIZE)}
    calibration_ids = {i.id for i in _balanced_sample(items("en", "dev"), "intent", MASSIVE_CALIBRATION_SIZE)}
    labels = sorted({r["intent"] for r in english.values()})
    descriptions = {n: n.replace("_", ": ", 1).replace("_", " ").capitalize() for n in labels}

    for code in languages:
        by_id = {i.id: i for p in ("train", "dev", "test") for i in items(code, p)}
        task = {
            "name": f"massive_{code}",
            "context": f"Requests a user spoke or typed to a voice assistant, written in {MASSIVE_NAMES[code]}.",
            "fields": [
                {
                    "name": "intent",
                    "instruction": "Which intent does the user's request express?",
                    "labels": labels,
                    "descriptions": descriptions,
                }
            ],
        }
        splits = {
            "train": items(code, "train"),
            "calibration": [by_id[i] for i in sorted(calibration_ids)],
            "eval": [by_id[i] for i in sorted(eval_ids)],
        }
        _write(f"massive_{code}", root, splits, task)


def prepare_custom(root: Path, name: str, source: Path, task_path: Path) -> None:
    """Freeze splits for an in-house set (e.g. ticket triage).

    `source` is a JSONL file of `{"id", "text", "labels": {field: label}}` rows, and
    `task_path` a task YAML written by hand (see tasks/triage.example.yaml). The first
    field of the task is used to stratify.
    """
    from jev_eval.types import TaskSpec, load_items

    task = TaskSpec.load(task_path)
    items = load_items(source)
    field = task.fields[0].name
    n_eval = min(EVAL_SIZE, len(items) // 2)
    n_cal = min(CALIBRATION_SIZE, len(items) // 4)
    eval_items, rest = _stratified_sample(items, field, n_eval)
    calibration, train = _stratified_sample(rest, field, n_cal)
    _write(name, root, {"train": train, "calibration": calibration, "eval": eval_items}, task.model_dump())


PUBLIC: dict[str, Callable[[Path], None]] = {
    "clinc150": prepare_clinc150,
    "banking77": prepare_banking77,
    "massive": prepare_massive,
}
