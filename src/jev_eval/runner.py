"""Run every arm of a config over its splits and store raw predictions.

Output layout:

    results/<run-id>/config.yaml            the config as run
    results/<run-id>/manifest.json          versions, timings, spend per arm
    results/<run-id>/<split>/<arm>.jsonl    one Prediction per call

Every call is cached in results/.cache.sqlite, keyed on the arm's fingerprint, the
item and the repeat index, so a rerun or a new analysis never pays twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml

from jev_eval import stress
from jev_eval.arms import ARMS, Arm, RetryableError
from jev_eval.config import ROOT, RunConfig
from jev_eval.types import Item, Prediction, TaskSpec, load_items, write_jsonl

MAX_ATTEMPTS = 6
SHUFFLE_SEED = 3


class CallCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, value TEXT, created TEXT)")

    @staticmethod
    def key(arm: Arm, item: Item, repeat: int) -> str:
        raw = json.dumps([arm.fingerprint(), item.id, item.text, repeat])
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Prediction | None:
        row = self.db.execute("SELECT value FROM calls WHERE key = ?", (key,)).fetchone()
        return Prediction.model_validate_json(row[0]) if row else None

    def put(self, key: str, prediction: Prediction) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO calls VALUES (?, ?, ?)",
            (key, prediction.model_dump_json(), datetime.now(UTC).isoformat()),
        )
        self.db.commit()


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, limit_usd: float):
        self.limit = limit_usd
        self.spent = 0.0

    def check(self) -> None:
        if self.spent >= self.limit:
            raise BudgetExceeded(f"spent ${self.spent:.4f} of ${self.limit:.2f}")


def select_items(items: list[Item], limit: int | None) -> list[Item]:
    if limit is None or limit >= len(items):
        return items
    shuffled = sorted(items, key=lambda i: i.id)
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda i: i.id)


async def _call_with_retries(arm: Arm, item: Item) -> Prediction:
    for attempt in range(MAX_ATTEMPTS):
        try:
            return await arm.predict(item)
        except RetryableError as e:
            if attempt == MAX_ATTEMPTS - 1:
                return arm._invalid(item, f"retries exhausted: {e}")
            wait = e.retry_after or min(30.0, 2**attempt) * (1 + random.random() * 0.25)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


async def run_arm(
    arm: Arm,
    items: list[Item],
    repeats: int,
    concurrency: int,
    cache: CallCache,
    budget: Budget,
    progress_label: str = "",
    use_cache: bool = True,
) -> list[Prediction]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Prediction | None] = [None] * (len(items) * repeats)
    done = 0
    total = len(results)
    stopped = False

    async def one(index: int, item: Item, repeat: int) -> None:
        nonlocal done, stopped
        key = cache.key(arm, item, repeat)
        cached = cache.get(key) if use_cache else None
        if cached is not None:
            results[index] = cached.model_copy(update={"cached": True, "arm": arm.name, "repeat": repeat})
        else:
            async with semaphore:
                try:
                    if arm.kind != "baseline":  # the local baseline costs nothing
                        budget.check()
                except BudgetExceeded:
                    stopped = True
                    return
                prediction = await _call_with_retries(arm, item)
            prediction = prediction.model_copy(update={"repeat": repeat})
            budget.spent += prediction.cost_usd
            # Failed calls are not cached, so a rerun tries them again.
            if prediction.valid or not (prediction.error or "").startswith(("retries exhausted", "HTTP 4")):
                cache.put(key, prediction)
            results[index] = prediction
        done += 1
        if done % 50 == 0 or done == total:
            print(f"  {progress_label}{arm.name}: {done}/{total}  new spend ${budget.spent:.4f}", file=sys.stderr)

    jobs = [one(r * len(items) + n, item, r) for r in range(repeats) for n, item in enumerate(items)]
    await asyncio.gather(*jobs)
    if stopped:
        print(f"  {arm.name}: budget of ${budget.limit:.2f} reached, run stopped early", file=sys.stderr)
    return [p for p in results if p is not None]


def build_arms(config: RunConfig, task: TaskSpec) -> list[Arm]:
    arms = []
    for a in config.arms:
        if a.kind not in ARMS:
            raise ValueError(f"Unknown arm kind '{a.kind}'. Known: {sorted(ARMS)}")
        arms.append(ARMS[a.kind](a.name, a.model, task, **a.params))
    return arms


def _package_versions() -> dict[str, str]:
    out = {}
    for pkg in ("jev-eval", "typesafe-sdk", "anthropic", "openai"):
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 - optional packages
            pass
    return out


async def run(config_path: Path, run_id: str | None = None) -> Path:
    config = RunConfig.load(config_path)
    run_id = run_id or f"{config.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))

    base_task = TaskSpec.load(config.task_path())
    train = load_items(config.data_dir() / "train.jsonl")
    cache = CallCache(ROOT / "results" / ".cache.sqlite")
    budget = Budget(config.max_cost_usd)

    split_items: dict[str, list[Item]] = {}
    for split in config.splits:
        items = select_items(load_items(config.data_dir() / f"{split}.jsonl"), config.limit)
        split_items[split], _ = stress.apply(items, base_task, config.stress)
    # Training data keeps its clean text, but only the classes that are in play.
    train, task = stress.apply(train, base_task, config.stress.model_copy(update={"typos": 0.0, "boilerplate": False}))

    arms = build_arms(config, task)
    manifest: dict = {
        "run_id": run_id,
        "config": str(config_path),
        "started": datetime.now(UTC).isoformat(),
        "packages": _package_versions(),
        "task": task.name,
        "n_labels": {f.name: len(f.labels) for f in task.fields},
        "arms": {},
    }
    for arm in arms:
        arm.fit(train)
        spent_before = budget.spent
        started = time.perf_counter()
        versions: set[str] = set()
        try:
            for split, items in split_items.items():
                print(f"[{run_id}] {arm.name} on {split} ({len(items)} items x {config.repeats})", file=sys.stderr)
                predictions = await run_arm(
                    arm,
                    items,
                    config.repeats,
                    config.concurrency,
                    cache,
                    budget,
                    progress_label=f"{split}/",
                    use_cache=config.use_cache,
                )
                write_jsonl(out / split / f"{arm.name}.jsonl", predictions)
                versions |= {p.model_version for p in predictions if p.model_version}
        finally:
            await arm.aclose()
        manifest["arms"][arm.name] = {
            "kind": arm.kind,
            "model": arm.model,
            "params": arm.params,
            "model_versions": sorted(versions),
            "fingerprint": arm.fingerprint(),
            "new_spend_usd": round(budget.spent - spent_before, 6),
            "wall_clock_s": round(time.perf_counter() - started, 2),
        }
    manifest["finished"] = datetime.now(UTC).isoformat()
    manifest["new_spend_usd"] = round(budget.spent, 6)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done: {out}  (new spend ${budget.spent:.4f})", file=sys.stderr)
    return out
