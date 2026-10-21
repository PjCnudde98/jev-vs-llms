"""Run configuration: one YAML file per experiment, copied into its results folder."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class ArmConfig(BaseModel):
    name: str
    kind: str
    model: str
    params: dict = Field(default_factory=dict)


class StressConfig(BaseModel):
    # Keep only this many classes (seeded pick; the out-of-scope label is always kept).
    max_classes: int | None = None
    # Share of characters hit by a typo (swap, drop, double), e.g. 0.05.
    typos: float = 0.0
    # Wrap each input in an email greeting, signature, disclaimer and quoted thread.
    boilerplate: bool = False


class RunConfig(BaseModel):
    name: str
    dataset: str
    splits: list[str] = Field(default_factory=lambda: ["eval"])
    # Cap on items per split, after a seeded shuffle. None = the whole split.
    limit: int | None = None
    # Calls per item. >1 measures consistency (same input, same answer?).
    repeats: int = 1
    # Parallel calls per arm. Use 1 for a clean latency measurement.
    concurrency: int = 8
    # Reuse cached calls. Turn off for latency runs, where every call must be fresh.
    use_cache: bool = True
    # Stop starting new calls once this much new money is spent in this run.
    max_cost_usd: float = 10.0
    stress: StressConfig = Field(default_factory=StressConfig)
    arms: list[ArmConfig]

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def data_dir(self) -> Path:
        return ROOT / "data" / "processed" / self.dataset

    def task_path(self) -> Path:
        return ROOT / "tasks" / f"{self.dataset}.yaml"
