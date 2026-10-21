"""End to end without any API: run fake arms, simulate a cascade, compute metrics, build the report."""

import asyncio
import json

import pytest
import yaml

from jev_eval import cascade, config, metrics, report, runner
from jev_eval.runner import Budget, CallCache, run_arm
from jev_eval.types import write_jsonl
from tests.conftest import FakeArm


@pytest.fixture
def project(tmp_path, monkeypatch, task, items):
    for module in (config, runner, metrics):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "toy.yaml").write_text(yaml.safe_dump(task.model_dump()))
    data = tmp_path / "data" / "processed" / "toy"
    write_jsonl(data / "train.jsonl", items)
    write_jsonl(data / "calibration.jsonl", items[:20])
    write_jsonl(data / "eval.jsonl", items[20:])
    cfg = {
        "name": "toy",
        "dataset": "toy",
        "splits": ["calibration", "eval"],
        "max_cost_usd": 5.0,
        "arms": [
            {
                "name": "jev",
                "kind": "fake",
                "model": "fake-small",
                "params": {"accuracy": 0.7, "cost": 0.0001, "latency": 0.1},
            },
            {
                "name": "opus",
                "kind": "fake",
                "model": "fake-large",
                "params": {"accuracy": 0.95, "cost": 0.01, "latency": 2.0},
            },
        ],
    }
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return tmp_path, path


def test_full_pipeline(project):
    root, cfg_path = project
    out = asyncio.run(runner.run(cfg_path, "toy-run"))
    assert (out / "eval" / "jev.jsonl").exists() and (out / "manifest.json").exists()

    result = cascade.run(out, "jev", "opus", ["intent"])
    ref = result["reference"]
    assert ref["primary_only"]["routed_share"] == 0.0
    assert ref["fallback_only"]["routed_share"] == 1.0
    # The fake arm is unsure exactly when wrong, so a cascade should beat Jev alone.
    best = max(p["eval"]["accuracy"] for p in result["operating_points"].values())
    assert best > ref["primary_only"]["accuracy"]

    m = metrics.compute(out)
    assert "cascade-jev-to-opus@match_fallback" in m["splits"]["eval"]
    assert m["splits"]["eval"]["opus"]["cost_per_1k"] == pytest.approx(10.0)

    page = report.build([out], root / "report.html")
    text = page.read_text()
    assert "Head-to-head" in text and "Cascade" in text
    data = json.loads(text.split('id="report-data">')[1].split("</script>")[0])
    assert {a["name"] for a in data["runs"][0]["arms"]} >= {"jev", "opus"}


def test_second_run_is_served_from_cache(project):
    _, cfg_path = project
    asyncio.run(runner.run(cfg_path, "first"))
    asyncio.run(runner.run(cfg_path, "second"))
    manifest = json.loads((config.ROOT / "results" / "second" / "manifest.json").read_text())
    assert manifest["new_spend_usd"] == 0


def test_budget_stops_the_run(task, items, tmp_path):
    arm = FakeArm("pricey", "fake", task, cost=1.0)
    budget = Budget(2.5)
    predictions = asyncio.run(run_arm(arm, items, 1, 1, CallCache(tmp_path / "c.sqlite"), budget))
    assert len(predictions) == 3
    assert budget.spent == 3.0


def test_retries_do_not_count_as_latency(task, items, tmp_path):
    arm = FakeArm("flaky", "fake", task, fail_first=2, latency=0.1)
    predictions = asyncio.run(run_arm(arm, items[:1], 1, 1, CallCache(tmp_path / "c.sqlite"), Budget(1.0)))
    assert predictions[0].valid and predictions[0].latency_s == 0.1
    assert arm.calls == 3
