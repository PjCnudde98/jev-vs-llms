"""Arm D, the cascade: Jev first, and only low-confidence cases go to an LLM.

Simulated offline from stored predictions, so it costs nothing extra: both arms
must have run on the calibration and eval splits. For each item, the cascade keeps
the primary answer when its confidence is at or above the threshold, and otherwise
uses the fallback answer and pays for both calls.

Thresholds are chosen on the calibration split only, then reported on eval:
- "match_fallback": the cheapest threshold whose accuracy is within `tolerance` of
  the fallback alone.
- "within_2x_cost": the most accurate threshold that costs at most 2x the primary.
The chosen points are also written as `<split>/<name>@<point>.jsonl`, so metrics and
the report treat them like any other arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jev_eval.types import read_jsonl, write_jsonl

THRESHOLDS = [round(float(t), 3) for t in np.linspace(0, 1, 101)]


def _by_item(rows: list[dict]) -> dict[str, dict]:
    return {r["item_id"]: r for r in rows if r.get("repeat", 0) == 0}


def _min_confidence(row: dict, fields: list[str]) -> float:
    if not row["valid"]:
        return -1.0
    values = [(row["answers"].get(f) or {}).get("confidence") for f in fields]
    return min((v if v is not None else -1.0) for v in values)


def _correct(row: dict, fields: list[str]) -> bool:
    return all((row["answers"].get(f) or {}).get("label") == row["gold"][f] for f in fields)


def simulate(primary: list[dict], fallback: list[dict], fields: list[str], threshold: float) -> dict:
    p, f = _by_item(primary), _by_item(fallback)
    ids = sorted(set(p) & set(f))
    routed = correct = 0
    cost = []
    latency = []
    for i in ids:
        conf = _min_confidence(p[i], fields)
        if conf < threshold:
            routed += 1
            correct += _correct(f[i], fields)
            cost.append(p[i]["cost_usd"] + f[i]["cost_usd"])
            latency.append(p[i]["latency_s"] + f[i]["latency_s"])
        else:
            correct += _correct(p[i], fields)
            cost.append(p[i]["cost_usd"])
            latency.append(p[i]["latency_s"])
    n = len(ids)
    return {
        "threshold": threshold,
        "n": n,
        "accuracy": correct / n if n else None,
        "routed_share": routed / n if n else None,
        "cost_per_1k": float(np.mean(cost) * 1000) if n else None,
        "latency_p50": float(np.percentile(latency, 50)) if n else None,
        "latency_p95": float(np.percentile(latency, 95)) if n else None,
    }


def merged_predictions(
    primary: list[dict], fallback: list[dict], fields: list[str], threshold: float, name: str
) -> list[dict]:
    p, f = _by_item(primary), _by_item(fallback)
    rows = []
    for i in sorted(set(p) & set(f)):
        routed = bool(_min_confidence(p[i], fields) < threshold)
        base = dict(f[i] if routed else p[i])
        base.update(
            arm=name,
            model=f"{p[i]['model']} -> {f[i]['model']}",
            cost_usd=p[i]["cost_usd"] + (f[i]["cost_usd"] if routed else 0.0),
            latency_s=p[i]["latency_s"] + (f[i]["latency_s"] if routed else 0.0),
            input_tokens=p[i]["input_tokens"] + (f[i]["input_tokens"] if routed else 0),
            output_tokens=p[i]["output_tokens"] + (f[i]["output_tokens"] if routed else 0),
            raw=json.dumps({"routed": routed}),
        )
        rows.append(base)
    return rows


def run(
    run_dir: Path,
    primary: str,
    fallback: str,
    fields: list[str],
    tolerance: float = 0.005,
    tune_split: str = "calibration",
    eval_split: str = "eval",
) -> dict:
    def load(split: str, arm: str) -> list[dict]:
        path = run_dir / split / f"{arm}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run both arms on the {split} split first.")
        return read_jsonl(path)

    tune_p, tune_f = load(tune_split, primary), load(tune_split, fallback)
    eval_p, eval_f = load(eval_split, primary), load(eval_split, fallback)

    sweep_tune = [simulate(tune_p, tune_f, fields, t) for t in THRESHOLDS]
    sweep_eval = [simulate(eval_p, eval_f, fields, t) for t in THRESHOLDS]
    # Threshold 0 never routes a valid answer; 1.01 routes everything.
    primary_only = sweep_tune[0]
    fallback_only = simulate(tune_p, tune_f, fields, 1.01)

    points: dict[str, float] = {}
    target = fallback_only["accuracy"] - tolerance
    ok = [s for s in sweep_tune if s["accuracy"] >= target]
    if ok:
        points["match_fallback"] = min(ok, key=lambda s: s["cost_per_1k"])["threshold"]
    budget = 2 * primary_only["cost_per_1k"]
    cheap = [s for s in sweep_tune if s["cost_per_1k"] <= budget]
    if cheap:
        points["within_2x_cost"] = max(cheap, key=lambda s: (s["accuracy"], -s["cost_per_1k"]))["threshold"]

    name = f"cascade-{primary}-to-{fallback}"
    chosen = {}
    for point, t in points.items():
        chosen[point] = {
            "threshold": t,
            tune_split: simulate(tune_p, tune_f, fields, t),
            eval_split: simulate(eval_p, eval_f, fields, t),
        }
        write_jsonl(
            run_dir / eval_split / f"{name}@{point}.jsonl",
            merged_predictions(eval_p, eval_f, fields, t, f"{name}@{point}"),
        )

    result = {
        "primary": primary,
        "fallback": fallback,
        "fields": fields,
        "tolerance": tolerance,
        "reference": {
            "primary_only": simulate(eval_p, eval_f, fields, 0.0),
            "fallback_only": simulate(eval_p, eval_f, fields, 1.01),
        },
        "operating_points": chosen,
        "sweep": {tune_split: sweep_tune, eval_split: sweep_eval},
    }
    out = run_dir / "cascade" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
