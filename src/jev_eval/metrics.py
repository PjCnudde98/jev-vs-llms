"""Compute every metric from stored predictions. No API calls happen here.

Definitions (see docs/methodology.md for the reasoning):
- Quality uses the first repeat of each item. An invalid answer counts as wrong.
- Accuracy and macro-F1 carry a 95% bootstrap interval (1,000 resamples).
- Calibration uses `confidence` = the arm's estimated probability that its label is
  correct: ECE over 10 equal-width bins, and the Brier score of that estimate.
- Cost per 1k is the mean cost of one call x 1,000; cost per 1k correct divides the
  same spend by correct answers only.
- Latency percentiles are over every valid call, cached ones included (a cached call
  keeps the latency measured when it was made).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support

from jev_eval.config import ROOT, RunConfig
from jev_eval.types import TaskSpec, load_items, read_jsonl

BOOTSTRAP = 1000
BINS = 10
INVALID = "__invalid__"


def _bootstrap(values: np.ndarray, stat, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = [stat(values[rng.integers(0, n, n)]) for _ in range(BOOTSTRAP)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def calibration(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """ECE, Brier and the reliability curve for confidence estimates in [0, 1]."""
    if len(confidence) == 0:
        return {"ece": None, "brier": None, "bins": []}
    edges = np.linspace(0, 1, BINS + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, BINS - 1)
    ece = 0.0
    bins = []
    for b in range(BINS):
        mask = idx == b
        if not mask.any():
            continue
        conf_b, acc_b = float(confidence[mask].mean()), float(correct[mask].mean())
        ece += mask.mean() * abs(conf_b - acc_b)
        bins.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(mask.sum()),
                "confidence": conf_b,
                "accuracy": acc_b,
            }
        )
    return {"ece": float(ece), "brier": float(np.mean((confidence - correct) ** 2)), "bins": bins}


def field_metrics(rows: list[dict], field: str, labels: list[str], oos_label: str | None) -> dict:
    first = [r for r in rows if r.get("repeat", 0) == 0]
    gold = np.array([r["gold"][field] for r in first])
    pred = np.array([(r["answers"].get(field) or {}).get("label") or INVALID for r in first])
    correct = (gold == pred).astype(float)
    gold_labels = sorted(set(gold))

    def macro_f1(idx: np.ndarray) -> float:
        return f1_score(gold[idx], pred[idx], labels=gold_labels, average="macro", zero_division=0)

    index = np.arange(len(first))
    out: dict = {
        "n": len(first),
        "accuracy": float(correct.mean()) if len(first) else None,
        "accuracy_ci": _bootstrap(correct, np.mean),
        "macro_f1": float(macro_f1(index)) if len(first) else None,
        "macro_f1_ci": _bootstrap(index, macro_f1),
    }

    conf = [((r["answers"].get(field) or {}).get("confidence"), c) for r, c in zip(first, correct, strict=True)]
    conf = [(x, c) for x, c in conf if x is not None]
    out["confidence_coverage"] = len(conf) / len(first) if first else 0.0
    out["calibration"] = calibration(np.array([x for x, _ in conf]), np.array([c for _, c in conf]))

    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=gold_labels, zero_division=0)
    out["per_class"] = {
        label: {"precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
        for label, pi, ri, fi, si in zip(gold_labels, p, r, f, s, strict=True)
    }
    if oos_label:
        tp = int(((pred == oos_label) & (gold == oos_label)).sum())
        out["out_of_scope"] = {
            "precision": tp / max(1, int((pred == oos_label).sum())),
            "recall": tp / max(1, int((gold == oos_label).sum())),
        }

    # Consistency: share of items whose repeats all agree.
    by_item: dict[str, set] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        by_item[row["item_id"]].add((row["answers"].get(field) or {}).get("label"))
        counts[row["item_id"]] += 1
    repeated = [i for i, n in counts.items() if n > 1]
    out["consistency"] = (sum(len(by_item[i]) == 1 for i in repeated) / len(repeated)) if repeated else None
    return out


def _correct_by_item(rows: list[dict], field: str) -> dict[str, float]:
    return {
        r["item_id"]: float((r["answers"].get(field) or {}).get("label") == r["gold"][field])
        for r in rows
        if r.get("repeat", 0) == 0
    }


def paired_difference(rows_a: list[dict], rows_b: list[dict], field: str) -> dict:
    """Accuracy of A minus accuracy of B on the items both answered, with a paired test.

    Works for two arms in one run and for the same arm across runs (e.g. Jev in English vs
    Japanese on the paired MASSIVE items). The interval is a paired bootstrap over items;
    the p-value is an exact McNemar test on the items where exactly one of A and B is right.
    """
    a, b = _correct_by_item(rows_a, field), _correct_by_item(rows_b, field)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("The two result sets share no items; paired comparison needs the same item ids.")
    diff = np.array([a[i] - b[i] for i in ids])
    only_a, only_b = int((diff > 0).sum()), int((diff < 0).sum())
    p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {
        "n": len(ids),
        "accuracy_a": float(np.mean([a[i] for i in ids])),
        "accuracy_b": float(np.mean([b[i] for i in ids])),
        "difference": float(diff.mean()),
        "difference_ci": _bootstrap(diff, np.mean),
        "only_a_right": only_a,
        "only_b_right": only_b,
        "p_value": float(p_value),
    }


def arm_metrics(rows: list[dict], task: TaskSpec) -> dict:
    first = [r for r in rows if r.get("repeat", 0) == 0]
    valid_lat = np.array([r["latency_s"] for r in rows if r["valid"] and r["latency_s"] > 0])
    costs = np.array([r["cost_usd"] for r in first])
    fields = {f.name: field_metrics(rows, f.name, f.labels, f.out_of_scope_label) for f in task.fields}
    # An item is "correct" only when every field is right.
    all_correct = sum(
        all((r["answers"].get(f.name) or {}).get("label") == r["gold"][f.name] for f in task.fields) for r in first
    )
    total_cost = float(costs.sum())
    return {
        "model": rows[0]["model"] if rows else None,
        "model_versions": sorted({r["model_version"] for r in rows if r.get("model_version")}),
        "n_calls": len(rows),
        "invalid_rate": float(np.mean([not r["valid"] for r in rows])) if rows else None,
        "errors": sorted({r["error"] for r in rows if r.get("error")})[:10],
        "cost_per_1k": float(costs.mean() * 1000) if len(costs) else None,
        "cost_per_1k_correct": (total_cost / all_correct * 1000) if all_correct else None,
        "mean_input_tokens": float(np.mean([r["input_tokens"] for r in first])) if first else None,
        "mean_output_tokens": float(np.mean([r["output_tokens"] for r in first])) if first else None,
        "latency": {
            q: (float(np.percentile(valid_lat, int(q[1:]))) if len(valid_lat) else None) for q in ("p50", "p95", "p99")
        },
        # Share of items where every field is right; the figure cost per 1k correct is based on.
        "task_accuracy": all_correct / len(first) if first else None,
        "fields": fields,
    }


def error_examples(rows: list[dict], items: dict[str, str], field: str, k: int = 10) -> list[dict]:
    wrong = [
        r for r in rows if r.get("repeat", 0) == 0 and (r["answers"].get(field) or {}).get("label") != r["gold"][field]
    ]
    # Most confident mistakes first: those are the ones a cascade cannot catch.
    wrong.sort(key=lambda r: -((r["answers"].get(field) or {}).get("confidence") or 0))
    return [
        {
            "item_id": r["item_id"],
            "text": items.get(r["item_id"], "")[:400],
            "gold": r["gold"][field],
            "pred": (r["answers"].get(field) or {}).get("label"),
            "confidence": (r["answers"].get(field) or {}).get("confidence"),
            "error": r.get("error"),
        }
        for r in wrong[:k]
    ]


def compute(run_dir: Path) -> dict:
    config = RunConfig.model_validate(yaml.safe_load((run_dir / "config.yaml").read_text()))
    from jev_eval import stress

    _, task = stress.apply([], TaskSpec.load(config.task_path()), config.stress)
    out: dict = {"run": run_dir.name, "dataset": config.dataset, "task": task.model_dump(), "splits": {}}
    for split_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name != "cascade"):
        split = split_dir.name
        source = ROOT / "data" / "processed" / config.dataset / f"{split}.jsonl"
        texts = {i.id: i.text for i in load_items(source)} if source.exists() else {}
        out["splits"][split] = {}
        for path in sorted(split_dir.glob("*.jsonl")):
            rows = read_jsonl(path)
            if not rows:
                continue
            m = arm_metrics(rows, task)
            m["error_examples"] = {f.name: error_examples(rows, texts, f.name) for f in task.fields}
            out["splits"][split][path.stem] = m
    cascade_dir = run_dir / "cascade"
    if cascade_dir.exists():
        out["cascades"] = {p.stem: json.loads(p.read_text()) for p in sorted(cascade_dir.glob("*.json"))}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


def _fmt(x, pct=False, money=False, digits=3):
    if x is None:
        return "–"
    if pct:
        return f"{100 * x:.1f}%"
    if money:
        return f"${x:.4f}" if x < 1 else f"${x:.2f}"
    return f"{x:.{digits}f}"


def summary_table(metrics: dict, split: str = "eval") -> str:
    """A markdown table per field, one row per arm."""
    lines = []
    arms = metrics["splits"].get(split, {})
    for field in [f["name"] for f in metrics["task"]["fields"]]:
        lines.append(f"\n### {metrics['dataset']} / {split} / {field}\n")
        lines.append(
            "| Arm | Accuracy (95% CI) | Macro-F1 | ECE | $ / 1k | $ / 1k correct "
            "| p50 s | p95 s | Invalid | Consistency |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, m in sorted(arms.items(), key=lambda kv: -(kv[1]["fields"][field]["accuracy"] or 0)):
            f = m["fields"][field]
            lo, hi = f["accuracy_ci"]
            lines.append(
                f"| {name} | {_fmt(f['accuracy'], pct=True)} ({_fmt(lo, pct=True)}–{_fmt(hi, pct=True)}) "
                f"| {_fmt(f['macro_f1'])} | {_fmt(f['calibration']['ece'])} | {_fmt(m['cost_per_1k'], money=True)} "
                f"| {_fmt(m['cost_per_1k_correct'], money=True)} | {_fmt(m['latency']['p50'], digits=2)} "
                f"| {_fmt(m['latency']['p95'], digits=2)} | {_fmt(m['invalid_rate'], pct=True)} "
                f"| {_fmt(f['consistency'], pct=True)} |"
            )
    return "\n".join(lines)
