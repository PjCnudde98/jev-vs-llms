"""Turn measured accuracy and cost into the numbers a buyer asks for.

    uv run jev-eval business results/<run> --volumes 10000 100000 1000000 --cost-per-error 2.50

For every arm (cascades included) on the eval split: API cost per month at each volume,
mistakes per month, and, when a cost per mistake is given, the all-in monthly cost. Each
arm is also compared with a reference arm (by default the most accurate LLM):

- `accuracy_points`: accuracy minus the reference's, in points.
- `saving_pct`: how much cheaper the API calls are than the reference's.
- `break_even_cost_per_error`: the price of one mistake at which the two cost the same
  overall. For a cheaper but less accurate arm, it wins whenever a mistake costs *less*
  than this; for a costlier but more accurate arm, whenever a mistake costs *more*.
- `verdict`: `dominates` (cheaper or equal and at least as accurate), `dominated`,
  `cheaper_but_less_accurate` or `costlier_but_more_accurate`.

Cost covers API calls only. The local baseline shows $0 because its hosting is not
measured. A task with several fields counts an item as right only when every field is.
"""

from __future__ import annotations

import json
from pathlib import Path

from jev_eval import metrics as metrics_mod

DEFAULT_VOLUMES = (10_000, 100_000, 1_000_000)


def arms_from_metrics(m: dict, split: str = "eval") -> list[dict]:
    """The numbers this module needs, from `metrics.compute` output."""
    return [
        {
            "name": name,
            "model": a["model"],
            "accuracy": a["task_accuracy"],
            "cost_per_1k": a["cost_per_1k"],
            "p95": a["latency"]["p95"],
        }
        for name, a in m["splits"].get(split, {}).items()
        if a["task_accuracy"] is not None and a["cost_per_1k"] is not None
    ]


def default_reference(arms: list[dict]) -> str | None:
    """The most accurate paid arm that is neither Jev nor a cascade: what a team would use without Jev."""
    candidates = [
        a
        for a in arms
        if not a["name"].startswith("cascade-") and not (a["model"] or "").startswith("jev") and a["cost_per_1k"] > 0
    ]
    return max(candidates, key=lambda a: a["accuracy"])["name"] if candidates else None


def _versus(arm: dict, ref: dict) -> dict:
    saving_per_call = (ref["cost_per_1k"] - arm["cost_per_1k"]) / 1000
    extra_errors = ref["accuracy"] - arm["accuracy"]  # per call; positive means `arm` is worse
    out = {
        "accuracy_points": 100 * (arm["accuracy"] - ref["accuracy"]),
        "saving_per_1k": saving_per_call * 1000,
        "saving_pct": 100 * saving_per_call * 1000 / ref["cost_per_1k"] if ref["cost_per_1k"] else None,
        "break_even_cost_per_error": None,
    }
    if saving_per_call >= 0 and extra_errors <= 0:
        out["verdict"] = "dominates"
    elif saving_per_call <= 0 and extra_errors >= 0:
        out["verdict"] = "dominated"
    else:
        out["verdict"] = "cheaper_but_less_accurate" if saving_per_call > 0 else "costlier_but_more_accurate"
        out["break_even_cost_per_error"] = saving_per_call / extra_errors
    return out


def analyse(
    arms: list[dict],
    volumes: tuple[int, ...] | list[int] = DEFAULT_VOLUMES,
    cost_per_error: float | None = None,
    reference: str | None = None,
) -> dict:
    by_name = {a["name"]: a for a in arms}
    if reference is not None and reference not in by_name:
        raise ValueError(f"Unknown reference arm '{reference}'. Arms in this run: {sorted(by_name)}")
    reference = reference or default_reference(arms)

    rows = []
    for a in arms:
        monthly = {}
        for volume in volumes:
            errors = volume * (1 - a["accuracy"])
            api = volume * a["cost_per_1k"] / 1000
            cell = {"api_usd": api, "errors": errors}
            if cost_per_error is not None:
                cell["error_usd"] = errors * cost_per_error
                cell["total_usd"] = api + errors * cost_per_error
            monthly[str(volume)] = cell
        row = {
            **a,
            "errors_per_1k": 1000 * (1 - a["accuracy"]),
            "monthly": monthly,
            "vs_reference": _versus(a, by_name[reference]) if reference and a["name"] != reference else None,
            "dominated_by": sorted(
                b["name"]
                for b in arms
                if b is not a
                and b["cost_per_1k"] <= a["cost_per_1k"]
                and b["accuracy"] >= a["accuracy"]
                and (b["cost_per_1k"] < a["cost_per_1k"] or b["accuracy"] > a["accuracy"])
            ),
        }
        rows.append(row)
    return {
        "reference": reference,
        "volumes": list(volumes),
        "cost_per_error": cost_per_error,
        "arms": sorted(rows, key=lambda r: -r["accuracy"]),
    }


def _money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 100 else f"${x:,.2f}"


def format_table(result: dict) -> str:
    volumes = result["volumes"]
    cpe = result["cost_per_error"]
    ref = result["reference"]
    head = ["Arm", "Accuracy", "$ / 1k", "Mistakes / 1k"]
    head += [f"API $/month @ {v:,}" for v in volumes]
    if cpe is not None:
        head += [f"All-in $/month @ {volumes[-1]:,}"]
    head += [f"vs {ref}" if ref else "vs reference", "Break-even $ per mistake", "Dominated by"]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in result["arms"]:
        v = r["vs_reference"]
        cells = [r["name"], f"{100 * r['accuracy']:.1f}%", f"${r['cost_per_1k']:.4f}", f"{r['errors_per_1k']:.0f}"]
        cells += [_money(r["monthly"][str(vol)]["api_usd"]) for vol in volumes]
        if cpe is not None:
            cells.append(_money(r["monthly"][str(volumes[-1])]["total_usd"]))
        if v is None:
            cells += ["reference" if ref else "–", "–"]
        else:
            saving = "n/a" if v["saving_pct"] is None else f"{v['saving_pct']:.0f}% cheaper"
            cells += [
                f"{v['accuracy_points']:+.1f} pts, {saving} ({v['verdict'].replace('_', ' ')})",
                "–" if v["break_even_cost_per_error"] is None else f"${v['break_even_cost_per_error']:.2f}",
            ]
        cells.append(", ".join(r["dominated_by"]) or "–")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run(
    run_dir: Path,
    volumes: list[int] | None = None,
    cost_per_error: float | None = None,
    reference: str | None = None,
    split: str = "eval",
) -> dict:
    result = analyse(
        arms_from_metrics(metrics_mod.compute(run_dir), split),
        volumes or DEFAULT_VOLUMES,
        cost_per_error,
        reference,
    )
    (run_dir / "business.json").write_text(json.dumps(result, indent=2))
    return result
