"""Command line: `uv run jev-eval <command>`. See README.md for the full workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from jev_eval.config import ROOT


def _cmd_prepare(args) -> None:
    from jev_eval import datasets

    if args.dataset in datasets.PUBLIC:
        datasets.PUBLIC[args.dataset](ROOT)
    elif args.source and args.task:
        datasets.prepare_custom(ROOT, args.dataset, Path(args.source), Path(args.task))
    else:
        sys.exit(
            f"Unknown dataset '{args.dataset}'. Public: {sorted(datasets.PUBLIC)}; "
            "for your own set pass --source items.jsonl --task tasks/<name>.yaml"
        )


def _cmd_estimate(args) -> None:
    from jev_eval.config import RunConfig
    from jev_eval.estimate import estimate

    rows = estimate(RunConfig.load(args.config), args.output_tokens)
    print("| Arm | Model | Calls | In tok/call | Out tok/call | $ / 1k | Total $ | Note |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(
            f"| {r['arm']} | {r['model']} | {r['calls']} | {r['input_tokens_per_call']} "
            f"| {r['output_tokens_per_call']} | {r['cost_per_1k']:.4f} | {r['total_usd']:.4f} | {r['note']} |"
        )
    total = sum(r["total_usd"] for r in rows if r["total_usd"] == r["total_usd"])
    print(
        f"\nEstimated total: ${total:.2f} (rough: chars/4 tokens, no prompt caching). "
        "Run a smoke config for real per-call costs."
    )


def _cmd_run(args) -> None:
    from jev_eval import runner

    out = asyncio.run(runner.run(Path(args.config), args.run_id))
    if not args.no_metrics:
        _print_metrics(out)


def _print_metrics(run_dir: Path) -> None:
    from jev_eval import metrics

    m = metrics.compute(run_dir)
    for split in m["splits"]:
        print(metrics.summary_table(m, split))


def _cmd_metrics(args) -> None:
    _print_metrics(Path(args.run_dir))


def _cmd_cascade(args) -> None:
    import yaml

    from jev_eval import cascade
    from jev_eval.types import TaskSpec

    run_dir = Path(args.run_dir)
    dataset = yaml.safe_load((run_dir / "config.yaml").read_text())["dataset"]
    fields = args.fields or [f.name for f in TaskSpec.load(ROOT / "tasks" / f"{dataset}.yaml").fields]
    result = cascade.run(run_dir, args.primary, args.fallback, fields, args.tolerance)
    print(json.dumps({"reference": result["reference"], "operating_points": result["operating_points"]}, indent=2))
    _print_metrics(run_dir)


def _cmd_report(args) -> None:
    from jev_eval import report

    out = report.build([Path(d) for d in args.run_dirs], Path(args.output), args.title)
    print(f"Report written to {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jev-eval", description="Benchmark Jev against LLMs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="download a dataset and freeze its splits")
    p.add_argument("dataset", help="clinc150, banking77, or a name for your own set")
    p.add_argument("--source", help="JSONL of your own labelled items")
    p.add_argument("--task", help="task YAML for your own set")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("estimate", help="rough cost of a run config before running it")
    p.add_argument("config")
    p.add_argument("--output-tokens", type=int, default=150, help="assumed LLM output tokens per call")
    p.set_defaults(func=_cmd_estimate)

    p = sub.add_parser("run", help="run every arm of a config (spends money)")
    p.add_argument("config")
    p.add_argument("--run-id", help="name of the results folder (default: <name>-<timestamp>)")
    p.add_argument("--no-metrics", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("metrics", help="recompute metrics for a results folder (free)")
    p.add_argument("run_dir")
    p.set_defaults(func=_cmd_metrics)

    p = sub.add_parser("cascade", help="simulate Jev-first cascades from stored predictions (free)")
    p.add_argument("run_dir")
    p.add_argument("--primary", required=True, help="arm name answering first, e.g. jev")
    p.add_argument("--fallback", required=True, help="arm name for low-confidence cases, e.g. opus-5.5")
    p.add_argument("--fields", nargs="*", help="fields whose confidence decides routing (default: all)")
    p.add_argument("--tolerance", type=float, default=0.005, help="accuracy gap allowed for 'match_fallback'")
    p.set_defaults(func=_cmd_cascade)

    p = sub.add_parser("report", help="build the HTML report from results folders (free)")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("-o", "--output", default=str(ROOT / "report" / "index.html"))
    p.add_argument("--title", default="Jev vs LLMs")
    p.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    args.func(args)
