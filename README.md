# Jev vs LLMs

A benchmark harness that compares Typesafe's **Jev** (a "system one" model: typed
questions in, probabilities out) with a cheap LLM and a frontier LLM on the same
task, data and input. It measures quality, calibration, cost, latency, consistency
and robustness, simulates a **Jev-first cascade** that sends only uncertain cases to
an LLM, and builds an HTML report.

It belongs to the "Jev vs LLMs" initiative. The plan, with the questions, arms,
experiments and timeline, is in the initiative doc. This repo is the harness and
the frozen datasets, so the study can be rerun when a new Jev version ships.

## Quick start

```bash
# 1. Install (uv manages Python 3.12 and all packages)
uv sync                      # add --extra openai for OpenAI arms

# 2. API keys
cp .env.example .env         # fill in TYPESAFE_API_KEY and ANTHROPIC_API_KEY
set -a; source .env; set +a

# 3. Data: downloads the public benchmarks and freezes the splits into data/processed/ (seeded, so reruns give identical files)
uv run jev-eval prepare clinc150
uv run jev-eval prepare banking77

# 4. Check the pipeline for free (local baseline only, no keys needed)
uv run jev-eval run configs/offline_check.yaml

# 5. Smoke test with real APIs: 20 items, well under $0.05
uv run jev-eval estimate configs/smoke.yaml
uv run jev-eval run configs/smoke.yaml

# 6. The main experiment (about $16 on CLINC150, mostly Opus; see `estimate`)
uv run jev-eval estimate configs/clinc150.yaml
uv run jev-eval run configs/clinc150.yaml --run-id clinc150-v1
uv run jev-eval cascade results/clinc150-v1 --primary jev --fallback opus-5.5
uv run jev-eval cascade results/clinc150-v1 --primary jev --fallback haiku-5.5

# 7. Report
uv run jev-eval report results/clinc150-v1 results/banking77-v1 -o report/index.html
open report/index.html
```

Every `run` stops starting new calls once it has spent `max_cost_usd` (set per
config). Every call is cached, so rerunning a config, or running `metrics`,
`cascade` and `report`, costs nothing.

## Which config answers which question

| Question | Experiment | Config | After the run |
| --- | --- | --- | --- |
| How does Jev compare on quality, cost and latency? | E1 head-to-head | `configs/clinc150.yaml`, `configs/banking77.yaml` | `report` |
| Are the probabilities usable? Does a cascade pay off? | E2 calibration + cascade | same runs (they include the calibration split) | `cascade`, then `report` |
| Same input, same answer? | E1 consistency | `configs/consistency_clinc150.yaml` (200 items x 5) | `metrics` |
| Clean latency numbers | E1 latency | `configs/latency_clinc150.yaml` (one call at a time, cache off) | `metrics`; run at 3 times of day |
| Do few-shot examples change the picture? | E1 variant | `configs/fewshot_clinc150.yaml` | `report` |
| Many classes | E3 stress | `configs/stress/clinc150_classes_{10,25,77}.yaml` | `report` with all runs |
| Long or messy input | E3 stress | `configs/stress/clinc150_messy.yaml` | `report` with the clean run |
| In-house triage set | E1-E3 on client-like data | copy `tasks/triage.example.yaml`, then `prepare triage --source ...` | as above |

Not automated yet: Dutch/French translations, unseen classes, reasoning-heavy rules,
and the build-effort log (E4). See [docs/methodology.md](docs/methodology.md#what-is-not-automated-yet).

## Commands

| Command | What it does | Costs money |
| --- | --- | --- |
| `jev-eval prepare <dataset>` | Download a public set (or load your own) and freeze train / calibration / eval splits | No |
| `jev-eval estimate <config>` | Rough cost per arm before running | No |
| `jev-eval run <config> [--run-id ID]` | Run every arm, store raw predictions, print metrics | Yes, capped by `max_cost_usd` |
| `jev-eval metrics <results/ID>` | Recompute all metrics from stored predictions | No |
| `jev-eval cascade <results/ID> --primary A --fallback B` | Simulate "A first, B when unsure" from stored predictions | No |
| `jev-eval report <results/ID>... [-o file]` | Build the HTML report | No |

## Layout

```
configs/          run configs (one per experiment) and pricing.yaml
tasks/            the label set + descriptions every arm receives, per dataset
data/processed/   frozen splits: train.jsonl, calibration.jsonl, eval.jsonl
src/jev_eval/
  arms/           jev.py (Typesafe), llm.py (Anthropic, OpenAI), baseline.py (TF-IDF)
  prompts.py      builds the same instructions for every arm from a task
  runner.py       concurrency, retries, call cache, budget guard
  metrics.py      accuracy, macro-F1, bootstrap CIs, ECE, cost, latency, consistency
  cascade.py      threshold sweep and operating points
  stress.py       class subsampling, typos, email boilerplate
  report.py       the HTML report
results/<run-id>/ config.yaml, manifest.json, <split>/<arm>.jsonl, metrics.json, cascade/, verdict.md
report/           generated reports
tests/            pytest suite; no API keys needed
docs/             methodology, configuration, extending, Typesafe API notes
```

## Documentation

- [docs/methodology.md](docs/methodology.md): how the comparison is kept fair, what each metric means, and the known caveats
- [docs/configuration.md](docs/configuration.md): run configs, arm parameters, task and data formats, pricing
- [docs/extending.md](docs/extending.md): adding an arm, a dataset, a metric or a stress test
- [docs/typesafe-notes.md](docs/typesafe-notes.md): what we learned about the Jev API and SDK
- [docs/build-effort-log.md](docs/build-effort-log.md): template for the build-effort measurements (E4)

## Rerunning for a new Jev version

1. Add the version to `configs/pricing.yaml` (for example `jev-1.15.0`).
2. Add an arm with that model to the config, for example `{name: jev-1.15, kind: jev, model: jev-1.15.0}`.
3. `run` the config again with a new `--run-id`. The other arms come from the cache for free.
4. `cascade` and `report` as before. The report banner shows which Jev version answered.

## Development

```bash
uv run pytest            # 20 tests, offline (mocked HTTP for Typesafe and Anthropic)
uv run ruff check src tests
uv run ruff format src tests
```
