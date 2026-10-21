# Configuration

## Run config (`configs/*.yaml`)

```yaml
name: clinc150                 # default prefix of the results folder
dataset: clinc150              # reads data/processed/<dataset>/ and tasks/<dataset>.yaml
splits: [calibration, eval]    # which frozen splits to run; the cascade needs both
limit: null                    # cap per split after a seeded shuffle; null = whole split
repeats: 1                     # calls per item; >1 measures consistency
concurrency: 8                 # parallel calls per arm; 1 for clean latency
use_cache: true                # false = every call is made fresh (latency runs)
max_cost_usd: 40.0             # stop starting new calls after this much new spend
stress:                        # optional, see docs/methodology.md#stress-tests
  max_classes: null
  typos: 0.0
  boilerplate: false
arms:
  - {name: jev, kind: jev, model: jev-1.13.0, params: {}}
```

`name` must be unique within a config; it becomes the file name
`results/<run-id>/<split>/<name>.jsonl` and the label in the report.

### Arm kinds and parameters

| Kind | Models | Parameters |
| --- | --- | --- |
| `jev` | `jev-1.13.0`, `jev-1.14.0`, `jev-latest`, `jev-preview` | `examples_per_label` (0), `timeout` seconds (30) |
| `anthropic` | `claude-haiku-5-5`, `claude-haiku-4-5`, `claude-sonnet-5-5`, `claude-opus-5-5` | `effort` (`low`...`max`; leave out for Haiku 4.5), `max_tokens` (4096, includes thinking), `prompt_cache` (true), `examples_per_label` (0), `timeout` (120) |
| `openai` | any chat model with JSON-schema output | `reasoning_effort`, `max_completion_tokens` (4096), `examples_per_label`, `timeout`. Needs `uv sync --extra openai` and a price row |
| `baseline` | `baseline-tfidf-logreg` | `C` (10.0), the regularisation strength |

Changing any parameter changes the arm's fingerprint, so its calls are not served
from the cache of the old setup.

## Environment variables

| Variable | Used by |
| --- | --- |
| `TYPESAFE_API_KEY` | `jev` arms |
| `ANTHROPIC_API_KEY` | `anthropic` arms (an `ant auth login` profile also works) |
| `OPENAI_API_KEY` | `openai` arms |

Keep them in `.env` (git-ignored); see `.env.example`.

## Pricing (`configs/pricing.yaml`)

USD per million tokens, per model ID:

```yaml
claude-opus-5-5: {input: 4.00, output: 20.00, cache_read: 0.20, source: "Anthropic API pricing", as_of: "2026-10-06"}
```

`cache_read` and `cache_write` are optional; without them the harness uses 0.1x and
1.25x the input price. A run refuses an arm whose model has no row, so a cost is
never silently zero. Jev bills input tokens only ($0.042 per million on Jev 1.13 and
1.14); output tokens are free.

## Task file (`tasks/<dataset>.yaml`)

The single source for what every arm is asked:

```yaml
name: clinc150
context: "Messages a user sent to a general-purpose virtual assistant."
fields:
  - name: intent
    instruction: "Which intent does the user's message express?"
    labels: [accept_reservations, account_blocked, ...]
    descriptions: {account_blocked: "Account blocked", oos: "Out of scope: ..."}
    out_of_scope_label: oos     # optional
```

A task can have several fields (the triage template has four). Jev asks them as
parallel Choice questions in one request; LLMs return them in one JSON object.

## Data format (`data/processed/<dataset>/*.jsonl`)

One item per line:

```json
{"id": "clinc-test-1007", "text": "maybe we can go to the next song", "labels": {"intent": "next_song"}, "meta": {}}
```

For your own set, write `items.jsonl` in this shape and run
`jev-eval prepare <name> --source items.jsonl --task tasks/<name>.yaml`.

## Results folder (`results/<run-id>/`)

| File | Content |
| --- | --- |
| `config.yaml` | the config as run |
| `manifest.json` | start and end time, package versions, per arm: model versions that answered, fingerprint, new spend, wall-clock time |
| `<split>/<arm>.jsonl` | one prediction per call: answers with label, probabilities and confidence; tokens; cost; latency; validity; raw output when invalid |
| `metrics.json` | written by `metrics`, `cascade` and `report` |
| `cascade/<name>.json` | threshold sweep and operating points |
| `verdict.md` | optional, written by you: shown at the top of that run's report section |

The call cache is `results/.cache.sqlite`. It is git-ignored because it can hold
responses for in-house data.
