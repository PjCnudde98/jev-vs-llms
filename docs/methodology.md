# Methodology

How the harness keeps the comparison fair, what every number means, and what to
keep in mind when reading the results.

## Arms

| Arm | Implementation | What it tells us |
| --- | --- | --- |
| A. Jev | `arms/jev.py`: one Choice question per output field, all in one request, through `typesafe-sdk` | The system one result |
| B. Cheap LLM | `arms/llm.py`, Claude Haiku 5.5 (`claude-haiku-5-5`) at `effort: low`, JSON-schema structured output | The realistic default a team would reach for |
| C. Frontier LLM | Claude Opus 5.5 (`claude-opus-5-5`) at `effort: low`; optionally an OpenAI model | The quality ceiling |
| D. Cascade | `cascade.py`: Jev first, B or C when Jev's confidence is under a threshold | Whether system one + system two beats either alone |
| E. Classic baseline | `arms/baseline.py`: TF-IDF + logistic regression trained on the train split | The floor. If Jev barely beats it, the story changes |

Claude Haiku 4.5, named in the initiative, is available as an optional arm
(`claude-haiku-4-5`, no `effort` parameter). The default cheap arm is Haiku 5.5,
the current generation, at a tenth of Haiku 4.5's price.

Opus 5.5 always thinks; `effort: low` keeps that short for a classification task.
Its thinking tokens are billed as output and are included in the cost.

## Fairness rules

1. **Same information for every arm.** `prompts.py` builds every arm's input from one
   task file (`tasks/<dataset>.yaml`): the same context sentence, instruction, label
   names and label descriptions. The LLM gets them as a system prompt plus a JSON
   schema whose `label` is an enum of the allowed labels. Jev gets them as Choice
   `instructions` and `criteria`.
2. **Same examples.** With `examples_per_label: k`, every arm gets the same k
   training texts per label (seeded). Jev receives them inside each option's
   criteria object (`{"description", "examples"}`), as Typesafe recommends; LLMs get
   them in the prompt.
3. **No tuning on the test set.** Each dataset has three frozen splits:
   - `train`: few-shot examples and baseline training only
   - `calibration` (500 items): prompt or schema iteration and cascade thresholds
   - `eval` (1,000 items, stratified by label): reported numbers only
4. **Equal tuning time.** Prompt and schema work is capped at 2 hours per arm, on the
   calibration split, and logged in the build-effort log. Then they are frozen.
5. **Pinned versions.** Configs name `jev-1.13.0`, not the moving `jev-latest` alias.
   Every prediction stores the model version that answered.
6. **Cost from usage, not estimates.** Every call's cost comes from the token counts
   the API returned, times `configs/pricing.yaml`. Prompt-cache reads and writes are
   priced separately. The LLM arms put a cache breakpoint on the system prompt, as a
   production setup would.

## Splits and data

| Dataset | Source | Labels | Eval | Calibration | Train |
| --- | --- | --- | --- | --- | --- |
| CLINC150 | `clinc/clinc_oos` "plus" config on Hugging Face | 150 intents + `oos` (out of scope) | 1,000 from test | 500 from validation | 15,250 |
| Banking77 | PolyAI `task-specific-datasets` on GitHub | 77 intents | 1,000 from test | 500 held out of train | 9,503 |
| MASSIVE (en, fr, nl, ja) | Amazon, human-translated, CC BY 4.0 | 60 intents | 300 from test | 150 from dev | 11,514 |

Splits are drawn with a fixed seed and stratified by label. CLINC150's eval split
keeps the test set's share of out-of-scope items, so out-of-scope handling is
measured on the same set as everything else.

## Multilingual (MASSIVE)

`uv run jev-eval prepare massive` writes `massive_{en,fr,nl,ja}`. The utterances are
human translations of each other, so the **same 300 eval and 150 calibration ids** are
used in every language (spread evenly over the intents, drawn once from English). English
is the control: a gap between languages is a language effect, not a different item mix.

- Labels, descriptions and instructions stay in English; only the user's text changes
  language. This is the common production setup (one English schema, mixed-language input).
- Few-shot examples come from the same language's train split.
- The baseline for Japanese uses character n-grams (`params: {analyzer: char_wb}`), since
  Japanese has no spaces for a word tokenizer to split on.
- MASSIVE is a voice-assistant domain, so it shows language robustness, not fit for
  support tickets or banking. The in-house triage set covers that.
- Compare languages or arms with the paired test (free, from stored predictions):
  `jev-eval compare results/multilingual-ja:jev results/multilingual-en:jev` reports the
  accuracy difference, a paired bootstrap interval and an exact McNemar p-value.

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy | Share of eval items where the label is right. Invalid answers count as wrong. 95% CI from 1,000 bootstrap resamples |
| Macro-F1 | F1 per gold class, averaged without weighting, with the same bootstrap CI |
| Per-class F1 | Precision, recall and F1 per class; the report shows the 25 hardest classes |
| Out-of-scope precision/recall | For tasks with an out-of-scope label: how often "none of these" is said, and how often it is said when it should be |
| Confidence | The arm's estimate that its label is right. Jev and baseline: probability of the chosen label. LLMs: their self-reported 0-1 score |
| ECE | Expected calibration error over 10 equal-width confidence bins: the weighted gap between stated confidence and actual accuracy. 0 is perfect |
| Brier score | Mean squared gap between confidence and correctness (0 or 1) |
| $ / 1k | Mean cost per call x 1,000 |
| $ / 1k correct | Total cost divided by correct items x 1,000. Folds quality and cost into one number and makes cascades comparable to single arms |
| Latency p50 / p95 / p99 | Wall-clock seconds per successful call. Retries and backoff are excluded: arms do not retry themselves, the runner does, and only the successful attempt is timed |
| Consistency | With `repeats > 1`: share of items whose repeats all gave the same label |
| Invalid rate | Share of calls with no usable answer: errors, refusals, unparseable or out-of-schema output |
| Task accuracy | Share of items where every output field is right. Equals accuracy for single-field tasks; the basis of `$ / 1k correct` and of the cost calculator |
| Paired difference | Accuracy of one arm minus another on the same items, with a paired bootstrap 95% interval and an exact McNemar p-value on the items where exactly one arm is right. Also used across languages for the same arm |

Typesafe also returns its own `confidence` statistic (a normalised distance from an
even split, see [typesafe-notes.md](typesafe-notes.md)). It is stored as
`vendor_confidence`, but calibration uses the top probability, because only a
probability can be compared with observed accuracy.

## The cascade

`jev-eval cascade` simulates "Jev first, LLM when unsure" offline from stored
predictions, so it costs nothing extra. For each item: if Jev's confidence (the
lowest over the fields) is at or above the threshold, Jev's answer stands.
Otherwise the LLM's answer is used, and the item costs both calls and takes both
latencies.

Thresholds from 0 to 1 are swept on the **calibration** split. Two operating points
are chosen there and then reported on **eval**:

- `match_fallback`: the cheapest threshold whose accuracy is within 0.5 points of the
  LLM alone
- `within_2x_cost`: the most accurate threshold that costs at most twice Jev alone

Each point is also written as an arm (`eval/cascade-jev-to-opus-5.5@match_fallback.jsonl`),
so metrics and the report treat it like any other arm.

Cascade latency is the sum of the two calls for routed items, a worst case. A real
system could start the LLM call speculatively.

## Cost at scale

`jev-eval business results/<run>` (and the "What it costs at scale" section of the
report) turns the measured accuracy and cost per call into monthly figures:

- **API cost per month** at each volume (default 10k, 100k, 1M calls): volume x cost
  per call. Input and output tokens are priced as in the run; hosting and engineering
  time are not included, so the local baseline shows $0.
- **Mistakes per month**: volume x (1 - task accuracy).
- **All-in cost** when a price per mistake is given: API cost plus mistakes x price.
- **Break-even price per mistake**: against a reference arm (by default the most
  accurate LLM), the price at which an arm costs the same overall. A cheaper, less
  accurate arm wins when a mistake costs *less* than this; a costlier, more accurate
  arm wins when a mistake costs *more*. Below it, the extra accuracy does not pay for
  itself.
- **Dominated arms**: another arm is at least as cheap and at least as accurate, and
  better on one of the two. There is no case for choosing a dominated arm.

The report also embeds a calculator ("Your numbers") where a reader enters their volume
and cost per mistake and sees the cheapest all-in option for their case. The figures use
the eval split's accuracy, so they carry the same sampling uncertainty as the accuracy
interval in the head-to-head table; at a few hundred items, treat close totals as ties.

## Reading the report

| Section | What it answers |
| --- | --- |
| Summary | Per run: Jev against the most accurate LLM, the paired gap and p-value, cost ratio, and the cascade's saving. Generated, not written; the human verdict goes in `verdict.md` |
| By language | Appears when two or more runs come from a dataset with `meta.language` (MASSIVE). Accuracy per arm and language, paired change against English, and cost per 1k correct |
| Across runs | Accuracy of every arm in every run passed to `report` (e.g. clean vs stress runs) |
| Per run: head-to-head, is the difference real, cost at scale, cost vs accuracy, latency, calibration, cascade, hardest classes, confident mistakes, method | Detail behind the summary |

## Stress tests

Each test changes one thing and keeps the rest fixed (`stress.py`, seeded per item):

| Test | Config | What changes |
| --- | --- | --- |
| Many classes | `stress/clinc150_classes_{10,25,77}.yaml` | Labels cut to k intents (seeded pick, `oos` kept); items outside them dropped. Compare with the full 150 |
| Messy input | `stress/clinc150_messy.yaml` | 5% character typos, plus an email greeting, signature, disclaimer and quoted thread |

Pass every stress run plus the clean run to `report` to get the comparison chart.

## What is not automated yet

| Test | Plan |
| --- | --- |
| Native NL/FR tickets | Native-language tickets in the triage set (MASSIVE covers the language comparison, not the support domain). Typesafe states English is where Jev is best |
| Unseen classes | Hold back 3 classes from examples and descriptions, then add them at test time with a description only |
| Reasoning-heavy | About 100 triage items whose label follows from a rule ("high priority if outage and paying customer") |
| Build effort (E4) | Fill in [build-effort-log.md](build-effort-log.md) while building each arm |

## Caveats when reading results

- **Benchmark contamination.** Public sets such as CLINC150 and Banking77 are likely
  in the training data of LLMs, and maybe of Jev. Weight the in-house triage set more
  in the verdict, and report public and in-house results separately.
- **Different kinds of confidence.** Jev returns a probability distribution; Claude
  has no log-probabilities, so its confidence is self-reported. Calibration therefore
  compares a model's probabilities with a model's self-assessment, which is the
  realistic comparison for routing, but not like for like.
- **Latency depends on load.** Use the latency config (one call at a time, cache off)
  for headline numbers and run it at three times of day. Main runs use concurrency 8,
  which can inflate latency under rate limits.
- **Label descriptions are humanised names.** For the public sets, descriptions are
  generated from label names (`card_arrival` -> "Card arrival"). Richer descriptions
  would help every arm; if you write them, write them once in the task file.
