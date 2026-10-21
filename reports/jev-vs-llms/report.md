---
title: "jev-vs-llms: evaluation harness comparing Typesafe's Jev with LLMs on cost, quality and language coverage"
description: >
  Built a reproducible harness and ran a first benchmark of Typesafe's Jev against gpt-5-mini, gpt-5.6-sol and a
  TF-IDF baseline on the same 300 intent-detection items in English, French, Dutch and Japanese. Jev lost only 1 to
  3 points outside English (not significant), trails gpt-5.6-sol by 5 to 7 points but costs about 80 times less,
  is not significantly different from gpt-5-mini, and a Jev-first cascade matches gpt-5.6-sol at 50 to 60% lower
  cost. Claude arms and in-house data are still to run, and no adopt decision has been taken.
competence: [genai]
owner: PjCnudde98
people:
  - {name: "Pieter-Jan Cnudde", role: "R&D lead: designed and built the evaluation harness, with Claude Code"}
tags: [jev, typesafe, llm-evaluation, benchmark, multilingual, cost-analysis, classification]
links:
  - {type: repo, url: "https://github.com/PjCnudde98/jev-vs-llms"}
---

# jev-vs-llms: evaluation harness comparing Jev with LLMs

## Summary
We built a harness that answers one question with data: when does Typesafe's Jev beat an LLM on a
classification task, in quality, cost, latency, calibration and language coverage? The harness runs Jev, a cheap
and a frontier LLM and a classic baseline on identical inputs, adds a paired French, Dutch and Japanese test and
turns the results into monthly cost and break-even figures. A first run on MASSIVE (300 items per language, with
OpenAI models standing in for Claude) shows Jev holds up outside English, trails the frontier model by 5 to 7
points at about 80 times lower cost, and that a Jev-first cascade matches the frontier model at half the cost.
The decision is deferred until the Claude arms and in-house data have run.

## What was built
A Python package (`jev-eval`, managed with uv) with these parts:

- **Arms:** Jev (`jev-1.13.0`, pinned), Claude Haiku 5.5, Claude Opus 5.5, OpenAI models (used for the first
  run: `gpt-5-mini` and `gpt-5.6-sol`) and a TF-IDF plus logistic regression baseline, all built from one shared
  task file so every arm receives the same labels, descriptions and examples.
- **Datasets:** CLINC150 and Banking77 (English), plus MASSIVE in English, French, Dutch and Japanese. The same
  300 evaluation and 150 calibration utterances exist in all four languages, so English is a paired control.
  An in-house triage template is included for client-like data.
- **Metrics:** accuracy and macro-F1 with bootstrap intervals, expected calibration error, cost per 1,000 calls
  and per 1,000 correct answers, latency percentiles, consistency across repeats, and paired significance tests
  (bootstrap and exact McNemar) between arms or languages.
- **Cascade:** a simulation of "Jev first, LLM when Jev is unsure", with thresholds chosen on the calibration
  split and reported on the evaluation split.
- **Business view:** `jev-eval business` and a report calculator give monthly API cost, mistakes, the price per
  mistake at which two arms cost the same, and arms that are beaten on both cost and accuracy.
- **Report:** a self-contained HTML report with a summary, per-language comparison, significance tables and an
  interactive cost calculator.
- **Safeguards:** frozen train, calibration and evaluation splits, pinned model versions, a budget cap per run,
  a call cache and 35 offline tests that need no API keys.

## Decision
Defer. The first run covers one public dataset and OpenAI models only, so it is not enough for an adopt or reject
call. Decision rule: Jev is worth adopting for a use case when its accuracy is not significantly below the best
LLM in the paired test, or when its break-even price per mistake is above what a mistake actually costs, either on
its own or as the first stage of a cascade. It is rejected for a language or task where it is significantly worse
and the cascade does not recover the gap at acceptable cost.

Applied to the first run only, the rule points to Jev on its own when a mistake costs less than about $0.09, and
to a Jev-first cascade otherwise; no language would be rejected. To turn this into a decision: run the Claude
arms, run the in-house triage set with native Dutch and French tickets, and set a real cost per mistake with the
business owner.

## Outcome
First benchmark run, 2026-10-21: MASSIVE intent detection (60 intents), the same 300 evaluation items in English,
French, Dutch and Japanese, zero-shot with English labels. Arms: Jev (`jev-1.13.0`), `gpt-5-mini`, `gpt-5.6-sol`
(both at low reasoning effort) and a TF-IDF baseline trained on 11,514 labelled examples. Claude models are not
included yet (no Anthropic key). Total spend about $9 for the four runs.

| Language | Jev | gpt-5-mini | gpt-5.6-sol | Baseline |
| --- | --- | --- | --- | --- |
| English | 82.0% | 83.7% | 86.7% | 78.3% |
| French | 80.7% | 84.0% | 86.0% | 77.3% |
| Dutch | 80.0% | 81.0% | 86.7% | 76.7% |
| Japanese | 79.3% | 82.0% | 85.0% | 79.7% |

Each cell is accuracy on 300 items; the 95% interval is about ±4 points.

- **Language effect is small for Jev.** Against English, Jev loses 1.3 points in French, 2.0 in Dutch and 2.7 in
  Japanese; none is statistically significant (paired McNemar p of 0.20 to 0.54). `gpt-5.6-sol` shows the same
  pattern (0 to 1.7 points). With 300 items we can rule out large drops, not drops of 2 to 3 points.
- **Jev against `gpt-5.6-sol`:** 4.7 to 6.7 points lower in every language, and significant in every one
  (p from 0.0003 to 0.016). Jev costs $0.056 per 1,000 calls against $4.5, about 80 times less, and answers in
  about 0.25 s against 1.3 s.
- **Jev against `gpt-5-mini`:** 1.0 to 3.3 points lower, not significant in any language (p from 0.08 to 0.69),
  at about 6 times lower cost ($0.056 against $0.33 to $0.36 per 1,000 calls).
- **Break-even:** in English Jev is cheaper overall than `gpt-5.6-sol` whenever one mistake costs less than about
  $0.09; above that, the extra accuracy pays for itself.
- **Cascade (Jev first, `gpt-5.6-sol` when Jev is unsure):** matches `gpt-5.6-sol` within 1.4 points in every
  language (84.7% to 86.3%) while sending 33% to 44% of items to the large model. Cost is $1.66 to $2.20 per
  1,000 calls against $4.5 to $4.6, about 50 to 60% lower. Thresholds were picked on the calibration split.
- **Baseline caveat:** the TF-IDF baseline, trained on 11.5k labelled examples, scores within about 1 to 4 points
  of Jev in every language, and slightly above it in Japanese (79.7% against 79.3%). Jev and the LLMs ran
  zero-shot from label names. If labelled data is available, a classic model is a serious competitor.
- **Not yet tested:** Claude models, few-shot prompting, in-house or support-ticket data, messy input, unseen
  classes. MASSIVE is a voice-assistant domain and may be in model training data.

## R&D characteristics
- **Novelty**: Jev is a new kind of model (typed probability judgments from a small System One model), and we
  have not yet checked the literature for a public benchmark of it against LLMs on cost, calibration and French,
  Dutch and Japanese input (to do). The paired design and the break-even price per mistake are new ways for us to frame that comparison.
- **Uncertainty & risk**: Typesafe states that English is Jev's primary language and that others work "but not
  equally well", so the size of the drop in French, Dutch and Japanese is unknown. The confidence scores are not
  like for like (Jev returns probabilities, Claude self-reports), and the public datasets may be in training
  data, so the benchmark may not reflect real client traffic.
- **Systematic investigation**: One shared task definition per dataset, seeded and frozen splits, the same
  evaluation items in every language, pinned versions, cost taken from API token usage, equal tuning time per
  arm on the calibration split only, and paired significance tests. Planned experiments: head-to-head, calibration
  and cascade, stress (many classes, typos, email boilerplate), multilingual, Japanese few-shot.
- **Goal-directedness**: Give a defensible, data-based answer on when to use Jev, an LLM or a cascade for
  routing and classification, including which languages and at what volume, instead of relying on vendor claims.
- **Creativity**: Treating the question as a cost-of-error problem, with break-even prices and dominated arms,
  rather than ranking by accuracy alone. Using human-translated MASSIVE with shared ids as a controlled language
  test instead of translating English data with an LLM. We did not find an off-the-shelf tool that compares a System One model and
  LLMs on calibration and cascade routing under one fairness protocol (not exhaustively searched).

## Context & problem
Jev is Typesafe's System One model: cheap ($0.042 per million input tokens, output free), and it
returns a probability for each option. The open questions for client work are whether it is accurate enough to
replace an LLM for classification and routing, whether it can send only uncertain cases to an LLM, and whether it
holds up outside English. Dutch, French and Japanese were chosen as the test languages; how much Jev loses outside English decides
whether it is usable for multilingual work at all.

## Approach
- Fix the task and data first (one task file, frozen splits), then add arms, so no arm gets extra help.
- Use MASSIVE for languages: human-translated, parallel across locales, 60 intents. Labels and instructions stay
  in English and only the user's text changes, which is the usual production setup.
- Use character n-grams for the Japanese baseline, since Japanese has no spaces for a word tokenizer.
- Compute every metric offline from stored predictions, so analysis never needs new API calls.
- Limits to keep in mind: MASSIVE is a voice-assistant domain, so it shows language robustness, not fit for
  support tickets. Native Dutch and French tickets in the in-house triage set are still to do, and that set
  should be weighted above the public ones in the final verdict.

## Links & deliverables
- Repository: https://github.com/PjCnudde98/jev-vs-llms (branch `multilingual-evals-and-business-report`)
- Method and metric definitions: `docs/methodology.md` in the repository
- Configs for the first run: `configs/multilingual_openai_{en,fr,nl,ja}.yaml`; results in
  `results/multilingual-openai-{en,fr,nl,ja}/`
- Configs for the planned Claude runs: `configs/multilingual_{en,fr,nl,ja}.yaml`, `configs/multilingual_fewshot_ja.yaml`
- Generated report for the first run: `report/jev-vs-gpt.html` (rebuild with
  `uv run jev-eval report results/multilingual-openai-* -o report/jev-vs-gpt.html`)
