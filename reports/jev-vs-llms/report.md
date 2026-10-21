---
title: "jev-vs-llms: evaluation harness comparing Typesafe's Jev with LLMs on cost, quality and language coverage"
description: >
  Built a reproducible harness and benchmarked Typesafe's Jev against Claude Haiku 5.5, Claude Opus 5.5,
  gpt-5-mini, gpt-5.6-sol and a TF-IDF baseline on the same 300 intent-detection items in English, French, Dutch
  and Japanese. Jev holds up outside English, but Haiku 5.5 with prompt caching is at least as accurate and
  cheaper per call in every language, and Opus 5.5 is 8 to 9 points more accurate. Jev's remaining value is
  latency and routing: a Jev-first cascade matches Opus at about half its cost. Decision deferred until in-house
  data has run.
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
classification task, in quality, cost, latency, calibration and language coverage? On MASSIVE intent detection
(300 paired items in English, French, Dutch and Japanese), Jev loses only 1 to 3 points outside English, but
Claude Haiku 5.5 is at least as accurate and, with prompt caching, cheaper per call in all four languages.
Claude Opus 5.5 is the most accurate model (88 to 90%), and a Jev-first cascade gets within 1 point of it at
about half the cost. The decision is deferred until the in-house triage data has run.

## What was built
A Python package (`jev-eval`, managed with uv) with these parts:

- **Arms:** Jev (`jev-1.13.0`, pinned), Claude Haiku 5.5, Claude Opus 5.5, OpenAI `gpt-5-mini` and `gpt-5.6-sol`,
  and a TF-IDF plus logistic regression baseline, all built from one shared task file so every arm receives the
  same labels, descriptions and examples.
- **Datasets:** CLINC150 and Banking77 (English), plus MASSIVE in English, French, Dutch and Japanese. The same
  300 evaluation and 150 calibration utterances exist in all four languages, so English is a paired control.
  An in-house triage template is included for client-like data.
- **Metrics:** accuracy and macro-F1 with bootstrap intervals, expected calibration error, cost per 1,000 calls
  and per 1,000 correct answers, latency percentiles, consistency across repeats, and paired significance tests
  (bootstrap and exact McNemar) between arms or languages.
- **Cascade:** a simulation of "cheap model first, large model when it is unsure", with thresholds chosen on the
  calibration split and reported on the evaluation split.
- **Business view:** `jev-eval business` and a report calculator give monthly API cost, mistakes, the price per
  mistake at which two arms cost the same, and arms that are beaten on both cost and accuracy.
- **Report:** a self-contained HTML report with a summary, per-language comparison, significance tables and an
  interactive cost calculator.
- **Safeguards:** frozen train, calibration and evaluation splits, pinned model versions, a budget cap per run,
  a call cache and 35 offline tests that need no API keys.

## Decision
Defer. The benchmark covers one public, voice-assistant dataset, which is not enough for an adopt or reject call
on client work. Decision rule: Jev is worth adopting for a use case when its accuracy is not significantly below
the best LLM in the paired test, or when its break-even price per mistake is above what a mistake actually costs,
either on its own or as the first stage of a cascade. It is rejected for a language or task where it is
significantly worse and the cascade does not recover the gap at acceptable cost.

What the MASSIVE results say under that rule:

- **Jev on its own is not the best choice.** Haiku 5.5 is cheaper and at least as accurate in every language, so
  Jev only wins on its own where latency matters (about 0.25 s against 1 s for Haiku), or where prompt caching
  is not possible (Haiku would then cost about 3.6 times more than Jev).
- **Jev as a router is worth keeping.** A Jev-first cascade to Opus recovers the gap at about half the cost of
  Opus, and in Dutch and Japanese it routes far fewer items to Opus than a Haiku-first cascade does.
- **No language is rejected.** Jev's drop outside English is small and not significant.

To turn this into a decision: run the in-house triage set with native Dutch and French tickets, and set a real
cost per mistake with the business owner.

## Outcome
Benchmark runs, 2026-10-21: MASSIVE intent detection (60 intents), the same 300 evaluation items in English,
French, Dutch and Japanese, zero-shot with English labels. All LLMs at low effort. The TF-IDF baseline is trained
on 11,514 labelled examples. Total spend about $11 for the eight runs ($8.78 for the OpenAI arms, $2.32 for Claude).

**Accuracy** (300 items per cell; the 95% interval is about ±4 points):

| Language | Jev | Haiku 5.5 | Opus 5.5 | gpt-5-mini | gpt-5.6-sol | Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| English | 82.0% | 85.7% | 89.7% | 83.7% | 86.7% | 78.3% |
| French | 80.7% | 83.7% | 89.3% | 84.0% | 86.0% | 77.3% |
| Dutch | 80.0% | 81.0% | 88.7% | 81.0% | 86.7% | 76.7% |
| Japanese | 79.3% | 82.7% | 87.7% | 82.0% | 85.0% | 79.7% |

**Cost and speed** (evaluation split, same in every language within a few cents):

| | Jev | Haiku 5.5 | Opus 5.5 | gpt-5-mini | gpt-5.6-sol |
| --- | --- | --- | --- | --- | --- |
| $ per 1,000 calls | $0.056 | $0.035 to $0.036 | $1.03 to $1.12 | $0.33 to $0.36 | $4.49 to $4.56 |
| Median latency | 0.25 s | 1.0 s | 1.7 s | 1.3 s | 1.3 s |

Claude costs include prompt caching: the shared task prompt (about 1,900 tokens) is read from cache and only the
user's text (about 20 to 30 tokens) is billed at the full input price. The OpenAI runs got no cache hits. The Haiku
cache-read price is the harness default (0.1 times the input price), as no explicit price is configured. Without
caching, Haiku would cost about $0.20 per 1,000 calls.

- **Jev against Haiku 5.5:** Haiku is 1.0 to 3.7 points more accurate, not significant in any language (paired
  McNemar p from 0.08 to 0.77), and about 35% cheaper per call. Haiku beats Jev on both cost and accuracy in all
  four languages. Jev answers about 4 times faster.
- **Jev against Opus 5.5:** 7.7 to 8.7 points lower, significant in every language (p ≤ 0.0002). Opus costs about
  20 times more than Jev. In English, Opus is cheaper overall than Jev whenever one mistake costs more than about
  $0.013, and cheaper than Haiku above about $0.025.
- **Claude against OpenAI:** Opus is 2.0 to 3.3 points above `gpt-5.6-sol` (significant only in French, p = 0.04)
  at about a quarter of the cost. Haiku is within 2 points of `gpt-5-mini` (not significant) at about a tenth of
  the cost.
- **Language effect:** against English, Jev loses 1.3 to 2.7 points and Opus 0.3 to 2.0 points, none significant.
  Haiku loses 2.0 to 4.7 points, and its Dutch drop is significant (4.7 points, p = 0.02). With 300 items we can
  rule out large drops, not drops of 2 to 3 points.
- **Cascade Jev first, Opus when Jev is unsure:** within 0.3 to 1.0 points of Opus in English, Dutch and Japanese
  (87.3% to 89.3%), sending 40% to 50% of items to Opus, at $0.53 to $0.60 per 1,000 calls against $1.03 to $1.12
  for Opus alone (43% to 52% cheaper). In French no threshold on the calibration split stayed within tolerance.
- **Cascade Haiku first, Opus when Haiku is unsure:** cheaper than Jev-first in English ($0.46, 39% routed) and the
  only one that works in French ($0.62, 50% routed, 89.3%). In Dutch and Japanese, Haiku's self-reported confidence
  sends 74% to 76% of items to Opus, so the cascade costs $0.87 to $0.91, against $0.53 to $0.60 with Jev first.
- **Cascade Jev first, Haiku when Jev is unsure:** never worth it, because Haiku on its own is already cheaper
  than calling Jev.
- **Calibration:** on the evaluation split, the self-reported confidence of Opus (ECE 0.04 to 0.06) was better
  calibrated than Jev's probabilities (0.08 to 0.09), yet Jev's probabilities still routed better in Dutch and
  Japanese. Calibration error alone does not predict routing quality.
- **Baseline caveat:** the TF-IDF baseline, trained on 11.5k labelled examples, scores within about 1 to 4 points
  of Jev in every language, and slightly above it in Japanese (79.7% against 79.3%). Jev and the LLMs ran
  zero-shot from label names. If labelled data is available, a classic model is a serious competitor.
- **Not yet tested:** few-shot prompting, in-house or support-ticket data, messy input, unseen classes, LLMs
  without prompt caching in the same run. MASSIVE is a voice-assistant domain and may be in model training data.

## R&D characteristics
- **Novelty**: Jev is a new kind of model (typed probability judgments from a small System One model), and we
  have not yet checked the literature for a public benchmark of it against LLMs on cost, calibration and French,
  Dutch and Japanese input (to do). The paired design and the break-even price per mistake are new ways for us to frame that comparison.
- **Uncertainty & risk**: Typesafe states that English is Jev's primary language and that others work "but not
  equally well", so the size of the drop in French, Dutch and Japanese was unknown. The confidence scores are not
  like for like (Jev returns probabilities, LLMs self-report), cost depends on whether prompt caching applies, and
  the public datasets may be in training data, so the benchmark may not reflect real client traffic.
- **Systematic investigation**: One shared task definition per dataset, seeded and frozen splits, the same
  evaluation items in every language, pinned versions, cost taken from API token usage, equal tuning time per
  arm on the calibration split only, and paired significance tests. Experiments run: head-to-head, calibration,
  cascade and multilingual on MASSIVE with five LLMs. Still planned: stress (many classes, typos, email
  boilerplate), Japanese few-shot, in-house triage data.
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
- The OpenAI arms were run first while no Anthropic key was available; the Claude arms reuse the cached Jev and
  baseline answers, so all arms are compared on identical predictions for those two.
- Limits to keep in mind: MASSIVE is a voice-assistant domain, so it shows language robustness, not fit for
  support tickets. Native Dutch and French tickets in the in-house triage set are still to do, and that set
  should be weighted above the public ones in the final verdict.

## Links & deliverables
- Repository: https://github.com/PjCnudde98/jev-vs-llms (branch `multilingual-evals-and-business-report`)
- Method and metric definitions: `docs/methodology.md` in the repository
- Configs: `configs/multilingual_{en,fr,nl,ja}.yaml` (Claude), `configs/multilingual_openai_{en,fr,nl,ja}.yaml`
  (OpenAI), `configs/multilingual_fewshot_ja.yaml` (planned)
- Results: `results/multilingual-claude-{en,fr,nl,ja}/` and `results/multilingual-openai-{en,fr,nl,ja}/`,
  including cascades and `business.json` break-even tables
- Generated reports: `report/jev-vs-claude.html` and `report/jev-vs-gpt.html` (rebuild with
  `uv run jev-eval report results/multilingual-claude-* -o report/jev-vs-claude.html`)
