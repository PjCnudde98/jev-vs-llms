---
# Copy this file to reports/<your-slug>/report.md and fill it in.
# Slug is kebab-case (e.g. pytest-agent-eval). One folder per initiative.
# A report exists only once the initiative is CONCLUDED — live work lives on Issues.
#
# Seven keys. Everything git/GitHub already knows (dates, authorship, status,
# the linked issue) is deliberately excluded — it is derived, never typed.
title: "<slug>: one-line what-it-is"
description: >
  2–4 sentences, OUTCOME-FIRST. This is the routing key: Claude reads this
  (not the body) to decide if a report is relevant. Say what was built, what
  was decided, and the result.
competence: [genai]                       # ≥1 from the controlled vocab below; ≥2 ⇒ xCompetence
owner: your-github-handle                  # single accountable person (GitHub handle)
people:                                    # who did the R&D work (≠ git file authors)
  - {name: "Full Name", role: "role — contribution"}
tags: [keyword, another-keyword]           # free-form, lowercase-kebab
links:                                     # external deliverables / references (optional list)
  - {type: repo,   url: "https://github.com/datarootsio/..."}
  - {type: docs,   url: "https://..."}
  - {type: slides, url: "https://..."}     # e.g. a SharePoint/Google Slides deck
# Controlled vocab for `competence` (pick from these exact values):
#   software-engineering, genai, mlops, cloud-engineering,
#   databricks, azure, analytics-engineering, data-governance, traditional-bi
# `links[].type` is free-form but keep it short: repo, docs, pypi, slides, dataset, demo, paper.
---

# <Title>

## Summary
<!-- required — 2–4 sentences, outcome-first. -->

## What was built
<!-- required — the artifact / experiment / capability. -->

## Decision
<!-- required — adopt / reject / defer, and why. -->

## Outcome
<!-- required — the result. Negative results ("no for now") are valid and useful. -->

## R&D characteristics
<!-- required — the 5 O&O characteristics, 1–2 sentences each. -->
- **Novelty** — what's new (for us or the market).
- **Uncertainty & risk** — technical unknowns; the risk the result isn't reached.
- **Systematic investigation** — the experiments / prototypes / tests run.
- **Goal-directedness** — the specific application/outcome targeted.
- **Creativity** — the non-standard approach; why no off-the-shelf solution.

## Context & problem
<!-- optional — why we ran it; the business/client trigger. -->

## Approach
<!-- optional — method, what we tried. -->

## Links & deliverables
<!-- optional — external links + pointers to files committed in this folder. -->
