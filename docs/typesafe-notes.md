# Notes on the Typesafe API and Jev

What we learned from Typesafe's docs and SDK while building arm A. Sources:
[docs.typesafe.ai](https://docs.typesafe.ai/llms.txt), checked 2026-10-21, and
`typesafe-sdk` 0.7.5.

## The call

- One endpoint: `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <TYPESAFE_API_KEY>`.
- A request carries a `state` (string, object or array), a `model`, and `questions`:
  a map of names you choose to typed questions. Questions are evaluated in parallel
  against the same state, so adding questions barely changes latency.
- Question types: **Choice** (pick one of up to 255 options), **Score** (ordered
  rubric of up to 10 levels), **Noul** (probability that a statement is true), and
  **Box** (image location, behind a header).
- `instructions` and each option's `criteria` can be a string, an object or an
  array. An object such as `{"description": ..., "examples": [...]}` is recommended
  when options are easy to confuse. The model never sees the question name.
- The SDK: `AsyncTypeSafeClient(model=..., retry=RetryPolicy(...), timeout=...)`,
  `await client.system_one(state, questions)`, then `result.choices[name].choice`,
  `.probabilities`, `.confidence`, and `result.usage.input_tokens`. Close with
  `await client.aclose()`.

## Answers and confidence

- A Choice answer returns the chosen option, a probability for every option, and a
  `confidence`.
- `confidence` is derived from the probabilities: for n options,
  `(p_max - 1/n) / (1 - 1/n)`. It is 0 for an even split and 1 when all mass is on one
  option. It is not a probability that the answer is right; the harness stores it
  as `vendor_confidence` and uses `p_max` for calibration.
- Typesafe suggests `p_max` or the top-to-second ratio as alternatives for routing,
  and recommends a companion Noul ("does any option fit?") over a "none of the
  above" option when out-of-scope inputs matter. This is worth a variant on CLINC150,
  which has an `oos` class.

## Models, price and limits

| | Jev 1.14 | Jev 1.13 |
| --- | --- | --- |
| Model ID | `jev-1.14.0` | `jev-1.13.0` |
| Price | $0.042 per million input tokens; output free | same |
| Context | 64k per request, 32k per question (262k with the long-context header) | same |

- Aliases: `jev-latest` points to `jev-1.13.0` (latest stable), `jev-preview` to
  `jev-1.14.0` (released 2026-10-14). The response's `model` field gives the version
  that answered. Configs pin `jev-1.13.0`.
- Rate limits are per organisation, in tokens and requests per second; see the
  console. A 429 can carry `retry-after`, which the runner honours.
- Language: English is the primary training language. Other languages work "but not
  equally well", so the NL/FR test matters.
- Jev is not fine-tuned per customer. Domain knowledge goes into the state and the
  question criteria.

## Gotchas we hit

- Creating a client with a missing or malformed key raises `TypeSafeError` at
  construction, before any request.
- `TypeSafeAPIError` carries `status`, `headers` and `request_id`; connection and
  timeout errors are separate classes (`TypeSafeAPIConnectionError`,
  `TypeSafeAPITimeoutError`).
- The SDK retries by default (2 retries, exponential backoff). The harness turns
  this off so backoff never counts as latency, and retries in the runner instead.
- The SDK compresses request bodies with zstd when it is available.
