# Extending the harness

## Add an arm

1. Subclass `Arm` in `src/jev_eval/arms/` and set a unique `kind`.
2. Build the input from `self.task` with the helpers in `prompts.py`, so the new arm
   sees the same information as the others.
3. In `predict(item)`, return a `Prediction` with one `FieldAnswer` per task field.
   Set `confidence` to your best estimate of the probability that the label is right.
4. Raise `RetryableError` for rate limits, 5xx and connection errors, and create the
   SDK client with its own retries turned off: the runner retries, so backoff never
   counts as latency. Return `self._invalid(item, reason)` for anything else.
5. Compute `cost_usd` with `pricing.cost_usd(model, ...)` and add the model to
   `configs/pricing.yaml`.
6. Register the class in `src/jev_eval/arms/__init__.py`.
7. Add a mocked-HTTP test next to `tests/test_arms_http.py`.

## Add a public dataset

Write a `prepare_<name>(root)` function in `datasets.py` that loads the raw data into
`Item`s, draws `eval` and `calibration` with `_stratified_sample`, builds the task
dict, and calls `_write`. Register it in `PUBLIC`. Keep the seed and split sizes, so
results stay comparable across datasets.

For a dataset that exists in several languages, follow `prepare_massive`: draw the
eval and calibration ids once from one language and reuse them in every other, write
one dataset per language (`<name>_<code>`), and store `meta: {"language": code}` on
each item. The report finds that field and adds its by-language section when two or
more such runs are passed to it. Use `_balanced_sample` instead of
`_stratified_sample` when some labels have a single item.

## Add a metric

Add it to `field_metrics` (per output field) or `arm_metrics` (per arm) in
`metrics.py`. Metrics are computed from stored predictions only, so a new metric
needs no new API calls: run `jev-eval metrics results/<run-id>`.

Comparisons between two arms or two runs belong in `paired_difference`, which aligns
answers by item id. The report's summary, significance and by-language sections use it.

## Add a report section or cost scenario

Cost scenarios live in `business.py` (`analyse` takes plain arm dicts and returns
JSON, so it is easy to test). The report calls it with default volumes; the "Your
numbers" calculator in `report.py` redoes the same arithmetic in JavaScript from the
embedded arm data, so change both together. Add the new payload field in
`report._run_payload`, render it in a `_..._section` function, and cover it in
`tests/test_business_report.py`.

## Add a stress test

Add a field to `StressConfig` in `config.py` and handle it in `stress.apply`. Seed
any randomness per item (`random.Random(f"{SEED}-{item.id}")`) so every arm sees the
same stressed text. Then add a config under `configs/stress/`.
