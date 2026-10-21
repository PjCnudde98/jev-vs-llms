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

## Add a metric

Add it to `field_metrics` (per output field) or `arm_metrics` (per arm) in
`metrics.py`. Metrics are computed from stored predictions only, so a new metric
needs no new API calls: run `jev-eval metrics results/<run-id>`.

## Add a stress test

Add a field to `StressConfig` in `config.py` and handle it in `stress.apply`. Seed
any randomness per item (`random.Random(f"{SEED}-{item.id}")`) so every arm sees the
same stressed text. Then add a config under `configs/stress/`.
