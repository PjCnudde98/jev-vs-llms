# Build-effort log (experiment E4)

Fill in one row per arm per task while building. Two people build one arm each,
independently, on one task to get a second data point. Count only hands-on time.

| Date | Person | Task | Arm | Hours to first working version | Hours to frozen version | Iterations (schema or prompt) | Lines of task-specific code | Friction (docs gaps, SDK quirks, rate limits) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | clinc150 | jev | | | | | |
| | | clinc150 | haiku-5.5 | | | | | |
| | | triage | jev | | | | | |
| | | triage | haiku-5.5 | | | | | |

The shared harness code (runner, metrics, report) is not counted: it is the same
for every arm. What counts is the arm-specific work: writing the schema or prompt,
the label descriptions, and the iterations on the calibration split.
