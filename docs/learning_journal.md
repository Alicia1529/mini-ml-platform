# Milestone 1 learning journal

Fill this in yourself. Make a prediction before measuring; disagreement between
the prediction and observation is useful evidence.

## Environment and baseline

- Date / Mac model / physical or logical CPU count:
- Python and Ray versions (`.venv/bin/python -c 'import sys, ray; print(sys.version, ray.__version__)'`):
- Other laptop workload / power mode:
- Command and seed:
- What happens between `.remote()` and `ray.get()` in my own words:

## Experiment 1: CPU slots

Prediction and reasoning:

| CPUs | Batch size | Delay ms/message | Sequential s | Ray init s | Ray execution s | Ray init + execution s | Ray messages/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 | 10 | | | | | |
| 4 | 50 | 10 | | | | | |
| 8 | 50 | 10 | | | | | |

Observations / surprises / next question:

## Experiment 2: batch size

Prediction and reasoning (including number of tasks):

| CPUs | Batch size | Task count | Delay ms/message | Ray execution s | Ray messages/s |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | | 10 | | |
| 4 | 10 | | 10 | | |
| 4 | 50 | | 10 | | |
| 4 | 250 | | 10 | | |
| 4 | 1000 | | 10 | | |
| 4 | 1 | | 0 | | |
| 4 | 10 | | 0 | | |
| 4 | 50 | | 0 | | |
| 4 | 250 | | 0 | | |
| 4 | 1000 | | 0 | | |

Dashboard observations / overhead hypothesis / evidence:

## Experiment 3: intentional failure

- Prediction:
- Edit and command:
- Worker exception and driver traceback location:
- Did I get a result list? What happened to the other tasks?
- How does the decorator's retry policy affect my interpretation?
- Did I revert the edit and get the tests passing again?

## Explain it back

- When would I choose tasks for a workload?
- What does an ObjectRef represent?
- Why submit all batches before waiting?
- How are logical CPU slots different from physical CPU use?
- Why might Ray be slower? Which timing supports that explanation?
- What would a real model introduce that sleeping cannot capture?
- One question to discuss before moving to Actors:

## Verified practice run (2026-10-14)

These measurements were produced by the assistant on the local Mac. The tables above
remain open for your own predictions and repeat runs. Times vary with machine load.
The workload sleeps; it demonstrates scheduling concurrency, not CPU-bound inference.

With 100 messages, batch size 10, and 50 ms of sleep per message, each batch needs
about 0.5 s. Before running, the expected number of scheduling waves was 10, 3,
and 2 for 1, 4, and 8 Ray CPU slots, respectively.

| Ray CPU slots | Sequential s | Ray init s | Ray execution s | Ray init + execution s | Ray messages/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.487 | 9.198 | 5.831 | 15.029 | 17.1 |
| 4 | 5.359 | 9.254 | 2.177 | 11.432 | 45.9 |
| 8 | 5.371 | 11.371 | 1.616 | 12.988 | 61.9 |

All three runs matched sequential results: 32 of 100 messages were flagged.
Increasing Ray slots reduced execution time here, but Ray startup made every
one-off run slower than the sequential pass overall.

For batch size, one Ray cluster with 4 CPU slots processed the same 100 messages.
Four short warm-up tasks ran before these measurements. Every result matched the
sequential output.

| Delay s/message | Batch size | Tasks | Ray execution s |
| ---: | ---: | ---: | ---: |
| 0.01 | 1 | 100 | 0.326 |
| 0.01 | 10 | 10 | 0.362 |
| 0.01 | 50 | 2 | 0.596 |
| 0.01 | 100 | 1 | 1.183 |
| 0 | 1 | 100 | 0.030 |
| 0 | 10 | 10 | 0.005 |
| 0 | 50 | 2 | 0.003 |
| 0 | 100 | 1 | 0.002 |

With a delay, batch sizes 50 and 100 leave some of the four slots idle. Without
the delay, small batches do so little work that submitting many tasks dominates.
Batch size 1 happened to be slightly faster than 10 in this delayed run; repeat
measurements before treating that small difference as a general result.

The dashboard run used 20 messages, batch size 5, 2 CPU slots, and 0.1 s delay.
The local dashboard returned HTTP 200, and `ray summary tasks` reported four
finished `moderate_batch_task` tasks. Its initialization took 11.912 s, while Ray
execution took 1.471 s. Keep the CLI running at its Enter prompt to inspect it.

For the failure experiment, three tasks were submitted together. The middle task
received an invalid delay of -1. The combined `ray.get` raised
`RayTaskError(ValueError)` instead of returning a partial list. Fetching the two
other references afterward returned `[[False], [True]]`. The integration test
checks this propagation. The decorator sets `max_retries=0` so the failed attempt
is not silently retried.

Ray also warned that `/tmp` is over 95% full. These small experiments completed,
but object spilling may fail if future workloads need disk space.
