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
