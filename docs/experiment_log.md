# Experiment log: results and what they taught us

This is a record of **observations**, not performance promises. All timings are
wall-clock measurements on the local Mac. The moderation rule is deterministic;
`sleep()` simulates work, so these results show scheduling and HTTP behavior, not
real model throughput. Keep the command, code/configuration, replica count, and
machine conditions with future measurements so they remain comparable.

For Milestone 1 predictions and the original detailed notes, see the
[learning journal](learning_journal.md). For the remaining hands-on checkpoints,
see the [learning plan](learning_plan.md).

## Milestone 1: sequential Python versus Ray Tasks

Verified practice run on 2026-10-14: 100 seeded messages, batch size 10, 50 ms
of simulated work per message, one CPU requested per Ray task. There were 10
batches, each needing about 0.5 s of simulated work. Sequential and Ray results
matched: **32/100 flagged**. The expected scheduling waves were 10, 3, and 2
with 1, 4, and 8 Ray CPU slots, respectively.

| Ray CPU slots | Sequential | Ray startup | Ray execution | Startup + execution | Ray throughput |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.487 s | 9.198 s | 5.831 s | 15.029 s | 17.1 messages/s |
| 4 | 5.359 s | 9.254 s | 2.177 s | 11.432 s | 45.9 messages/s |
| 8 | 5.371 s | 11.371 s | 1.616 s | 12.988 s | 61.9 messages/s |

**What this shows:** More Ray CPU slots reduced *task execution* time, but a
one-off Ray run was slower after including cluster startup. `num_cpus` is a Ray
scheduling resource, not a limit or guarantee for physical CPU usage. Here,
sleeping tasks hold their slots while doing no CPU computation. The startup-plus-
execution column excludes other command overhead, such as input creation and
shutdown; it is not the full shell-command duration.

In the separate CPU-resource exercise, the requested configurations were
`--ray-cpus 2 --task-cpus 1 --tasks 4 --seconds 0.5` and the same command with
`--task-cpus 2`. Their *ideal sleep-only* schedules are two concurrent tasks
over two waves (at least **1.0 s**) versus one task over four waves (at least
**2.0 s**). Those are predictions, not measured times; Ray startup and task
overhead add to them. Reserving two Ray CPU units does not make a sleeping task
finish twice as fast.

### Batch-size sweep

Same 100 messages, one already-started Ray cluster with 4 CPU slots, four warm-up
tasks, and matching results in every run:

| Delay per message | Batch size | Task count | Ray execution |
| ---: | ---: | ---: | ---: |
| 10 ms | 1 | 100 | 0.326 s |
| 10 ms | 10 | 10 | 0.362 s |
| 10 ms | 50 | 2 | 0.596 s |
| 10 ms | 100 | 1 | 1.183 s |
| 0 ms | 1 | 100 | 0.030 s |
| 0 ms | 10 | 10 | 0.005 s |
| 0 ms | 50 | 2 | 0.003 s |
| 0 ms | 100 | 1 | 0.002 s |

**What this shows:** With a delay, large batches leave CPU slots idle: two
batches cannot occupy four slots. Without a delay, dispatching 100 tiny tasks
costs more than doing the almost-empty work. The 0.326-versus-0.362 s difference
is too small and under-repeated to declare batch size 1 generally faster.

The separate Dashboard exercise used 20 messages, batch size 5, 2 Ray CPU slots,
and 0.1 s delay per message. Four tasks finished; Ray startup took **11.912 s**
and task execution took **1.471 s**. In the failure exercise, three tasks were
submitted, the middle task raised `ValueError`, and the combined `ray.get()`
raised `RayTaskError(ValueError)`. Fetching the other two references afterward
returned `[[False], [True]]`: one failed reference does not erase successful
results stored under other references.

## Milestone 2: Tasks, Actors, and sequential execution

The current [comparison script](../examples/compare_tasks_actors.py) uses 100
messages in 10 batches, 2 Ray CPU slots, 2 Actors, 50 ms of simulated inference
per message, and 1 s of simulated model loading. It measures these phases:

| Path | Simulated load frequency | Timed phases |
| --- | --- | --- |
| Sequential | Once | Load + processing |
| Ray Tasks | Once per batch (10 loads) | Ray startup + Task execution |
| Ray Actors | Once per Actor (2 loads) | Ray startup + Actor setup + first pass; a separate reuse pass has no new load |

The script checks that the three paths return identical results. Actor reuse is
the important distinction: an Actor keeps its loaded state across calls. Tasks
do **not** inherently reload a model; this example makes each Task do so to
expose the cost. The Ray paths execute one after the other in one process, so
their startup-inclusive sums are composed phase measurements, not independent
cold-start trials.

An earlier run printed **Tasks 0.4004 s, Actors 1.8350 s, Sequential 0.0003 s**.
The exact parameters and included setup phases for that historical run were not
recorded, so do not compare those numbers directly with the current script's
1-second-load configuration. The near-zero sequential time was the clue that
the initial example did not charge the same simulated work to every path; the
comparison script was subsequently changed to make the work explicit.

## Milestone 4: Ray Serve HTTP benchmarks

The [Serve endpoint](../src/mini_ml_platform/serve_moderation.py) sleeps **0.2 s
per valid request** and returns the serving replica's PID. In the fixed-replica
experiment, `max_ongoing_requests=1` meant each replica could handle one
in-flight request at a time. The client ran
[`benchmark_serve.py`](../examples/benchmark_serve.py), which POSTs `SCAM`,
checks the response, and measures total run time and each successful request's
client-observed latency. All requests below succeeded.

The first benchmark screenshot showed **20 requests, concurrency 8, 7.857 s,
2.55 successful requests/s**. Its live replica count and warm-up state were not
recorded, so it is a historical observation, not part of the controlled
one-versus-two-replica comparison.

### One fixed replica: vary client concurrency

The single-replica service was verified before measuring. The client sent 20
requests in each run. "Concurrency" is the client's maximum simultaneous
requests; it does not create more Serve replicas.

| Client concurrency | Total time | Throughput | Request p50 | Request p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.229 s | 4.73 requests/s | 211.0 ms | 219.9 ms |
| 2 | 4.551 s | 4.39 requests/s | 422.2 ms | 687.2 ms |
| 8, run 1 | 7.818 s | 2.56 requests/s | 2689.4 ms | 4354.0 ms |
| 8, run 2 | 7.905 s | 2.53 requests/s | 3149.5 ms | 3939.4 ms |
| 8, run 3 | 7.861 s | 2.54 requests/s | 2803.2 ms | 5068.9 ms |

With one replica, requests above its capacity wait. The 20 requests have at
least 20 × 0.2 = **4.0 s** of simulated service work. At concurrency 1, total
time was close to that floor. At concurrency 8, it was consistently about
**7.86 s**, so higher offered concurrency did not improve throughput in this
setup. The reason for the additional high-concurrency overhead was **not
isolated** by this experiment; proxy routing, request scheduling, and local
machine load are possibilities, not proven causes.

### One versus two fixed replicas: same client load

On 2026-11-06, after restarting Serve with `num_replicas=2`, Ray reported two
live `ModerationService` Actors (PIDs **16500** and **16501**). Both PIDs appeared
in responses to eight simultaneous requests. We warmed the two-replica service
with 2 successful requests at concurrency 2, then ran three 20-request trials
at concurrency 8. The earlier one-replica trials used the same 20/8 client load.

| Replicas | Trial | Total time | Throughput | Request p50 | Request p99 | Success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 7.818 s | 2.56/s | 2689.4 ms | 4354.0 ms | 20/20 |
| 1 | 2 | 7.905 s | 2.53/s | 3149.5 ms | 3939.4 ms | 20/20 |
| 1 | 3 | 7.861 s | 2.54/s | 2803.2 ms | 5068.9 ms | 20/20 |
| 2 | 1 | 3.963 s | 5.05/s | 1415.7 ms | 2024.2 ms | 20/20 |
| 2 | 2 | 3.567 s | 5.61/s | 1286.4 ms | 1909.1 ms | 20/20 |
| 2 | 3 | 3.921 s | 5.10/s | 1413.7 ms | 2177.4 ms | 20/20 |

| Replicas | Median total | Median throughput | Median request p50 | Median request p99 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.861 s | 2.54/s | 2803.2 ms | 4354.0 ms |
| 2 | 3.921 s | 5.10/s | 1413.7 ms | 2024.2 ms |

The two-replica median total was about **2.0× faster** (7.861 / 3.921), and
median throughput was about **2.0× higher** (5.10 / 2.54). This is evidence of
both replicas serving work, not a guarantee of linear scaling for other loads.
The ideal sleep-only floor is 4.0 s with one replica and 2.0 s with two; both
observed high-concurrency totals exceeded their respective floors. We did not
measure proxy, queue, and replica processing phases separately. The two
configurations ran at different times, so background load could also differ.

`p50` is the median of **individual client request durations**; `p99` is near
the slowest of these 20 requests. Both include waiting and network/Serve
overhead. Total time measures the whole 20-request run. A run can therefore
have a much larger p50 than 0.2 s while completing multiple requests in
parallel. With only 20 samples, p99 is especially noisy.

### Autoscaling: idle 1 → busy 2 → idle 1

Measured on 2026-11-06 after restoring the deployment's autoscaling settings:
`min_replicas=1`, `max_replicas=2`, `target_ongoing_requests=0.5`,
`upscale_delay_s=1`, `downscale_delay_s=10`, and `max_ongoing_requests=1`.
The service was already running on the local Mac. Before the load, Ray's live
Actor list showed one `ModerationService` replica (PID **17424**). We sent
**100 requests at client concurrency 8**, with the same 0.2 s simulated delay
per request. A separate monitor polled the live Actor list, and the Serve
controller log gave the more precise transition events.

| Event | Local time (PST) | Relative to load | Evidence |
| --- | --- | ---: | --- |
| Load started | 17:27:31.448 | 0 s | Benchmark wrapper |
| Controller chose 1 → 2 | 17:27:44.305 | +12.857 s | Controller log; ongoing requests 3.20 |
| Second replica ready | 17:27:48.123 | +16.675 s | Controller log; new PID **17784** |
| Monitor observed 2 live replicas | 17:27:48.653 | +17.205 s | Actor-list poll: PIDs 17424 and 17784 |
| Load ended | 17:27:58.480 | +27.032 s | Benchmark wrapper; **100/100 succeeded** |
| Controller chose 2 → 1 | 17:28:33.449 | 34.969 s after load ended | Controller log; ongoing requests 0.00 |
| Extra replica stopped | 17:28:35.635 | 37.155 s after load ended | Controller log |
| Monitor observed 1 live replica | 17:28:36.441 | 37.961 s after load ended | Actor-list poll: PID 17424 |

The benchmark itself reported **26.877 s total**, **3.72 successful
requests/s**, **p50 1891.7 ms**, and **p99 3996.9 ms**. The wrapper's 27.032 s
also includes launching the benchmark subprocess and printing its output; use
26.877 s for comparison with other `benchmark_serve.py` runs. This 100-request
run is *not* directly comparable to the earlier 20-request fixed-replica runs.

**What this shows:** Autoscaling did complete the intended 1 → 2 → 1 cycle.
The configured 1- and 10-second delays did not equal the observed wall times:
the controller decided to scale up 12.857 s after load started, and the second
replica took another 3.818 s to become ready. It decided to scale down 34.969 s
after load ended. Actor-list polling detected changes slightly later than the
controller log, so the log timestamps are the better evidence for *when* a
decision or startup occurred. This experiment did not isolate why the
downscale decision took about 35 s; do not treat the configured delay as a
promise that a replica will disappear exactly that many seconds after the last
client request.

### Reproduce the Serve comparison

From the repository root, set `num_replicas` in
`src/mini_ml_platform/serve_moderation.py` to the value being tested and restart
Serve; changing the file alone does not change an already-running service.
In one terminal:

```bash
.venv/bin/serve run mini_ml_platform.serve_moderation:app
```

In another terminal, warm up and then repeat the measured command three times:

```bash
.venv/bin/python examples/benchmark_serve.py --requests 2 --concurrency 2
.venv/bin/python examples/benchmark_serve.py --requests 20 --concurrency 8
```

Confirm the live replica count in the Ray Dashboard **Actors** page and check
that different `replica_pid` values appear in concurrent HTTP responses. The
Dashboard port may change when another Ray cluster is already running. Keep the
request count, client concurrency, simulated delay, and `max_ongoing_requests`
fixed when attributing differences to replica count.

## Checkpoint 8: Ray Jobs SDK success flow

On 2026-11-06, we used the real local Ray Jobs service through
[`MLPlatformClient`](../src/mini_ml_platform/client.py), rather than the mock
used by the unit tests. The final run of
[`submit_job.py`](../learning/milestone_3/submit_job.py) submitted
`job_hello.py` as **`raysubmit_Kayzrzhw4DSjcRcU`**. Its observed statuses were
`PENDING → RUNNING → SUCCEEDED`, and `get_logs()` included
**`hello from a Ray Job`**. The dashboard was at `127.0.0.1:8265`, and the Job
driver connected to that same cluster at `127.0.0.1:6379`.

Three failed attempts were useful diagnostics:

1. Two Ray clusters were running on the same Mac. The newer Serve cluster's
   Dashboard Agent could not bind port **52365**, already held by the older
   cluster's Agent. Its dashboard was accessible at 8266, but Job submission
   returned `500: No available agent to submit job`. A minimal submission
   through Ray's own `JobSubmissionClient` reproduced the same error, ruling out
   the project's wrapper and job script.
2. The older dashboard at 8265 accepted submissions, but one Job driver
   connected to the *newer* cluster's GCS address and could not download the
   working-directory package from the cluster that received the upload. The
   logs showed the address mismatch; this was not a moderation-code failure.
3. After stopping both clusters and starting one clean cluster, submission
   worked but the entrypoint failed with `/bin/sh: python: command not found`.
   The cluster had been launched without `.venv/bin` on `PATH`. Restarting it
   with the project virtual environment on `PATH` made the same Job succeed.

**What this shows:** Dashboard availability alone does not prove its Job agent
can accept submissions. A submitted Job's driver and workers must connect to
the same cluster that received its working-directory package, and the
entrypoint command needs a Python executable in the cluster's environment.
For this local exercise, keeping **one** Ray cluster and starting it from the
project environment removed both sources of confusion. Serve was then
restarted on that same cluster; its HTTP endpoint returned 200.

## Checkpoint 9: failed and stopped Ray Jobs

On 2026-11-07, [`submit_failure.py`](../learning/milestone_3/submit_failure.py)
submitted `ray_retry.py` as **`raysubmit_ZjB2iHCFNGMq3TmD`**. The status
progressed through `PENDING → RUNNING → FAILED`. The logs showed one
`Task attempt`, followed by `RayTaskError(ValueError)` at the Job driver's
`ray.get()` and the original `ValueError: intentional failure` from the Task.
The Task's `retry_exceptions=False` disables retries for this application
exception; its `max_retries=1` setting does not retry this `ValueError`.
Because the driver did not catch the error, its entrypoint exited with code 1
and the **Job** became `FAILED`.

[`stop_agent_demo.py`](../learning/milestone_3/stop_agent_demo.py) submitted
`job_slow.py` as **`raysubmit_TaAUDeXFfMFSgZRg`**. It observed
`PENDING → RUNNING`, requested a stop after 15 seconds, received
`Stop request accepted: True`, and then observed `STOPPED`. The logs contained
`Starting a slow task...` but not `Slow task completed.` The `SIGTERM` stack
trace was the expected result of terminating the running Job driver while it
waited at `ray.get()`; it did not indicate a cluster failure. The accepted stop
request and final `STOPPED` status are separate observations.

## Other recorded observations

- In the first 1,000-message Milestone 1 data check, batch size 50 produced
  **20 batches** and **373 flagged messages**. The sequential and Ray paths
  agreed on the flag count.
- A separate Ray Job submission for 100 messages reported **5.533 s Ray
  execution** and **32/100 flagged**. It includes a different job/runtime
  context from the Serve HTTP measurements, so it is not a Serve comparison.
- A previous question mentioned approximately **5.36 s sequential** and
  **2.93 s Ray**. The exact invocation for the 2.93 s result was not preserved
  here; use the fully specified Milestone 1 table above for comparisons.

## Template for the next run

```text
Date / machine load:
Code commit or uncommitted config:
Server mode / live replica count / max_ongoing_requests:
Messages or HTTP requests / batch size / client concurrency / delay:
Warm-up:
Command:
Results (each trial: success, total, throughput, p50, p99):
What changed from the control:
What the result supports / what remains uncertain:
```
