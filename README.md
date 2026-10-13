# Mini ML Platform: Milestone 1

A small, local learning project for a software engineer exploring ML infrastructure.
Generate 1,000 synthetic chat messages, apply a deterministic toy moderation rule,
and compare sequential Python with Ray Tasks on your Mac. Only Milestone 1 is
implemented. Actors, the Jobs SDK, and Serve are future work.

**This is a concurrency demonstration, not a realistic ML inference benchmark.**
`time.sleep()` simulates latency without doing model computation. Its scaling does
not predict CPU/GPU inference performance, and the keyword rules are not suitable
for real content moderation. There are no external moderation calls.

## Run locally

From this repository's root, use `uv` (already installed on the development Mac):

```bash
uv sync --locked --extra dev
.venv/bin/python -m pytest
.venv/bin/python examples/run_batch_inference.py
```

`.python-version` selects Python 3.12; `uv` can install it if needed. The default
demo uses 1,000 messages, batches of 50, four logical CPUs, and 10 ms of delay per
message. Expect several seconds of Ray startup plus roughly ten seconds or more
for the sequential pass. Ray 2.59.0 is pinned and `uv.lock` records the dependency
resolution. No GPU, Docker, Kubernetes, or cloud account is needed.

Alternatively, with Python 3.12 already installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
python examples/run_batch_inference.py
```

The pip path uses the dependency ranges in `pyproject.toml`; use the locked `uv`
installation for the same resolved versions. Run `.venv/bin/python` directly after
syncing: Ray detects `uv run` and may package the project and recreate environments
for workers, adding setup costs unrelated to this local exercise. Direct execution
also avoids an interactive terminal job-control issue observed with `uv run`.
Modern Ray does not support the macOS
development environment's old default Python 3.8. Use native Python on Apple
Silicon. If an older Intel Mac has no wheel for this Ray release, use a Ray/Python
combination supported by the [installation documentation](https://docs.ray.io/en/latest/ray-overview/installation.html).

Configuration examples:

```bash
.venv/bin/python examples/run_batch_inference.py --batch-size 25 --num-cpus 4 --delay-ms 5
.venv/bin/python examples/run_batch_inference.py --mode sequential
.venv/bin/python examples/run_batch_inference.py --mode ray
.venv/bin/python examples/run_batch_inference.py --messages 1000 --seed 42 --delay-ms 0
.venv/bin/python examples/run_batch_inference.py --help
```

`--mode both` is the default and checks complete result equality, including IDs,
flags, matched keywords, and ordering. Single-mode runs only execute that mode.
Counts, CPU slots, and batch size must be positive; delay must be finite and
non-negative. The library also handles empty inputs, though the benchmark CLI
requires at least one message. `--num-cpus` only affects Ray.

## Read the code in this order

1. `src/mini_ml_platform/moderation.py`: immutable messages/results, seeded data,
   and a pure moderation function. Matching is case-insensitive substring matching
   for `scam`, `password`, and `hack`. For example, `hackathon` is also flagged;
   that limitation is intentional and tested.
2. `src/mini_ml_platform/batch_inference.py`: batching, simulated work, and the
   two execution paths. Both paths call the same `moderate_batch` function.
3. `examples/run_batch_inference.py`: configuration, cluster lifetime, timing,
   result validation, and optional dashboard inspection.
4. `tests/test_batch_inference.py`: rules, batching boundaries, validation, and
   integration checks using real local Ray processes.

The data flow is:

```text
Seeded messages → batches → sequential calls → ordered results
                         ↘ Ray task per batch → ObjectRefs → ray.get → ordered results
                                                        compare both result lists ↑
```

The driver owns `ray.init()` and `ray.shutdown()`; the library requires an already
initialized runtime. This makes the CPU configuration and lifetime visible while
letting tests reuse a runtime. `address="local"` explicitly creates a fresh local
instance. Shutdown happens in `finally`, including when tasks fail.

## What you are learning about Ray

A **Ray Task** is an asynchronous function invocation running in a worker process.
Tasks suit independent units of work: here, moderating one batch needs no results
from other batches. Ray runs multiple workers on this one machine, using the
same task abstraction that can schedule across a cluster.

`@ray.remote(num_cpus=1, max_retries=0)` turns our Python function into a Ray remote
function and specifies its resource/retry options. Decorating it does not execute
the function. Calling `moderate_batch_task.remote(batch, delay)` submits a task
and returns an **ObjectRef**, a reference to its eventual result, rather than the
result list itself. Arguments and results cross process boundaries and must be
serializable. [Ray Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)

`ray.get(refs)` waits until the referenced results are available, returns them in
the order of the input references, or raises if a task failed. Completion order
can differ from submission order; our result order still follows the input.
We submit every batch first and then call `ray.get()` once. Putting a blocking
`ray.get()` immediately after each submission would prevent those batches from
overlapping. [Ray objects](https://docs.ray.io/en/latest/ray-core/objects.html)

`ray.init(num_cpus=4)` gives the local scheduler four logical CPU resources. With
one CPU requested per task, at most four of these batch tasks can execute at once.
Additional tasks wait for resources; a finished task frees its slot for another.
A logical CPU is scheduling capacity, not a pinned physical core or an OS CPU
usage limit. A sleeping task still occupies its requested slot. Ray also runs
system processes beyond your worker tasks.
[Ray resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)

The task decorator explicitly disables retries for this first exercise, so one
failed attempt stays visible. Ray supports separate retry policies for system
failures and application exceptions; we will study those when building job
lifecycle behavior. [Task fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html)

## Interpret the measurements

The CLI reports wall-clock time using `perf_counter()` and throughput as
`message_count / elapsed_seconds`:

| Measurement | Includes |
| --- | --- |
| Sequential execution | Batching, calls, sleeps, and result assembly |
| Ray initialization | The `ray.init()` call |
| Ray execution | Batching, task submission, scheduling, serialization, waits, and result assembly |
| Ray init + execution | Sum of those two Ray measurements |

Input generation, correctness checking, dashboard waiting, and shutdown are outside
these measurements. Ray execution may still include first-use worker/import costs;
this is a fresh-runtime measurement, not a warmed steady-state benchmark. The
init-plus-execution row is not total command duration. Speedup is sequential time
divided by the corresponding Ray time; a value below 1 means Ray took longer.

Why can Ray lose? Process startup, scheduling, and data movement have costs.
Tiny tasks may do too little work to amortize them. Very large batches can leave
too few independent tasks to use available slots. For this bounded 1,000-message
demo we submit all batches and keep all results in memory; a large production
pipeline would need bounded in-flight work and streaming output.

The per-message delay means changing batch size doesn't change total simulated
work. It does not simulate vectorized model batching, model loading, or actual
CPU computation. Laptop load and OS sleep timing also affect results. Repeat a
configuration a few times and compare typical measurements, rather than treating
one run as a benchmark. We test correctness, not a machine-dependent speedup.

## Inspect the Ray Dashboard

The dashboard needs Ray's optional dependencies. Core-only installation stays small:

```bash
uv sync --locked --extra dev --extra dashboard
.venv/bin/python examples/run_batch_inference.py --mode ray --dashboard --delay-ms 30
```

For pip, install `python -m pip install -e '.[dev,dashboard]'` and run the script
with the same arguments. Open the dashboard URL printed by the script, normally
[http://127.0.0.1:8265](http://127.0.0.1:8265), while tasks run. It is bound to
localhost. If Ray asks for an authentication token, in a second terminal run:

```bash
.venv/bin/ray get-auth-token | pbcopy
```

Paste into the dashboard's token prompt. This uses the token Ray generated for
local development; no authentication settings need to be changed.
[Ray token authentication](https://docs.ray.io/en/latest/ray-security/token-auth.html)

Under **Jobs**, select this driver and inspect its tasks; look for
`moderate_batch_task`, pending/running/finished states, durations, worker IDs, and
logs. Under **Cluster**, inspect workers and resource usage. Task state reporting
can lag briefly. The UI's Jobs view also includes ordinary Python drivers; this
project has not implemented Ray Jobs submission.

The script waits for Enter after a successful run, keeping the dashboard alive.
Press Enter to shut down, or Ctrl-C to interrupt and clean up. In a noninteractive
shell, EOF ends the wait. Without `--dashboard`, Ray shuts down immediately after
the comparison. Detailed metrics graphs may require additional monitoring
services; task inspection does not require setting those up for this milestone.
[Dashboard guide](https://docs.ray.io/en/latest/ray-observability/getting-started.html)

You can also inspect task counts from a second terminal while the dashboard run
is waiting (use the printed URL if the port differs):

```bash
RAY_AUTH_MODE=token .venv/bin/ray summary tasks --address http://127.0.0.1:8265
```

## Tests

```bash
.venv/bin/python -m pytest                    # unit tests and real Ray integration tests
.venv/bin/python -m pytest -m 'not integration' # quick tests without starting a cluster
.venv/bin/python -m pytest -m integration      # real worker processes, including error propagation
```

The integration tests cover empty input, uneven final batches, one-message batches,
oversized batches, and all 1,000 messages. They compare complete ordered results.
An independent expected-output test checks the moderation rule so parity alone
cannot hide the same rule bug in both runners.

## Your three experiments

Write a prediction in `docs/learning_journal.md` before each run, then record what
you observe. Keep the dataset/seed fixed. You own the edits for these experiments.

1. **CPU slots: 1 → 4 → 8.** Keep 1,000 messages, batch size 50, and delay 10 ms.
   Run the default comparison with `--num-cpus 1`, then `4`, then `8`. Predict
   the execution and startup-inclusive speedups separately. Where does the time
   go? Does changing logical slots tell you how much physical CPU is being used?
2. **Batching.** Fix `--num-cpus 4`; try `--batch-size 1`, `10`, `50`, `250`,
   and `1000`. Repeat with `--delay-ms 0`. Predict task counts before running.
   Which configurations keep workers busy? How can the dashboard and zero-delay
   run help distinguish scheduling costs from simulated work?
3. **A worker failure.** In `moderate_batch_task`, before `return moderate_batch(...)`,
   add your own `raise RuntimeError("intentional failure")` when the batch contains
   message ID 123. Run with `--mode ray` so the failure happens in a Ray worker.
   Predict where the driver notices it, whether a partial result list is returned,
   and what other submitted tasks might be doing. Read the traceback and the retry
   setting; don't change the retry policy yet. Remove the intentional failure and
   run the tests to restore the baseline.

Bring back your prediction, measurements, and one surprise. Before Milestone 2,
explain why we submit before waiting, what one CPU slot means, and how batch size
changes the amount of schedulable work. We stop here until you have experimented.
