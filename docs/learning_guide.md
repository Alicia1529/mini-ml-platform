# Mini ML Platform: learning and review guide

This guide connects the code to the questions explored while building the project.
Read it alongside the [README](../README.md) for commands and the
[experiment log](experiment_log.md) for measured evidence. The four original
milestones and ten final checkpoints are complete as of 2026-11-07.

The examples use Python 3.12 and the repository's pinned Ray 2.59.0. This is a
single-machine learning project: sleeps simulate work and keyword rules simulate
a model. Explanations about measured speed apply to this workload.

## Review route

| Topic | Main question |
| --- | --- |
| [1. Data and correctness](#1-data-and-correctness) | What stays the same across execution modes? |
| [2. Tasks and resources](#2-tasks-and-resources) | What runs in parallel, and when do results arrive? |
| [3. Actors](#3-actors-state-and-reuse) | When is keeping a worker alive useful? |
| [4. Jobs](#4-jobs-a-program-running-on-a-cluster) | How does a complete program reach a running cluster? |
| [5. Serve](#5-serve-requests-replicas-and-autoscaling) | How does a request become an inference result? |
| [6. Measurements](#6-measurements-and-fair-comparisons) | What can the numbers actually tell us? |
| [7. Tests](#7-testing-what-each-layer-proves) | Which behavior have we verified? |
| [8. Python](#8-python-concepts-used-in-this-project) | What do the language features mean? |
| [9. Terminal](#9-terminal-and-environment-reference) | What do these commands do? |
| [10. Troubleshooting](#10-troubleshooting-from-the-experiments) | Which layer failed? |
| [11. Self-check](#11-check-your-understanding) | Can I explain and predict the behavior myself? |

## 1. Data and correctness

Read [moderation.py](../src/mini_ml_platform/moderation.py) first.

`Message` holds an ID and text. `ModerationResult` holds the same message ID, a
boolean `flagged`, and a tuple of matched keywords. These shared types let us
compare results from sequential execution, Tasks, and Actors directly.

The rule case-folds text and checks for substrings `scam`, `password`, and `hack`.
Therefore `SCAM` matches, and `hackathon` also matches. That second result is a
limitation of the toy rule, not a parallelism bug. There is no trained model.

For the texts `['hello', 'SCAM', 'nice', 'send PASSWORD', 'hackathon']`, expected
flags are `[False, True, False, True, True]`. In the main package these texts are
wrapped in `Message` objects; early practice files may use plain strings.

`split_batches(messages, 2)` makes three batches: the first two messages, the next
two, and the final one. It does not execute anything. The sequential runner uses
the same batches to keep its workload comparable with the parallel runners.

Correctness means equal **ordered results**, not merely the same flagged count.
Two implementations could produce the same count while assigning flags to the
wrong messages. A fixed random seed makes generated input reproducible.

## 2. Tasks and resources

Read [batch_inference.py](../src/mini_ml_platform/batch_inference.py).

- `@ray.remote(...)` defines work that Ray can execute in a worker process.
- `moderate_batch_task.remote(batch, delay)` submits work and returns an
  `ObjectRef`, a reference to a future result. Work can start without `ray.get()`.
- `ray.get(ref)` waits if necessary and returns the value, or raises the task's
  error. Calling it does not initiate the task.
- `ray.get(refs)` returns results in the order of the supplied references, even
  if tasks finish in a different order.

Submitting every batch first lets execution overlap. Submitting one batch and
immediately waiting for it inside the loop would serialize that loop. A driver
must remain alive long enough for its work; “get is not required to start” does
not mean a script can exit immediately and reliably leave ordinary tasks running.

### Cluster CPUs versus task CPUs

`ray.init(address='local', num_cpus=4)` creates a local runtime advertising four
logical CPU resources. `@ray.remote(num_cpus=1)` requests one of those resources
per running task. For identical tasks in this CPU-only example:

```text
maximum simultaneous tasks ≈ floor(cluster CPU resources / CPUs per task)
4 / 1 → 4 tasks
4 / 2 → 2 tasks
```

A resource request controls scheduling. It does not give a Python function extra
threads or make it twice as fast. Increasing `num_cpus` beyond the machine's
hardware does not create cores; CPU-bound processes will compete for real CPU
time. Sleep-based tasks may still overlap because they mostly wait.

Ten equal batches with four slots need roughly three waves: 4 + 4 + 2. Ray does
not impose a wave barrier: any free slot can start the next pending task. With
unequal batch durations, that distinction matters.

Batch size trades scheduling overhead against parallelism. Very small batches
create many submissions; very large batches may leave too few tasks to occupy
all slots. `delay` in the main runner is **per message**, so changing batch size
does not change the intended total simulated work.

### Startup, execution, and failures

Cold total time includes runtime initialization plus execution. Warm execution
excludes initialization. The recorded four-CPU trial took 2.177 s to execute but
11.432 s including startup; sequential took 5.359 s. Those statements are
consistent: the faster execution lost its advantage during startup.

Validation inside a remote function runs in the worker. If `validate_delay`
raises there, Ray stores the failure with the result; the driver encounters it
at `ray.get()`. Input validation done in the driver before submission would fail
immediately instead. The first Task exercise disables retries to expose failures.

## 3. Actors: state and reuse

Read [actor_inference.py](../src/mini_ml_platform/actor_inference.py) and
[compare_tasks_actors.py](../examples/compare_tasks_actors.py).

An Actor is an instance living in a worker process. Its constructor initializes
state once; later method calls use that same instance. `ModelWorker` keeps its
model function and `batches_processed` counter. `stats()` makes this state
observable and can act as a readiness check after construction.

These Actors use synchronous methods. One Actor processes its methods serially
in this exercise; four cluster CPU slots do not turn one Actor into four workers.
Create multiple Actors and distribute batches to obtain parallelism. The runner
uses round-robin dispatch, then restores input ordering by collecting references
in submission order. It is not an adaptive load balancer.

| Mode in the comparison | Simulated loading | Reuse |
| --- | --- | --- |
| Sequential | Once before all batches | One local model function |
| Tasks | Once per batch task | No model cache in this example |
| Actors, first pass | Once per Actor during setup | Each Actor handles multiple batches |
| Same Actors, second pass | No new constructor/load | Existing instances and state remain |

Per-task loading is a deliberate choice in this comparison, not a universal Ray
requirement. A different Task implementation could cache state. Likewise,
sequential Python can load once and reuse a model. Actors make persistent remote
state explicit; they do not uniquely enable reuse.

The comparison's `expected` is the sequential result used to check correctness,
not another execution framework. Actor setup includes process/construction costs;
Actor reuse timing excludes those costs. Always name which costs you compare.

Explicit `num_cpus=1` Actors hold their reserved CPU resources for their lifetime.
The example runs Tasks before constructing Actors. Creating enough idle Actors
to reserve every slot, then waiting for additional CPU-requiring Tasks, can stall
that work even though the laptop's physical CPU utilization looks low.

## 4. Jobs: a program running on a cluster

Read [client.py](../src/mini_ml_platform/client.py) and
[submit_job.py](../learning/milestone_3/submit_job.py).

A Task is a unit of work inside a Ray program. A submitted Job runs a whole
entrypoint program, whose driver can create many Tasks or Actors. The submission
client communicates with the Jobs API on an already-running cluster; it does not
start a cluster for you.

```text
local submission script
  → Jobs API on :8265
  → upload working_dir / prepare runtime environment
  → start entrypoint Python driver
  → driver connects to Ray and submits work
  → final status and logs available through Jobs API
```

### Paths, resources, and IDs

`working_dir` is the directory uploaded to the cluster. With
`working_dir='learning/milestone_3'`, the entrypoint is `python job_hello.py`,
relative to that uploaded root. A file outside it, an unsaved editor buffer, or
an incorrect filename will not magically appear on the worker.

Uploading source is not equivalent to installing every dependency. Our local
cluster shares the development environment. Remote deployment would need an
explicit, reproducible runtime/dependency strategy.

`entrypoint_num_cpus` reserves CPU resources for the driver. Child Tasks and
Actors request their own resources. On a two-slot cluster, a one-slot driver
leaves only one slot available to one-slot child Tasks. Reserving every slot for
a driver that waits for CPU-requiring children can prevent progress.

A supplied `submission_id` identifies the submission. Generate one ID with
`uuid4().hex` and submit it twice to test rejection of a duplicate. Generating a
new ID on each call would submit two distinct jobs. Duplicate rejection is not
an exactly-once guarantee or an automatic replay of the original result.

The Dashboard also displays a Core Job ID such as `04000000`. That is different
from the Jobs submission ID. A directly launched driver can appear on the
Dashboard with no submission ID; do not use its Core ID as a Jobs API stop ID.

### Lifecycle and errors

| State | Meaning in our demonstrations |
| --- | --- |
| `PENDING` | Submission accepted; setup/startup has not completed |
| `RUNNING` | Entrypoint is executing, possibly waiting for tasks |
| `SUCCEEDED` | Entrypoint finished successfully |
| `FAILED` | Setup or entrypoint failed |
| `STOPPED` | Job was stopped through the Jobs lifecycle |

The scripts poll with a `monotonic()` deadline, then print logs. Submission success
is not execution success. A polling timeout also does not itself stop the Job.

In [ray_retry.py](../learning/milestone_3/ray_retry.py), `max_retries=1` is paired
with `retry_exceptions=False`. The intentional `ValueError` is an application
exception, so it is not retried under that configuration. It becomes a
`RayTaskError` at the driver's `ray.get()`. Because the driver does not catch it,
the program exits unsuccessfully and the Job becomes `FAILED`.

In [stop_agent_demo.py](../learning/milestone_3/stop_agent_demo.py), a successful
`stop_job()` request is followed by polling until `STOPPED`. Its return value is
not proof that termination has already completed. A SIGTERM stack trace while
the driver waits in `ray.get()` is consistent with deliberate termination. Check
who requested the stop and the final status; the trace alone does not prove a
cluster crash.

## 5. Serve: requests, replicas, and autoscaling

Read [serve_moderation.py](../src/mini_ml_platform/serve_moderation.py).

A client sends JSON over HTTP to the Serve proxy. The proxy routes it to a replica
of `ModerationService`. The handler parses and validates input, simulates 0.2 s
of inference, applies the shared moderation rule, and returns JSON.

| Input | Response |
| --- | --- |
| `{"text":"SCAM"}` | HTTP 200, flagged true |
| `{"text":"hello"}` | HTTP 200, flagged false |
| Missing `text`, non-string `text`, or non-object JSON | HTTP 400, validation error |
| Malformed JSON, e.g. `{"text":` | HTTP 400, invalid JSON |

Before the JSON parsing exception was handled, malformed JSON produced HTTP 500.
The fix makes an invalid client payload a 400 response instead of an unhandled
server error. `request.json()` parses bytes; it does not guarantee the result is
an object or that `text` has the correct type.

Replicas are Actors, but not every Actor in the Dashboard is a model replica.
Serve also uses controller and proxy Actors. Two replicas plus those components
can therefore appear as four Actors. `replica_pid` helps observe routing locally.

The handler uses `asyncio.sleep`, which yields to the event loop while waiting.
This is different from blocking `time.sleep`. However, the deployment explicitly
sets `max_ongoing_requests=1`, so each replica accepts one ongoing request at a
time in this exercise. `async` alone is not unlimited inference capacity.

Current autoscaling bounds are one to two replicas, with target ongoing requests
0.5, upscale delay 1 s, and downscale delay 10 s. These are controller settings,
not promises that a new replica will be ready in one second. Detection, control
loops, scheduling, startup, and readiness contribute to observed time. Our log
records when these events were observed; it does not isolate every cause.

A brief burst can finish before extra capacity is ready. Longer sustained load
is better for observing scaling. Code edits also require redeployment; inspect
the running deployment instead of assuming the saved file describes live state.

## 6. Measurements and fair comparisons

Read [benchmark_serve.py](../examples/benchmark_serve.py) and the
[recorded experiments](experiment_log.md).

| Metric | Definition in this benchmark |
| --- | --- |
| Requests | Number of attempted HTTP requests |
| Concurrency | Maximum client thread workers sending requests |
| Total time | Wall time around the executor and all requests |
| Throughput | Successful requests divided by total seconds |
| p50 | Median of successful request durations |
| p99 | Estimated 99th percentile of successful request durations |

A success requires HTTP 200, a JSON object, and `flagged` equal to true for the
benchmark's `SCAM` payload. HTTP/connection errors and timeouts count as failures.

Latency starts **when a thread begins `send_one`**, not when all work is initially
enqueued. It includes network/HTTP and service waiting, but excludes waiting for
an available executor thread. Failed requests are excluded from p50/p99, so
always read failure counts too. With only 20 requests, p99 is a small-sample
estimate, not a robust production tail-latency measurement.

### Why more concurrency can increase latency

One replica doing 0.2 s of work has an ideal capacity of about five requests per
second before overhead. Sending eight requests concurrently does not give that
replica eight times the capacity: requests can wait. Twenty sequential requests
have an ideal service-work total of four seconds; twenty concurrent requests to
one serial replica still require roughly four seconds of service work.

That ideal assumes negligible overhead and a continuously busy server. Our
one-replica concurrency-eight runs took about 7.9 s, versus about 4.2 s at
concurrency one. Queueing explains higher individual latency, but **does not by
itself explain all the extra total time**. The experiment did not isolate the
additional routing, scheduling, HTTP, or runtime costs. Treat these as possible
contributors, not a measured diagnosis.

Two fixed replicas reduced median total time to 3.921 s at concurrency eight.
This supports a capacity benefit for this workload; it does not establish a
universal two-times ML speedup. Current autoscaling behavior also differs from a
benchmark with a fixed, warmed replica count.

### A repeatable experiment checklist

1. Keep input, seed, message count, per-message delay, and correctness checks fixed.
2. Change one setting: batch size, CPU slots, concurrency, or replica count.
3. Record the commit, command, Python/Ray versions, and actual running configuration.
4. Separate cold startup/model load from warm execution. Label both if reporting both.
5. Warm up consistently, repeat trials, and report variation rather than a single best run.
6. Record failures, background load, and whether autoscaling occurred.
7. Write a prediction, observation, and remaining uncertainty separately.

## 7. Testing: what each layer proves

| Layer | What it checks | What it cannot establish alone |
| --- | --- | --- |
| Pure unit tests | Splitting, validation, moderation results | Distributed scheduling |
| Mocked client tests | Wrapper arguments and forwarded return values | Cluster reachability or real Job success |
| Ray integration tests | Task/Actor results, state, and error propagation | HTTP serving behavior |
| Live Serve tests | HTTP status and response behavior | Sustained load capacity |
| Manual Job lifecycle runs | Real submission, success, failure, stopping | Every failure mode or remote deployment |

Recorded final validation: **32 passed, 5 skipped** in the full suite without a
live Serve endpoint; separately **5 live HTTP tests passed**. These were local
runs on 2026-11-07, not evidence of multi-node testing.

`@pytest.fixture(scope='module')` provides setup shared by tests in that module.
A test parameter named `ray_runtime` asks pytest to resolve and execute that
fixture. You do not call it manually. A fixture using `yield` performs setup
before `yield`, hands control to tests, and performs teardown afterward.

`@pytest.mark.integration` labels a test. `python -m pytest` asks Python to run
the pytest module; pytest's own `-m integration` selects a marker expression.
Registering a marker does not skip tests automatically. Our Serve fixture skips
when `SERVE_URL` is absent because it needs a live HTTP endpoint.

In [test_client.py](../tests/test_client.py):

- `jobs = Mock(spec=JobSubmissionClient)` creates a stand-in with attribute names
  based on that class. It is not a real connection; `spec` does not fully enforce
  method argument signatures.
- `factory = Mock(return_value=jobs)` is a separate stand-in for construction:
  calling the factory returns `jobs`.
- `monkeypatch` replaces the class name where our client module looks it up.
- `factory.assert_called_once_with(address)` checks constructor use.
- `jobs.submit_job.assert_called_once_with(...)` checks submission arguments.

The path `scripts/job moderation.py` in that test is deliberate fake data used to
verify shell quoting of a filename with a space. No such file needs to exist
because the test never submits a real Job.

## 8. Python concepts used in this project

### Imports and public API

Installing the project makes `mini_ml_platform` importable from the environment.
The filesystem layout is `src/mini_ml_platform`, but imports use the package name,
not `src` or an absolute file path.

```python
from mini_ml_platform.client import MLPlatformClient
```

The package's `__init__.py` re-exports the same class, allowing:

```python
from mini_ml_platform import MLPlatformClient
```

Checkpoint 7 made that shorter public import available. It did not implement a
new client or make every submodule automatically appear at package level.
Inside the package, `from .client import MLPlatformClient` is a relative import.

### Data classes, attributes, and decorators

`@dataclass` generates methods such as `__init__`, `__repr__`, and equality from
annotated fields. For `ModerationResult`, you can construct an instance with
`message_id`, `flagged`, and `matched_keywords` instead of manually assigning
those attributes in an initializer. `frozen=True` prevents ordinary field
reassignment; it is not recursive freezing of every possible nested object.

`tuple[str, ...]` means a tuple containing any number of strings.
`submission_id: str | None = None` means the argument may be a string or `None`,
and omission supplies `None`. Type hints do not automatically validate runtime
input, which is why our HTTP handler checks types explicitly.

A decorator transforms or configures the object defined immediately below it.
Here, `@ray.remote`, `@serve.deployment`, `@dataclass`, and pytest decorators serve
different purposes; `@` does not always mean remote execution.

A normal attribute stores a value. `@property` lets attribute-style access invoke
a method, often to compute or validate a value. `@classmethod` receives the class
as `cls`, often for alternative constructors. Neither is required merely to make
a class importable, and neither is central to our current worker implementation.

### Timers and output

Use `time.time()` for wall-clock timestamps. Use `monotonic()` for deadlines
because it does not go backward with wall-clock adjustments. Use `perf_counter()`
for measuring elapsed durations. Their absolute origins are not useful as calendar
times; subtract two readings from the same clock.

`print(..., flush=True)` requests that Python flush its output buffer immediately.
Without it, visibility depends on buffering: terminal output is often line-buffered,
while redirected output can be block-buffered. There is no fixed “flush after N
seconds” rule. Ray's log transport can still add delay after Python flushes.

`if __name__ == '__main__':` runs the entrypoint when a file is executed directly,
but not when imported as a module. It prevents demonstration prints and startup
code from unexpectedly running during imports or tests.

## 9. Terminal and environment reference

Run project commands from the repository root unless an exercise says otherwise.

| Command / syntax | Meaning |
| --- | --- |
| `uv sync --locked --extra dev --extra dashboard` | Sync the locked project environment with both optional extras |
| `.venv/bin/python script.py` | Execute directly with that environment's interpreter |
| `uv run python script.py` | Let uv prepare/select the project environment, then run Python |
| `uv python ...` | Manage Python installations; not the command for executing scripts |
| `python -m pytest` | Run pytest using this Python interpreter |
| `RAY_AUTH_MODE=token command` | Set an environment variable for that command |
| `export PATH="$PWD/.venv/bin:$PATH"` | Make this environment's executables discoverable by child processes |
| `command > output.txt` | Redirect stdout, replacing the file's previous contents |
| `command >> output.txt` | Append stdout to the file |
| `command 2>&1` | Send stderr to the current stdout destination |
| `a \| b` | Pipe stdout from program a into stdin of b |
| `wc -l file` | Count newline characters, commonly used to count lines |
| `awk '{print $1}' file` | Print the first whitespace-separated field of each line |

`python - <<'PY'` reads Python code from standard input until a line containing
`PY`. The quoted delimiter prevents shell expansion within that block. The dash
means “read the program from stdin.” A shell continuation backslash must be the
last character before the newline; backslash followed by a space escapes the
space instead, which caused one of our malformed Job entrypoints.

A typical JSON request:

```bash
curl -i http://127.0.0.1:8000/ \
  -H 'Content-Type: application/json' \
  -d '{"text":"SCAM"}'
```

`-H` sets a header; `-d` supplies the body and defaults to POST; shell single quotes
preserve the JSON's double quotes. `-i` includes response headers. `-sS` hides the
progress meter but shows errors. `-o /dev/null` discards the response body, which
explains a successful request appearing to have “no response.”

`seq 1 8 | xargs -P 8 -I{} curl ...` feeds eight lines to `xargs`, which launches
one command per line with at most eight commands running concurrently. `{}` is
the replacement placeholder; even when the curl command does not use it, each
input line triggers an invocation.

For shell `time`, `real` is elapsed wall time; `user` is CPU time in user code;
`sys` is CPU time in the kernel for the measured processes. They are not direct
measurements of a remote server's CPU usage. Waiting for HTTP can make `real`
large while `user` and `sys` remain small.

For Vim: press Escape, then `:w` to save, `:q` to quit, `:wq` to save and quit,
or `:q!` to discard unsaved changes and quit. `:set number` displays line numbers,
`:42` jumps to line 42, and `:set expandtab tabstop=4 shiftwidth=4 softtabstop=4`
uses four-space indentation. Put settings without the initial colon in `~/.vimrc`
for future sessions. `E212` can mean the destination directory does not exist or
is not writable; creating a new buffer does not create its parent directories.

## 10. Troubleshooting from the experiments

A cluster consists of processes such as the scheduler, object store components,
workers, and Dashboard services. Those processes may have threads. `ps` lists
processes; searching only for a specific string can miss processes with different
command names. `lsof` identifies which process owns an open file/socket.

Read-only checks on macOS:

```bash
lsof -nP -iTCP:8265 -sTCP:LISTEN
lsof -nP -iTCP:8000 -sTCP:LISTEN
ps -ax -o pid,ppid,command | grep -E 'raylet|gcs_server|dashboard|serve'
```

A listening socket proves a process is listening, not that the whole cluster is
healthy. A Dashboard page loading successfully does not prove the Job agent can
launch an entrypoint.

| Symptom | What to check |
| --- | --- |
| Connection refused / wrong port | Is the intended cluster running? Jobs uses 8265 here; Serve uses 8000 |
| Dashboard asks for token | Get the current local token with `.venv/bin/ray get-auth-token`; paste into Dashboard |
| `No available agent to submit job` | Inspect cluster/agent health and conflicting old local clusters; this is not proof of a script bug |
| `python: command not found` in Job logs | The environment used to start the cluster must expose Python on PATH |
| `can't open file` | Saved filename and entrypoint path relative to uploaded working_dir |
| Actor/Task work waits indefinitely | Check logical reservations, including idle Actors and the driver |
| Malformed JSON gives 500 | Confirm current handler catches JSON parsing errors and is redeployed |
| Unexpected replica count | Check the running deployment and autoscaling, not only the editor |
| SIGTERM after stop request | Check final Job status; deliberate stopping can produce termination traces |

For a clean local reset, stop only when you intend to terminate all local Ray
workloads: `.venv/bin/ray stop`, then follow the README's cluster startup steps.
`ray.shutdown()` disconnects the calling driver; it does not shut down a separately
started shared cluster. An explicitly local runtime and a persistent cluster have
different lifetimes. Use one intended cluster for the Jobs/Serve exercises.

## 11. Check your understanding

Try answering before opening each explanation. For a practical review, change
one parameter, predict the outcome, and add your observation to the experiment log.

<details>
<summary>1. Does a Task run if I do not immediately call ray.get()?</summary>

Yes. `.remote()` submits it. `ray.get()` waits for and retrieves its result or
failure. The driver still needs an appropriate lifetime; exiting immediately is
not a reliable background-job strategy.
</details>

<details>
<summary>2. Ten batches, four CPU slots, one CPU per task: how many waves?</summary>

Roughly three for equal-duration batches: 4 + 4 + 2. There is no actual global
wave barrier; slots are reused as individual tasks finish.
</details>

<details>
<summary>3. Why can requesting two CPUs per task make the run slower?</summary>

It can halve the number of simultaneous tasks without changing the function's
speed. Resource reservation does not automatically parallelize the function.
</details>

<details>
<summary>4. Why can sequential be fastest, and why does it have a load cost?</summary>

Small work can cost less than Ray startup, scheduling, and transport. A sequential
model also needs initialization, so the simulated model comparison loads it once.
Compare warm execution separately from cold total time.
</details>

<details>
<summary>5. Why does one Actor take the same time with one or four cluster CPUs?</summary>

Our synchronous Actor processes its methods serially. More cluster resources do
not create additional Actor instances or concurrent method execution.
</details>

<details>
<summary>6. Why does a worker ValueError appear at the driver's ray.get()?</summary>

The exception occurred in a different process. Ray associates it with the result
reference and propagates it when the driver retrieves that result. An unhandled
driver exception can then fail the whole submitted Job.
</details>

<details>
<summary>7. Why do the mock tests pass even if no cluster is running?</summary>

They replace the Jobs client with a controlled stand-in and check wrapper
behavior. Real cluster communication is verified separately through lifecycle
exercises. Passing mock tests is not proof of cluster health.
</details>

<details>
<summary>8. Are client concurrency, replicas, and CPU slots the same setting?</summary>

No. Concurrency controls client requests in flight; replicas are service instances;
CPU slots are scheduler resources. They interact but do not imply equal capacity.
</details>

<details>
<summary>9. Does higher p50 explain why total time doubled?</summary>

Not by itself. Queueing can increase request latency with the same total service
work. Extra total time requires examining overhead, idle gaps, contention, and
other runtime behavior; our measurements did not isolate those causes.
</details>

<details>
<summary>10. What did checkpoint 7 change, and is the project complete?</summary>

It re-exported MLPlatformClient at package level for a shorter public import.
The four-milestone local learning project and ten final checkpoints are complete.
Production deployment and real ML inference remain separate future work.
</details>

## Further reading

Use the pinned project code when reproducing experiments; upstream “latest”
documentation may describe a newer Ray release.

- [Ray Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html)
- [Ray Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray resource scheduling](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Ray Jobs SDK](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
