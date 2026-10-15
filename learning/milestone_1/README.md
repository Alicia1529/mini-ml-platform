# Milestone 1 practice

**Status: Complete.** Hands-on verification was confirmed on 2026-10-14.

This is the small version you built while learning. The repository's `src/` and
`examples/run_batch_inference.py` contain the more complete reference version.
From the repository root, enter this directory and use the repository's Python environment:

```bash
cd learning/milestone_1
../../.venv/bin/python batch_inference.py --count 100 --batch-size 10 --cpus 2 --delay 0.05
```

`--delay` is seconds per message. The script checks that the ordered sequential
and Ray results match, then reports messages per second. It measures Ray startup
separately from task execution. Sleeping makes concurrency visible; it is not a
model inference benchmark. Each Ray task handles one batch and requests one
logical CPU, so `--cpus 4` allows up to four batches to run at once.

## Run the tests

From this directory, choose the test run you need:

```bash
../../.venv/bin/python -m pytest -q test_batch_inference.py
../../.venv/bin/python -m pytest -q -m 'not integration' test_batch_inference.py
../../.venv/bin/python -m pytest -q -m integration test_batch_inference.py
../../.venv/bin/python -m pytest -q test_batch_inference.py::test_worker_error_reaches_driver
```

These run all tests, fast tests without starting Ray, tests with real Ray workers,
and just the worker-error test, respectively.

## Inspect the dashboard

To inspect tasks in the dashboard, first install the optional dependencies from
the repository root with `uv sync --locked --extra dev --extra dashboard`. Then:

```bash
../../.venv/bin/python batch_inference.py --count 1000 --batch-size 10 --cpus 4 --delay 0.1 --dashboard
```

This run takes roughly 100 seconds sequentially, then at least 25 seconds for
the Ray tasks, plus startup overhead. Open `http://127.0.0.1:8265` as soon as
Ray prints its dashboard URL if you want to watch tasks running. The later Enter
prompt keeps the cluster alive so you can inspect finished tasks.
If the browser asks for a token, in another terminal run
`../../.venv/bin/ray get-auth-token | pbcopy` and paste it into the prompt.
Look for `moderate_batch_task` under Jobs and inspect the finished task count.
Press Enter to shut down the local Ray instance.

To observe a worker error directly, run the following. It is expected to exit
with a `RayTaskError` raised by `ray.get()`:

```bash
../../.venv/bin/python - <<'PY'
import ray
from batch_inference import moderate_batch_task

ray.init(address="local", num_cpus=2, include_dashboard=False)
try:
    ref = moderate_batch_task.remote(["hello"], -1)
    print(ray.get(ref))
finally:
    ray.shutdown()
PY
```

The experiment measurements and interpretations are in
`../../docs/learning_journal.md`. Repeat them with your own prediction before
moving to Actors.
