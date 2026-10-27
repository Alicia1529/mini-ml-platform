from time import perf_counter, sleep

import ray
from moderation import is_unsafe

@ray.remote(num_cpus=1)
class ModelWorker:
    def __init__(self, load_seconds: float):
        sleep(load_seconds)
        self.model = is_unsafe
        self.batches_processed = 0

    def predict_batch(self, batch: list[str], delay_seconds: float = 0.0) -> list[bool]:
        results = []
        for message in batch:
            sleep(delay_seconds)
            results.append(self.model(message))
        self.batches_processed += 1
        return results

    def stats(self) -> dict[str, int]:
        return {"batches_processed": self.batches_processed}


if __name__ == "__main__":
    # PYTHONPATH=learning/milestone_1 .venv/bin/python learning/milestone_2/actor_inference.py
    ray.init(address="local", num_cpus=4, include_dashboard=True)

    try:
        load_start = perf_counter()
        worker1 = ModelWorker.remote(load_seconds=3.0)
        worker2 = ModelWorker.remote(load_seconds=3.0)
        ray.get([worker1.stats.remote(), worker2.stats.remote()])
        load_seconds = perf_counter() - load_start

        process_start = perf_counter()
        first = worker1.predict_batch.remote(["hello", "SCAM", "nice"], 2.0)
        second = worker2.predict_batch.remote(["nice", "send PASSWORD", "hackathon"], 2.0)
        results = ray.get([first, second])
        processed_seconds = perf_counter() - process_start

        print(results)
        print("Worker stats:", ray.get([worker1.stats.remote(), worker2.stats.remote()]))
        print(f"Actor startup and model load: {load_seconds:.2f} seconds")
        print(f"Batch processing: {processed_seconds:.2f} seconds")
    finally:
        ray.shutdown()
