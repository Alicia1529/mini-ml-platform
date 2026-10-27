from time import perf_counter, sleep

import ray
from mini_ml_platform.batch_inference import split_batches
from mini_ml_platform.moderation import Message, ModerationResult, moderate_message

@ray.remote(num_cpus=1)
class ModelWorker:
    def __init__(self, load_seconds: float):
        sleep(load_seconds)
        self.model = moderate_message
        self.batches_processed = 0

    def predict_batch(
        self, batch: list[Message], delay_seconds: float = 0.0
    ) -> list[ModerationResult]:
        results = []
        for message in batch:
            sleep(delay_seconds)
            results.append(self.model(message))
        self.batches_processed += 1
        return results

    def stats(self) -> dict[str, int]:
        return {"batches_processed": self.batches_processed}


if __name__ == "__main__":
    # .venv/bin/python -m mini_ml_platform.actor_inference
    ray.init(address="local", num_cpus=4, include_dashboard=True)

    try:
        texts = ["hello", "SCAM", "nice", "send PASSWORD", "hackathon"]
        messages = [Message(i, text) for i, text in enumerate(texts)]
        batches = split_batches(messages, 2)  # sizes: 2, 2, 1


        load_start = perf_counter()
        worker1 = ModelWorker.remote(load_seconds=3.0)
        worker2 = ModelWorker.remote(load_seconds=3.0)
        workers = [worker1, worker2]
        ray.get([worker1.stats.remote(), worker2.stats.remote()])
        load_seconds = perf_counter() - load_start

        process_start = perf_counter()
        refs = [
            workers[i % len(workers)].predict_batch.remote(batch, 2)
            for i, batch in enumerate(batches)
        ]
        batch_results = ray.get(refs)
        results = [result for batch in batch_results for result in batch]

        processed_seconds = perf_counter() - process_start

        print(results)
        print("Worker stats:", ray.get([worker1.stats.remote(), worker2.stats.remote()]))
        print(f"Actor startup and model load: {load_seconds:.2f} seconds")
        print(f"Batch processing: {processed_seconds:.2f} seconds")
        print([result.flagged for result in results])
    finally:
        ray.shutdown()
