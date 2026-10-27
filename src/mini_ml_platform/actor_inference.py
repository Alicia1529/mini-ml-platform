from time import sleep

import ray
from mini_ml_platform.batch_inference import run_sequential, split_batches, validate_delay
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


def run_with_actors(
    workers: list,
    messages: list[Message],
    batch_size: int,
    delay_seconds: float = 0.0,
) -> list[ModerationResult]:
    if not workers:
        raise ValueError("workers must not be empty")
    validate_delay(delay_seconds)

    batches = split_batches(messages, batch_size)
    refs = [
        workers[i % len(workers)].predict_batch.remote(batch, delay_seconds)
        for i, batch in enumerate(batches)
    ]
    batch_results = ray.get(refs)
    return [result for batch in batch_results for result in batch]


if __name__ == "__main__":
    # .venv/bin/python -m mini_ml_platform.actor_inference
    ray.init(address="local", num_cpus=4, include_dashboard=True)

    try:
        texts = ["hello", "SCAM", "nice", "send PASSWORD", "hackathon"]
        messages = [Message(i, text) for i, text in enumerate(texts)]

        worker1 = ModelWorker.remote(load_seconds=3.0)
        worker2 = ModelWorker.remote(load_seconds=3.0)

        results = run_with_actors([worker1, worker2], messages, 2, 2.0)
        assert results == run_sequential(messages, batch_size=2)

        print(results)
        print("Worker stats:", ray.get([worker1.stats.remote(), worker2.stats.remote()]))
        print([result.flagged for result in results])
    finally:
        ray.shutdown()
