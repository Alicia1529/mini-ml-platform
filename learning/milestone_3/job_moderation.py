"""Run batch moderation inside an existing Ray cluster."""

import ray

from mini_ml_platform.batch_inference import run_ray
from mini_ml_platform.moderation import generate_messages


def main() -> None:
    try:
        ray.init(address="auto")
        messages = generate_messages(100, seed=42)
        ray_results = run_ray(messages, batch_size=10, delay_seconds=0.01)
        print(f"Flagged: {sum(result.flagged for result in ray_results)}/{len(messages)}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
