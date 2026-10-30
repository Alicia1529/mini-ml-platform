import ray
from time import sleep


@ray.remote(num_cpus=1, max_retries=0)
def slow_task():
    print("Starting a slow task...")
    sleep(30)  # Simulate a long-running task
    print("Slow task completed.")
    return "hello from a Ray Job"


if __name__ == "__main__":
    ray.init()
    try:
        print(ray.get(slow_task.remote()))
    finally:
        ray.shutdown()
