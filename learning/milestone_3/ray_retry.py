import ray

@ray.remote(num_cpus=1, max_retries=1, retry_exceptions=False)
def always_fail():
    print("Task attempt", flush=True)
    raise ValueError("intentional failure")


if __name__ == "__main__":
    ray.init(address="auto")
    try:
        ray.get(always_fail.remote())
    finally:
        ray.shutdown()