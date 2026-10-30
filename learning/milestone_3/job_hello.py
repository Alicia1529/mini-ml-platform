import ray


@ray.remote(num_cpus=1, max_retries=0)
def simple_task():
    """A simple Ray task that returns a string."""
    return "hello from a Ray Job"


if __name__ == "__main__":
    ray.init()
    try:
        print(ray.get(simple_task.remote()))
    finally:
        ray.shutdown()
