import ray

@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1
        return self.n

    def read(self):
        return self.n

if __name__ == "__main__":
    ray.init(address="local", num_cpus=2)

    try:
        A = Counter.remote()
        B = Counter.remote()

        a_first = A.increment.remote()
        a_second = A.increment.remote()
        b_first = B.increment.remote()

        print("Increment results:", ray.get([a_first, a_second, b_first]))
    finally:
        ray.shutdown()
