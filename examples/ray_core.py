import ray
import time
ray.init()

@ray.remote
def f(x):
    time.sleep(1)
    return x * x


def f2(x):
    time.sleep(1)
    return x * x


futures = [f.remote(i) for i in range(4)]
print("Core: Parallelizing Functions with Ray Tasks")
t1 = time.time()
print(ray.get(futures)) # [0, 1, 4, 9]
t2 = time.time()
print(f"Time taken with Ray: {t2 - t1}\n")


print("Non-Core: Running Functions Sequentially")
t1 = time.time()
regular = [f2(i) for i in range(4)]
print(regular) # [0, 1, 4, 9]
t2 = time.time()
print(f"Time taken with regular function: {t2 - t1}\n")

@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

counters = [Counter.remote() for i in range(4)]
[c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print("Core: Parallelizing Functions with Ray Actors")
print(ray.get(futures)) # [1, 1, 1, 1]
