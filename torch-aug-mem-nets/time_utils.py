import time


def timeit(func):
    def timed_func(*args):
        t = time.time()
        r = func(*args)
        print("Function " + func.__name__ + " took: " + str(time.time() - t))
        return r
    return timed_func


@timeit
def my_func(i):
    n = 0
    for x in range(i):
        n += x
    return n
