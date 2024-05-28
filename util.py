from collections import namedtuple
import pickle
import os
import time
from functools import wraps

PveData = namedtuple('PveData', ['train', 'test'])


def dump(value, fname):
    if not os.path.exists("pickled"):
        os.makedirs("pickled")
    with open(f"pickled/{fname}.pickle", 'wb') as f:
        pickle.dump(value, f)


def load(fname):
    with open(f"pickled/{fname}.pickle", 'rb') as f:
        value = pickle.load(f)
    return value


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to complete.")
        return result
    return wrapper

