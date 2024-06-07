import pickle
import os
import time

from functools import wraps
from collections import namedtuple

import numpy as np


PveData = namedtuple('PveData', ['train', 'test'])


def dump(value, fname):
    if not os.path.exists("pickled"):
        os.makedirs("pickled")
    with open(f"pickled/{fname}.pickle", 'wb') as f:
        pickle.dump(value, f)


def load(fname):
    loadr(f"pickled/{fname}.pickle")


def loadr(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def loadn(fname):
    if not os.path.exists(fname):
        return None
    else:
        return loadr(fname)


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


def are_distinct(array):
    reshaped = array.reshape(array.shape[0], -1)
    unique = np.unique(reshaped, axis=0)
    return unique.shape[0] == array.shape[0]



