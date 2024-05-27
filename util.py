from collections import namedtuple
import pickle

PveData = namedtuple('PveData', ['train', 'test'])


def dump(value, fname):
    with open(f"pickled/{fname}.pickle", 'wb') as f:
        pickle.dump(value, f)


def load(fname):
    with open(f"pickled/{fname}.pickle", 'rb') as f:
        value = pickle.load(f)
    return value


