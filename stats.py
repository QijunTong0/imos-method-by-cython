import numpy as np


def func(n: int, st: np.array, ed: np.array) -> np.array:
    res = np.zeros(n)
    for i in st:
        res[st[i]] += 1
    for i in ed:
        res[ed[i]] -= 1
    for i in range(1, n):
        res[i] += res[i - 1]
    return res
