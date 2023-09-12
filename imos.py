import numpy as np


def imos(shape: tuple[int, int, int], st: np.ndarray, ed: np.ndarray) -> np.ndarray:
    res = np.zeros(shape, dtype=np.int32)

    for i in st:
        res[i] += 1
    for i in ed:
        res[i] -= 1
    for i in range(1, n):
        res[i] += res[i - 1]
    return res


def imos_numpy(n: int, st: np.ndarray, ed: np.ndarray) -> np.ndarray:
    res = np.zeros(n, dtype=np.int32)
    np.add.at(res, st, 1)
    np.add.at(res, ed, -1)
    res = np.cumsum(res)
    return res
