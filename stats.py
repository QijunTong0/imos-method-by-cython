import numpy as np
import time
from imos_cython import imos1d_cython

n = 76
m = 70

st = np.random.randint(0, n // 2, size=m, dtype=np.int32)
ed = st + np.random.randint(0, n // 2, size=m, dtype=np.int32)

time_st = time.time()
for i in range(10**6):
    imos1d_cython(n, st, ed)
print("imos_cython:", (time.time() - time_st), "ns")

time_st = time.time()
for i in range(10**6):
    res = np.zeros(n)
    np.add.at(res, st, 1)
    np.add.at(res, ed, -1)
    np.cumsum(res)
print("imos_numpy:", (time.time() - time_st), "ns")

arr = np.random.randint(0, 2, size=(m, n), dtype=np.bool_)
time_st = time.time()
for i in range(10**6):
    arr.sum(axis=0)
print("naive:", (time.time() - time_st), "ns")
