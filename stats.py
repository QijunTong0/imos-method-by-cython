import numpy as np
import time
from tqdm import tqdm
from imos_cython import imos_cython

shape = np.array([200, 30, 70], dtype=np.int32)
n_index, d_index, _ = np.where(np.ones(shape=np.array([200, 30, 70]), dtype=np.int32))

st = np.random.randint(0, shape[2] // 2, size=shape, dtype=np.int32)
ed = st + np.random.randint(0, shape[2] // 2, size=shape, dtype=np.int32)

arr = np.random.randint(0, 2, size=(200, 30, 70, 70), dtype=np.bool_)
time_st = time.time()
for i in tqdm(range(100)):
    arr.sum(axis=2)
print("naive:", (time.time() - time_st), "s")

"""
st_ravel = np.ravel(st)
ed_ravel = np.ravel(ed)
time_st = time.time()
for i in tqdm(range(100)):
    res = np.zeros(shape, dtype=np.int32)
    np.add.at(res, (n_index, d_index, st_ravel), 1)
    np.add.at(res, (n_index, d_index, ed_ravel), -1)
    np.cumsum(res, out=res, axis=2)
print("imos_numpy:", (time.time() - time_st), "s")
"""

time_st = time.time()
for i in tqdm(range(1000)):
    imos_cython(shape, st, ed)
print("imos_cython:", (time.time() - time_st), "s")
