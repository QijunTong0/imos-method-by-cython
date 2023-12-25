import numpy as np
import time
from tqdm import tqdm
from imos_cython import imos_cython

t = 32
shape = np.array([200, 30, 70, t], dtype=np.int16)
n_index, d_index, _ = np.where(np.ones(shape=np.array([200, 30, 70]), dtype=np.int16))

st = np.random.randint(0, shape[2] // 2, size=shape[:3], dtype=np.int16)
ed = st + np.random.randint(0, shape[2] // 2, size=shape[:3], dtype=np.int16)
staff_skill = np.random.randint(0, 2**31, size=70, dtype=np.int32)
staff_skill_bool = np.random.randint(0, 2, size=(70, t), dtype=np.int16)

arr = np.random.randint(0, 2, size=(200, 30, 70, 70), dtype=np.bool_)
time_st = time.time()
for i in tqdm(range(20)):
    np.dot(arr, staff_skill_bool)
print("naive:", (time.time() - time_st), "s")


time_st = time.time()
for i in tqdm(range(1000)):
    imos_cython(shape, st, ed, staff_skill)
print("imos_cython:", (time.time() - time_st), "s")
