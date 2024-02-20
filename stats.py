import numpy as np
import time
from tqdm import tqdm
from precalc_naive import precalc_st_dur_mapping_naive

calender = np.random.randint(0, 7, 2000)

time_st = time.time()
for i in tqdm(range(10)):
    precalc_st_dur_mapping_naive(calender=calender)
print("naive:", (time.time() - time_st), "s")


time_st = time.time()
for i in tqdm(range(1000)):
    imos_cython(shape, st, ed)
print("imos_cython:", (time.time() - time_st), "s")
