import numpy as np
from tqdm import tqdm
from precalc_naive import precalc_st_dur_mapping_naive
from cython_module import precalc_st_dur_mapping_cython

calender = (np.random.randint(0, 7, 2000) > 0).astype(np.int16)

for i in tqdm(range(1000)):
    precalc_st_dur_mapping_cython(calender)

for i in tqdm(range(2)):
    precalc_st_dur_mapping_naive(calender=calender)
