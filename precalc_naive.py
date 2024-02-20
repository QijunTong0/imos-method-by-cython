import numpy as np
from numpy.typing import NDArray


def precalc_st_dur_mapping_naive(calender: NDArray[np.int16]):
    """
    Args:
        calender (np.ndarray): 1 dimensional nparray
    """
    st_ed_map = np.full(
        shape=(calender.size, calender.size, 2), fill_value=-1, dtype=np.int16
    )
    for st_ind in range(calender.size):
        actual_st_ind = st_ind
        curr_dur = 0
        while (actual_st_ind < calender.size) and (not calender[actual_st_ind]):
            actual_st_ind += 1
        for ed_ind in range(actual_st_ind, calender.size):
            st_ed_map[st_ind, curr_dur, 0] = actual_st_ind
            if st_ed_map[st_ind, curr_dur, 1] == -1:
                st_ed_map[st_ind, curr_dur, 1] = ed_ind
            curr_dur += calender[ed_ind]
    return st_ed_map
