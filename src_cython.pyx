# cython: language_level = 3
import numpy as np
import cython
cimport numpy as cnp
cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
def precalc_st_dur_mapping_cython(cnp.int16_t[:] calender):
    cdef int size = calender.size
    cdef cnp.int16_t[:,:,:] st_ed_map = np.full((size, size, 2), -1, dtype=np.int16)
    cdef int st_ind, actual_st_ind, curr_dur, ed_ind
    for st_ind in range(size):
        actual_st_ind = st_ind
        curr_dur = 0
        while actual_st_ind < size and not calender[actual_st_ind]:
            actual_st_ind += 1
        for ed_ind in range(actual_st_ind, size):
            st_ed_map[st_ind, curr_dur, 0] = actual_st_ind
            if st_ed_map[st_ind, curr_dur, 1] == -1:
                st_ed_map[st_ind, curr_dur, 1] = ed_ind
            curr_dur += calender[ed_ind]
    return st_ed_map