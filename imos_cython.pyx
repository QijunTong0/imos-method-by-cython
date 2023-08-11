#cython: language_level=3
#cython: boundcheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
cnp.import_array()

def imos_cython(int n, cnp.ndarray[cnp.int32_t, ndim=1] st, cnp.ndarray[cnp.int32_t, ndim=1] ed):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] res = np.zeros(n,dtype=np.int32)
    cdef int i

    for i in range(st.shape[0]):
        res[st[i]] += 1
    for i in range(ed.shape[0]):
        res[ed[i]] -= 1
    for i in range(1, n):
        res[i] += res[i - 1]

    return res
