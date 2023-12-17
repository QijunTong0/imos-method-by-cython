import numpy as np
import cython
cimport numpy as cnp
cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
def imos_cython(cnp.ndarray[cnp.int16_t, ndim=1] shape, cnp.ndarray[cnp.int16_t, ndim=3] st, cnp.ndarray[cnp.int16_t, ndim=3] ed):
    cdef cnp.ndarray[cnp.int16_t, ndim=3] res = np.zeros(shape,dtype=np.int16)
    cdef int i,j,k
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(st.shape[2]):
                res[i,j,st[i,j,k]] += 1
                res[i,j,ed[i,j,k]] -= 1
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(1,shape[2]):
                res[i,j,k] += res[i,j,k - 1]
    return res
