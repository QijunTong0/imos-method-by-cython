import numpy as np
import cython
cimport numpy as cnp
cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
def imos_cython(
        cnp.ndarray[cnp.int32_t, ndim=1] shape,
        cnp.ndarray[cnp.int32_t, ndim=3] st,
        cnp.ndarray[cnp.int32_t, ndim=3] ed,
        cnp.ndarray[cnp.int32_t, ndim=2] staff_skill,
        ):
    cdef cnp.ndarray[cnp.int32_t, ndim=4] res = np.zeros(shape,dtype=np.int32)
    cdef int i,j,k
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(st.shape[2]):
                for l in range(staff_skill.shape[1]):
                    res[i,j,st[i,j,k],l] += staff_skill[k,l]
                    res[i,j,ed[i,j,k],l] -= staff_skill[k,l]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(1,shape[2]):
                res[i,j,k] += res[i,j,k - 1]
    return res

def imos1d_cython(
        int n,
        cnp.ndarray[cnp.int32_t, ndim=1] st,
        cnp.ndarray[cnp.int32_t, ndim=1] ed,
        ):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] res = np.zeros(n,dtype=np.int32)
    cdef int i
    for i in range(st.shape[0],nogil=True):
        res[st[i]] += 1
        res[ed[i]] -= 1

    for i in range(n-1):
        res[i+1] += res[i]
    return res
