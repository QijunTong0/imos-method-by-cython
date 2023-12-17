import numpy as np
import cython
cimport numpy as cnp
cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
def imos_cython(
        cnp.int16_t[:] shape,
        cnp.int16_t[:,:,:] st,
        cnp.int16_t[:,:,:] ed,
        cnp.int16_t[:,:] staff_skill,
        ):
    cdef cnp.int16_t[:,:,:,:] res = np.zeros(shape,dtype=np.int16)
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
                for l in range(staff_skill.shape[1]):
                    res[i,j,k,l] += res[i,j,k - 1,l]
    return res
