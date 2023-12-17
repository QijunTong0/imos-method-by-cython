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
                res[i,j,st[i,j,k],0] += staff_skill[k,0]
                res[i,j,st[i,j,k],1] += staff_skill[k,1]
                res[i,j,st[i,j,k],2] += staff_skill[k,2]
                res[i,j,st[i,j,k],3] += staff_skill[k,3]
                res[i,j,st[i,j,k],4] += staff_skill[k,4]
                res[i,j,st[i,j,k],5] += staff_skill[k,5]
                res[i,j,st[i,j,k],6] += staff_skill[k,6]
                res[i,j,st[i,j,k],7] += staff_skill[k,7]
                res[i,j,st[i,j,k],8] += staff_skill[k,8]
                res[i,j,st[i,j,k],9] += staff_skill[k,9]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(1,shape[2]):
                res[i,j,k, 1] += res[i,j,k - 1, 1]
                res[i,j,k, 2] += res[i,j,k - 1, 2]
                res[i,j,k, 3] += res[i,j,k - 1, 3]
                res[i,j,k, 4] += res[i,j,k - 1, 4]
                res[i,j,k, 5] += res[i,j,k - 1, 5]
                res[i,j,k, 6] += res[i,j,k - 1, 6]
                res[i,j,k, 7] += res[i,j,k - 1, 7]
                res[i,j,k, 8] += res[i,j,k - 1, 8]
                res[i,j,k, 9] += res[i,j,k - 1, 9]
    return res
