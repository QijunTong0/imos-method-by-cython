import numpy as np
import cython
cimport numpy as cnp
cnp.import_array()

@cython.wraparound(False)
@cython.boundscheck(False)
def imos_cython(
        cnp.int16_t[::1] shape,
        cnp.int16_t[:,:,::1] st,
        cnp.int16_t[:,:,::1] ed,
        cnp.int32_t[::1] staff_skill,
        ):
    cdef cnp.int16_t[:,:,:,::1] res = np.zeros(shape,dtype=np.int16)
    cdef int i,j,k,l,p,q,r,s
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(st.shape[2]):
                p=st[i,j,k]
                q=ed[i,j,k]
                r=staff_skill[k]
                for l in range(32):
                    s=r>>l&1
                    res[i,j,p,l] = res[i,j,p,l] + s
                    res[i,j,q,l] = res[i,j,q,l] - s
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]-1):
                for l in range(32):
                    res[i,j,k+1,l] += res[i,j,k,l]
    return res
