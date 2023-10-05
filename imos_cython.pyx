#cython: language_level=3
#cython: boundcheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from cython.parallel import prange
cnp.import_array()

def imos_cython(cnp.ndarray[cnp.int32_t, ndim=1] shape, cnp.ndarray[cnp.int32_t, ndim=3] st, cnp.ndarray[cnp.int32_t, ndim=3] ed):
    cdef cnp.ndarray[cnp.int32_t, ndim=3] res = np.zeros(shape,dtype=np.int32)
    cdef int i,j,k
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(st.shape[2]):
                res[i,j,st[i,j,k]] += 1
            for k in range(ed.shape[2]):
                res[i,j,ed[i,j,k]] -= 1
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(1,shape[2]):
                res[i,j,k] += res[i,j,k - 1]
    return res

def imos1d_cython(
        int n,
        cnp.ndarray[cnp.int32_t, ndim=1] st,
        cnp.ndarray[cnp.int32_t, ndim=1] ed
        ):
    cdef cnp.ndarray[cnp.int32_t, ndim=1] res = np.zeros(n,dtype=np.int32)
    cdef int i
    for i in prange(st.shape[0],nogil=True):
        res[st[i]] += 1
        res[ed[i]] -= 1

    for i in range(n-1):
        res[i+1] += res[i]
    return res


def popcount_array(cnp.ndarray[cnp.uint64_t,ndim=3] arr):
    cdef int i,j,k
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                arr[i,j,k]=popcount(arr[i,j,k])
    return arr

cdef popcount(cnp.uint64_t x):
    # 2bitごとの組に分け、立っているビット数を2bitで表現する
    x = x - ((x >> 1) & 0x5555555555555555)

    # 4bit整数に 上位2bit + 下位2bit を計算した値を入れる
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)

    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f # 8bitごと
    x = x + (x >> 8) # 16bitごと
    x = x + (x >> 16) # 32bitごと
    x = x + (x >> 32) # 64bitごと = 全部の合計
    return x & 0x0000007f