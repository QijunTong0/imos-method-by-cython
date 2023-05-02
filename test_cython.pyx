cdef int i
cdef int[20000000] arr
for i in range(20000000):
    arr[i] = i + 1

for i in range(1,20000000):
    arr[i]+=arr[i-1]
