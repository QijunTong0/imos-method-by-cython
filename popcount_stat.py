from time import time_ns
import numpy as np
import numexpr as ne


def bit_count_org(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = t(-1)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return (arr * s01) >> (8 * (arr.itemsize - 1))


def bit_count(arr):
    # Make the values type-agnostic (as long as it's integers)
    t = arr.dtype.type
    mask = t(-1)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr -= np.bitwise_and(np.right_shift(arr, 1, out=arr), s55, out=arr)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return ne.evaluate("(arr * s01) >> (8 * (8 - 1))")


k = 200
m = 30
n = 180
arr = np.random.randint(0, 114514, size=(k, m, n), dtype=np.int64)

clock = time_ns()
for i in range(100):
    out1 = bit_count(arr)
print((time_ns() - clock) / 1000000, "ms")

clock = time_ns()
for i in range(100):
    out2 = bit_count_org(arr)
print((time_ns() - clock) / 1000000, "ms")
