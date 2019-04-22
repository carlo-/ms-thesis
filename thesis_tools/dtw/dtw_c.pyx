import numpy as np
cimport numpy as np
cimport cython

DIST_DTYPE = np.float32
PATH_DTYPE = np.int

ctypedef np.float32_t DIST_DTYPE_t
ctypedef np.int_t PATH_DTYPE_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def dynamic_time_warping_c(np.ndarray[DIST_DTYPE_t, ndim=2] precomputed_distances):
    assert precomputed_distances.dtype == DIST_DTYPE

    cdef int n = precomputed_distances.shape[0]
    cdef int m = precomputed_distances.shape[1]
    assert n > 1, m > 1

    cdef np.ndarray[DIST_DTYPE_t, ndim=2] cost = np.full([n + 1, m + 1], np.inf, dtype=DIST_DTYPE)
    cost[0, 0] = 0

    cdef DIST_DTYPE_t dist
    cdef int i, j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = precomputed_distances[i-1, j-1]
            cost[i, j] = dist + min(cost[i-1, j],
                                    cost[i, j-1],
                                    cost[i-1, j-1])
    cdef np.ndarray[PATH_DTYPE_t, ndim=2] path = compute_path_c(cost)
    return cost[1:, 1:], path


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def compute_path_c(np.ndarray[DIST_DTYPE_t, ndim=2] cost):

    cdef int i = cost.shape[0]
    cdef int j = cost.shape[1]

    cdef np.ndarray[PATH_DTYPE_t, ndim=2] p = np.zeros([i * j, 2], dtype=PATH_DTYPE)
    cdef int p_idx = p.shape[0] - 1

    i -= 2
    j -= 2

    p[p_idx, 0] = i
    p[p_idx, 1] = j
    p_idx -= 1

    cdef int k
    while i > 0 or j > 0:
        k = np.argmin((cost[i, j], cost[i, j + 1], cost[i + 1, j]))
        if k == 0:
            i -= 1
            j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1

        p[p_idx, 0] = i
        p[p_idx, 1] = j
        p_idx -= 1

    return p[(p_idx + 1):]
