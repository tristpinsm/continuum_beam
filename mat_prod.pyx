import numpy as np

cimport numpy as np
cimport cython
cimport openmp
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def outer_sum(np.ndarray[np.complex64_t, ndim=3, mode='c'] b not None,
              np.ndarray[np.float32_t, ndim=2, mode='c'] w not None,
              np.ndarray[np.float32_t, ndim=2, mode='c'] out not None):

    cdef int n = b.shape[2]
    cdef int n_vis = b.shape[0]
    cdef int n_time = b.shape[1]
    cdef int i, j, k, l
    cdef np.float32_t res

    for i in prange(n, nogil=True):
        for j in xrange(i, n):
                for k in xrange(n_vis):
                    for l in xrange(n_time):
                        res = (conj(b[k,l,i]) * b[k,l,j]).real * w[k,l]
                        out[i,j] += res
                        if i != j: out[j,i] += res

    return out

cdef inline np.complex64_t conj(np.complex64_t a) nogil:
    cdef np.complex64_t out
    out.real = a.real
    out.imag = - a.imag
    return out