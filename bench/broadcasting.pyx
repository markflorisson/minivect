import sys
import time
import numpy as np
cimport numpy as np

from demo import lazy_numpy
from demo.lazy_numpy import lazy_array, Binop

dtype = np.double

cdef double[:, :, :] a, b, c, result

DEF N = 200
DEF ntimes = 100

x, y, z = np.ogrid[:N, :N, :N]
x, y, z = x.astype(np.double), y.astype(np.double), z.astype(np.double)
a, b, c = x, y, z

a.strides[1] = a.strides[2] = b.strides[0] = b.strides[2] = c.strides[0] = c.strides[1] = 0

result = np.empty((N, N, N), dtype=np.double)

cdef int i

t = time.time()
for i in range(ntimes):
    result[...] = a * b * c
print 'cython', time.time() - t

assert np.allclose(result, x * y * z)

our_result = np.empty((N, N, N), dtype=dtype)

lazy_dst = lazy_array(our_result)
lazy_i, lazy_j, lazy_k = lazy_array(x), lazy_array(y), lazy_array(z)
lazy_result = Binop('=', lazy_dst, lazy_i * lazy_j * lazy_k)

cdef Py_intptr_t p
t = time.time()
p = lazy_result.getpointer()
t = time.time() - t
print 'compilation time', t

ctypedef int (*func_p)(Py_ssize_t *, double* op0_data, Py_ssize_t *op0_strides,
                                     double *op1_data, Py_ssize_t *op1_strides,
                                     double *op2_data, Py_ssize_t *op2_strides,
                                     double *op3_data, Py_ssize_t *op3_strides)
cdef func_p func = <func_p> p


result = np.empty((N, N, N), dtype=np.double)
func(shape, &result[0, 0, 0], &result.strides[0],
     &a[0, 0, 0], &a.strides[0],
     &b[0, 0, 0], &b.strides[0],
     &c[0, 0, 0], &c.strides[0])

assert np.allclose(result, x * y * z)

t = time.time()
for i in range(ntimes):
    func(shape, &result[0, 0, 0], &result.strides[0],
                          &a[0, 0, 0], &a.strides[0],
                          &b[0, 0, 0], &b.strides[0],
                          &c[0, 0, 0], &c.strides[0])
print 'lazy', time.time() - t

t = time.time()
for i in range(ntimes):
    x * y * z
t = time.time() - t
print 'numpy', t
