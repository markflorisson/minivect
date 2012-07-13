# cython: boundscheck=False
# cython: wraparound=False

"""
Simple hacked-up benchmarks for minivect.
"""

import os
import time
import string

import numpy as np
try:
    import numexpr
except ImportError:
    print "numexpr not found"
    numexpr = None

cdef extern from "fbench.h":
    void aplusb_ff(double *, double *, int, int)
    void aplusb_fc(double *, double *, int, int)
    void aplusb_strided_cc(double *, double *, int, int, int)
    void aplusb_strided_fc(double *, double *, int, int, int)

# ctypedef fused dtype_t:
#     int
#     float
#     double

DEF numpy_name = 'NumPy'
DEF cython_name = 'Cython'
DEF numexpr_name = 'NumExpr'
DEF numexpr_threaded = 'NumExpr 4 Threads'
DEF fortran_name = 'Fortran'

ctypedef double dtype_t

def ops(size1, size2, dtype, orders):
    arrays = []
    for order in orders:
        a = np.arange(size1 * size2, dtype=dtype).reshape(size1, size2, order=order)
        arrays.append(a)
    return arrays

cdef class Benchmark(object):
    name = None
    sizes = None

    N = 50
    dtype = np.double
    names = string.ascii_letters
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    strides = None

    xaxis = "Data Size"

    cdef verify_result(self, size):
        numpy_operands = self.get_operands(size)
        self.numpy(numpy_operands, 1)
        numpy_result = numpy_operands[0]

        cython_operands = self.get_operands(size)
        self.cython(cython_operands, 1)
        cython_result = cython_operands[0]

        fortran_result = None
        if hasattr(self, 'fortran'):
            fortran_operands = self.get_operands(size)
            self.fortran(fortran_operands, 1)
            fortran_result = fortran_operands[0]

        if not np.allclose(numpy_result, cython_result):
            print 'Different results for Cython', self.name, type(self).__name__, size
            print numpy_result
            print '--------------------------'
            print cython_result
            raise Exception
        elif fortran_result is not None and not np.allclose(numpy_result, fortran_result):
            print 'Different results for Fortran', self.name, type(self).__name__, size
            print numpy_result
            print '--------------------------'
            print fortran_result
            raise Exception

    def run_size(self, size):
        cdef int i

        self.verify_result(size)

        times = {}
        operands = self.get_operands(size)

        self.numpy(operands, 1)
        times[numpy_name] = self.numpy(operands, self.N)

        self.cython(operands, 1)
        times[cython_name] = self.cython(operands, self.N)

        if hasattr(self, 'fortran'):
            self.fortran(operands, 1)
            times[fortran_name] = self.fortran(operands, self.N)

        if numexpr is not None:
            numexpr_dict = dict(zip(self.names, operands))
            numexpr.set_num_threads(1)
            self.numexpr(self.expr, numexpr_dict, 1)
            times[numexpr_name] = self.numexpr(self.expr, numexpr_dict, self.N)

            numexpr.set_num_threads(4)
            self.numexpr(self.expr, numexpr_dict, 1)
            times[numexpr_threaded] = self.numexpr(self.expr, numexpr_dict, self.N)

        return times

    def run(self, out):
        d = {}
        for size in self.sizes:
            d[size] = self.run_size(size)
            # print self.name, d[size]

        self.report(d)
        self.dumpfile(d, out)

    def title(self, size_to_times):
        order = self.orders[0]
        for new_order in self.orders[1:]:
            if new_order != order:
                order_string = "mixed data order"
                break
        else:
            order_string = "%s data order" % order

        all_contig = all(op.flags['C_CONTIGUOUS'] or op.flags['F_CONTIGUOUS']
                             for op in self.get_operands(10))
        any_contig = any(op.flags['C_CONTIGUOUS'] or op.flags['F_CONTIGUOUS']
                             for op in self.get_operands(10))

        if all_contig:
            order_string = 'contiguous, %s' % order_string
        elif any_contig:
            order_string = 'mixed contiguous/strided, %s' % order_string
        else:
            order_string = 'strided, %s' % order_string

        return '%s %s' % (self.name, order_string)

    def report(self, size_to_times):
        print self.title(size_to_times)
        for size, times in sorted(size_to_times.iteritems()):
            for name, time in times.iteritems():
                print size, name, time

            print

        print

    def dumpfile(self, size_to_times, out):
        f = open(out, 'w')
        f.write(self.title(size_to_times) + '\n')

        columns = [numpy_name, cython_name]
        if hasattr(self, 'fortran'):
            columns.append(fortran_name)
        if numexpr is not None:
            columns.extend((numexpr_name, numexpr_threaded))

        f.write('"%s" %s\n' % (self.xaxis, " ".join('"%s"' % col for col in columns)))

        for size, times in sorted(size_to_times.iteritems()):
            f.write("%d %s\n" % (size, " ".join(str(times[col]) for col in columns)))

        f.close()

    def numexpr(self, expr, d, Py_ssize_t ntimes):
        cdef int i

        t = time.time()
        for i in range(ntimes):
            numexpr.evaluate(expr, d, out=d['a'])
        return time.time() - t

    def get_operands(self, size):
        return ops(size, size, self.dtype, self.orders)

    def cython(self, operands, Py_ssize_t ntimes):
        cdef int i

        t = time.time()
        for i in range(ntimes):
            self._cython(operands)
        return time.time() - t

    def numpy(self, operands, Py_ssize_t ntimes):
        cdef int i

        t = time.time()
        for i in range(ntimes):
            self._numpy(*operands)
        return time.time() - t


cdef class Contig2dC(Benchmark):
    name = "a + b"
    expr = "a + b"
    orders = ['C', 'C']
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    def cython(self, operands, Py_ssize_t ntimes):
        cdef int i
        cdef double[:, :] a, b
        a, b = operands

        t = time.time()
        for i in range(ntimes):
            a[...] = a + b
        return time.time() - t

    def numpy(self, operands, Py_ssize_t ntimes):
        cdef int i
        a, b = operands

        t = time.time()
        for i in range(ntimes):
            np.add(a, b, out=a)
        return time.time() - t

cdef class Contig2dF(Contig2dC):
    orders = ['F', 'F']

    def fortran(self, operands, Py_ssize_t ntimes):
        cdef int i
        cdef double[:, :] a, b

        a, b = operands

        t = time.time()
        self._fortran(&a[0, 0], &b[0, 0], a.shape[0], a.shape[1], ntimes)
        return time.time() - t

    cdef _fortran(self, double *a, double *b, int size1, int size2, Py_ssize_t ntimes):
        for i in range(ntimes):
            aplusb_ff(a, b, size1, size2)

cdef class MixedContig(Contig2dC):
    name = 'a[...] = a[:, ::1] + b[::1, :]'
    orders = ['C', 'F']

cdef class MixedContig2(Contig2dF):
    name = 'a[...] = a[::1, :] + b[:, ::1]'
    orders = ['F', 'C']

    cdef _fortran(self, double *a, double *b, int size1, int size2, Py_ssize_t ntimes):
        for i in range(ntimes):
            aplusb_fc(a, b, size1, size2)


cdef class Strided(Contig2dF):
    name = "a + b"
    strides = [2, 4, 8, 16, 32, 64, 128]
    size = 512
    orders = ['C', 'C']
    cdef int stride
    # order = ['C', 'C', 'F']

    xaxis = "Stride"

    def run(self, out):
        d = {}
        for stride in self.strides:
            self.stride = stride
            d[stride] = self.run_size(stride)

        self.report(d)
        self.dumpfile(d, out)

    def get_operands(self, stride):
        operands = []
        for order in self.orders:
            if order == 'C':
                size1 = self.size
                size2 = self.size * stride
                slices = slice(None), slice(None, None, stride)
            else:
                size1 = self.size * stride
                size2 = self.size
                slices = slice(None, None, stride), slice(None)

            op, = ops(size1, size2, self.dtype, [order])
            operands.append(op[slices])

        return operands

    cdef _fortran(self, double *a, double *b, int size1, int size2, Py_ssize_t ntimes):
        for i in range(ntimes):
            aplusb_strided_cc(a, b, size1, size2, self.stride)

cdef class MixedStrided(Strided):
    orders = ['F', 'C']

    cdef _fortran(self, double *a, double *b, int size1, int size2, Py_ssize_t ntimes):
        for i in range(ntimes):
            aplusb_strided_fc(a, b, size1, size2, self.stride)


contig_benchmarks = [Contig2dC, Contig2dF, MixedContig, MixedContig2]
strided_benchmarks = [Strided, MixedStrided]

benchmarks = contig_benchmarks + strided_benchmarks
# benchmarks = strided_benchmarks

def run():
    try:
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    except OSError:
        pass

    for i, benchmark in enumerate(benchmarks):
        benchmark().run("out/out%d" % i)
