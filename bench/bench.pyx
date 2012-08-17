# cython: boundscheck=False
# cython: wraparound=False

"""
Simple hacked-up benchmarks for minivect.
"""

import os
import time
import math
import string

import numpy as np
try:
    import numexpr
except ImportError:
    print "numexpr not found"
    numexpr = None
try:
    import theano
except ImportError:
    print "theano not found"
    theano = None

ctypedef float dtype_t
precision = "Single"

cdef extern from "fbench.h":
    void aplusb_ff(dtype_t *, dtype_t *, int, int)
    void aplusb_fcf(dtype_t *, dtype_t *, dtype_t *, int, int)
    void aplusb_cfcfcf(dtype_t *, dtype_t *, dtype_t *, dtype_t *, dtype_t *, dtype_t *, int, int, int)
    void aplusb_strided_cc(dtype_t *, dtype_t *, int, int, int)
    void aplusb_strided_fc(dtype_t *, dtype_t *, int, int, int)
    void innercontig2d(dtype_t *, dtype_t *, dtype_t *, dtype_t *, int, int, int)


# ctypedef fused dtype_t:
#     int
#     float
#     dtype_t

DEF numpy_name = 'NumPy'
DEF cython_name = 'Cython'
DEF numexpr_name = 'NumExpr'
DEF theano_name = 'Theano'
DEF numexpr_threaded = 'NumExpr 4 Threads'
DEF fortran_name = 'Fortran'


cdef inline double gettime():
    return time.time()

def ops(size1, size2, dtype, orders):
    arrays = []
    for order in orders:
        a = np.arange(size1 * size2, dtype=dtype).reshape(size1, size2, order=order)
        arrays.append(a)
    return arrays

def theano_compile(expr, d):
    """
    This compile a Theano function from string "expr"
    and a dict of local variable "d".
    """
    e = "out = " + expr
    d = d.copy()
    for k, v in d.iteritems():
        if isinstance(v, np.ndarray):
            d[k] = theano.shared(v, borrow=True)
    exec e in {}, d
    f = theano.function([],
                        #Tell Theano that it can reuse previously memory region
                        #This make theano allocate only the first time the output
                        theano.Out(d['out'], borrow=True))

    return f

cdef class Benchmark(object):
    name = None
    sizes = None

    cdef int nouter
    cdef int ninner
    dtype = np.float32
    names = string.ascii_letters
    sizes = [400, 800, 1200, 1600, 2000, 2400]
    strides = None

    xaxis = "Data Size"

    def __init__(self, nouter=5, ninner=50):
        self.nouter = nouter
        self.ninner = ninner

    cdef verify_result(self, size):
        numpy_operands = self.get_operands(size)
        self.numpy(numpy_operands, 1, 1)
        numpy_result = numpy_operands[0]

        cython_operands = self.get_operands(size)
        self.cython(cython_operands, 1, 1)
        cython_result = cython_operands[0]

        fortran_result = None
        if hasattr(self, 'fortran'):
            fortran_operands = self.get_operands(size)
            self.fortran(fortran_operands, 1, 1)
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
        cdef int i, j

        self.verify_result(size)

        times = {}
        operands = self.get_operands(size)

        cdef int nouter = self.get_nouter(size)
        cdef int ninner = self.get_ninner(size)

        self.numpy(operands, 1, 1)
        times[numpy_name] = self.numpy(operands, nouter, ninner)

        self.cython(operands, 1, 1)
        times[cython_name] = self.cython(operands, nouter, ninner)

        if hasattr(self, 'fortran'):
            self.fortran(operands, 1, 1)
            times[fortran_name] = self.fortran(operands, nouter, ninner)

        if numexpr is not None:
            numexpr_dict = dict(zip(self.names, operands))
            numexpr.set_num_threads(1)
            self.numexpr(self.expr, numexpr_dict, 1, 1)
            times[numexpr_name] = self.numexpr(self.expr, numexpr_dict, nouter, ninner)

            numexpr.set_num_threads(4)
            self.numexpr(self.expr, numexpr_dict, 1, 1)
            times[numexpr_threaded] = self.numexpr(self.expr, numexpr_dict, nouter, ninner)

        if theano is not None:
            theano_dict = dict(zip(self.names, operands))
            self.theano(self.expr, theano_dict, 1, 1)
            times[theano_name] = self.theano(self.expr, theano_dict, nouter, ninner)

        return times

    def run(self, out):
        d = {}
        for size in self.sizes:
            d[size] = self.run_size(size)
            # print self.name, d[size]

        self.report(d)
        self.dumpfile(d, out)

    def title(self, size_to_times):
        return self.name.replace('Double', precision)

    def _title(self, size_to_times):
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
                print size, name, self.flops(size, time), "MFlops", time, "seconds"

            print

        print

    def dumpfile(self, size_to_times, out):
        f = open(out, 'w')
        f.write(self.title(size_to_times) + '\n')

        columns = []
        if hasattr(self, 'fortran'):
            columns.append(fortran_name)
        columns.extend([cython_name, numpy_name])
        if numexpr is not None:
            columns.extend((numexpr_name, numexpr_threaded))
        if theano is not None:
            columns.append(theano_name)

        f.write('"%s" %s\n' % (self.xaxis,
                               " ".join('"%s"' % col for col in columns)))

        for size, times in sorted(size_to_times.iteritems()):
            f.write("%d %s\n" % (size, #int(math.log(size, 2)),
                                 " ".join(str(self.flops(size, times[col]))
                                                  for col in columns)))

        f.close()

    def get_operands(self, size):
        return ops(size, size, self.dtype, self.orders)

    def flop(self, size):
        return size * size * (len(self.orders) - 1) * self.get_ninner(size)

    def flops(self, size, time):
        return self.flop(size) / time / 1024 ** 2

    def get_nouter(self, size):
        return self.nouter

    def get_ninner(self, size):
        if size < 256:
            return self.ninner * (256/size)**2
        return self.ninner

    def numexpr(self, expr, d, int nouter, int ninner):
        cdef int i, j

        cdef double t
        times = []
        out = d['a']
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                numexpr.evaluate(expr, d, out=out)
            t = gettime() - t
            times.append(t)
        return min(times)

    def theano(self, expr, d, int nouter, int ninner):
        cdef int i, j

        cdef double t
        times = []
        #out = d['a']
        #We can't give Theano the output memory
        #but theano_compile compile the theano function
        #in such a way that it will allocate the output only at the first call.
        f = theano_compile(expr, d)
        #call it first to force the allocation of the output
        f()
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                f()
            t = gettime() - t
            times.append(t)
        return min(times)

    def numpy(self, operands, int nouter, int ninner):
        cdef int i, j

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                self._numpy(*operands)
            t = gettime() - t
            times.append(t)
        return min(times)


cdef class Contig2dC(Benchmark):
    name = "2D Double Precision, C Contig"
    expr = "a + b"
    orders = ['C', 'C']

    def cython(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b
        a, b = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                a[...] = a + b
            t = gettime() - t
            times.append(t)
        return min(times)

    def numpy(self, operands, int nouter, int ninner):
        cdef int i, j
        a, b = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                np.add(a, b, out=a)
            t = gettime() - t
            times.append(t)
        return min(times)

cdef class Contig2dF(Contig2dC):
    name = "2D Double Precision, Fortan Contig"
    orders = ['F', 'F']

    def fortran(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b

        cdef dtype_t *ap, *bp
        cdef int size1, size2

        a, b = operands
        ap = &a[0, 0]
        bp = &b[0, 0]
        size1 = a.shape[0]
        size2 = a.shape[1]

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                self._fortran(ap, bp, size1, size2)
            t = gettime() - t
            times.append(t)

        return min(times)

    cdef _fortran(self, dtype_t *a, dtype_t *b, int size1, int size2):
        aplusb_ff(a, b, size1, size2)

# cdef class MixedContig(Contig2dC):
#     orders = ['C', 'F']

cdef class MixedContig2(Contig2dF):
    name = "2D Double Precision, Mixed Contig Order"
    orders = ['F', 'C', 'F']
    expr = "b + c"

    def fortran(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b, c

        cdef dtype_t *ap, *bp, *cp
        cdef int size1, size2

        a, b, c = operands
        ap = &a[0, 0]
        bp = &b[0, 0]
        cp = &c[0, 0]
        size1 = a.shape[0]
        size2 = a.shape[1]

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                aplusb_fcf(ap, bp, cp, size1, size2)
            t = gettime() - t
            times.append(t)

        return min(times)

    def cython(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b, c
        a, b, c = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                a[...] = b + c
            t = gettime() - t
            times.append(t)
        return min(times)

    def numpy(self, operands, int nouter, int ninner):
        cdef int i, j
        a, b, c = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                np.add(b, c, out=a)
            t = gettime() - t
            times.append(t)
        return min(times)


cdef class MixedStridedPathological(Benchmark):

    #name = "a.T[:, :] = a + b.T + c + d.T + e + f.T" # assume all oeprands C contig
    name = "2D Double Precision, Mixed Strided Order\\n6 operands"
    expr = "b + c + d + e + f"
    orders = ['F', 'F', 'F', 'C', 'C', 'C']

    def get_operands(self, size):
        operands = ops(size, size, self.dtype, self.orders)
        return [op[::2, ::2] for op in operands]

    def flop(self, size):
        return super(MixedStridedPathological, self).flop(size) / 4

    def numpy(self, operands, int nouter, int ninner):
        cdef int i, j

        a, b, c, d, e, f = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                a[...] = b + c + d + e + f
            t = gettime() - t
            times.append(t)
        return min(times)

    def numexpr(self, expr, d, int nouter, int ninner):
        cdef int i, j
        d['a_T'] = d['a'].T

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                numexpr.evaluate(expr, d, out=d['a'])
            t = gettime() - t
            times.append(t)
        return min(times)

    def fortran(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b, c, d, e, f

        a, b, c, d, e, f = operands
        cdef dtype_t *ap, *bp, *cp, *dp, *ep, *fp
        ap = &a[0, 0]; bp = &b[0, 0]; cp = &c[0, 0]; dp = &d[0, 0]; ep =  &e[0, 0]; fp = &f[0, 0]
        cdef int size1 = a.shape[0]
        cdef int size2 = a.shape[1]

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                self._fortran(ap, bp, cp, dp, ep, fp, size1, size2)
            t = gettime() - t
            times.append(t)
        return min(times)

    cdef _fortran(self, dtype_t *a, dtype_t *b, dtype_t *c, dtype_t *d, dtype_t *e, dtype_t *f,
                        int size1, int size2):
        aplusb_cfcfcf(a, b, c, d, e, f, size1, size2, 2)

    def cython(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b, c, d, e, f
        a, b, c, d, e, f = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                a[...] = b + c + d + e + f
            t = gettime() - t
            times.append(t)
        return min(times)


cdef class Strided(Contig2dF):
    name = "2D Double Precision, Strided, C Order"
    strides = [2, 4, 8, 16, 32, 64, 128]
    size = 400
    orders = ['C', 'C']

    cdef int stride
    # order = ['C', 'C', 'F']

    xaxis = "Stride (power of two)"

    def run(self, out):
        d = {}
        for stride in self.strides:
            self.stride = stride
            d[stride] = self.run_size(stride)

        self.report(d)
        self.dumpfile(d, out)

    def flop(self, stride):
        return self.size ** 2 * self.get_ninner(stride)

    def get_ninner(self, stride):
        return self.ninner

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

    cdef _fortran(self, dtype_t *a, dtype_t *b, int size1, int size2):
        aplusb_strided_cc(a, b, size1, size2, self.stride)

cdef class MixedStrided(Strided):
    name = "2D Double Precision, Strided, Mixed Order"
    orders = ['F', 'C']

    cdef _fortran(self, dtype_t *a, dtype_t *b, int size1, int size2):
        aplusb_strided_fc(a, b, size1, size2, self.stride)

cdef class InnerContig(Benchmark):
    name = "2D Double Precision, Strided Inner Contig\\n4 operands"
    orders = ['F', 'F', 'F', 'F']
    expr = "a + b + c + d"

    def get_operands(self, size):
        operands = ops(size, size, self.dtype, self.orders)
        return [op[:, ::2] for op in operands]

    def flop(self, size):
        return super(InnerContig, self).flop(size) / 2

    def numpy(self, operands, int nouter, int ninner):
        cdef int i, j

        a, b, c, d = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                np.add(a, b, out=a)
                np.add(a, c, out=a)
                np.add(a, d, out=a)
            t = gettime() - t
            times.append(t)
        return min(times)

    def fortran(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b, c, d

        a, b, c, d = operands

        cdef dtype_t *ap, *bp, *cp, *dp
        ap = &a[0, 0]; bp = &b[0, 0]; cp = &c[0, 0]; dp = &d[0, 0]
        cdef int size1 = a.shape[0]
        cdef int size2 = a.shape[1]

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                self._fortran(ap, bp, cp, dp, size1, size2)
            t = gettime() - t
            times.append(t)
        return min(times)

    cdef _fortran(self, dtype_t *a, dtype_t *b, dtype_t *c, dtype_t *d, int size1, int size2):
        innercontig2d(a, b, c, d, size1, size2, 2)

    def cython(self, operands, int nouter, int ninner):
        cdef int i, j
        cdef dtype_t[:, :] a, b, c, d
        a, b, c, d = operands

        cdef double t
        times = []
        for i in range(nouter):
            t = gettime()
            for j in range(ninner):
                a[...] = a + b + c + d
            t = gettime() - t
            times.append(t)
        return min(times)

contig_benchmarks = [
    # Contig2dC,
    Contig2dF,
    #MixedContig,
]

strided_benchmarks = [
    Strided,
    InnerContig,
]

pathological = [
    MixedStrided,
    MixedContig2,
    MixedStridedPathological,
]

benchmarks = contig_benchmarks + strided_benchmarks + pathological
# benchmarks = strided_benchmarks
# benchmarks = pathological

def run():
    try:
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
    except OSError:
        pass

    for i, benchmark in enumerate(benchmarks):
        benchmark().run("out/out%d.txt" % i)
