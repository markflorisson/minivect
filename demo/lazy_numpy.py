import sys

import numpy as np

import miniast
import specializers
import minitypes
import codegen
import xmldumper
import treepath
from ctypes_conversion import get_data_pointer, convert_to_ctypes

context_debug = True

class LazyLLVMContext(miniast.LLVMContext):
    debug = context_debug
    def stridesvar(self, variable):
        return miniast.StridePointer(self.pos, minitypes.NPyIntp.pointer(),
                                     variable)

context = LazyLLVMContext()
b = context.astbuilder

ccontext = miniast.CContext()
ccontext.debug = context_debug

func_counter = 0

def specialize(specializer_cls, ast, context=context):
    specializers = [specializer_cls]
    result = iter(context.run(ast, specializers)).next()
    _, specialized_ast, _, code_output = result
    return specialized_ast, code_output

def specialize_c(specializer_cls, ast):
    specialized_ast, (proto, impl) = specialize(specializer_cls, ast, ccontext)
    return specialized_ast, impl


class Lazy(object):

    def __init__(self, parent=None):
        self.parent = parent

    def map(self):
        global func_counter
        assert self.parent is None

        variables = []
        body = self._map(variables)

        shapevar = b.variable(minitypes.NPyIntp().pointer(), 'shape')
        func = b.build_function(variables, body, 'lazy%d' % func_counter,
                                shapevar=shapevar)

        specializer = specializers.ContigSpecializer

        specialized_func, (llvm_func, ctypes_func) = specialize(specializer, func)
        func_counter += 1

        # print specialize_c(specializer, func)[1]
        # print specialized_func.print_tree(context)
        # print llvm_func

        return ctypes_func, variables, specializer

    def eval(self):
        ctypes_func, variables, specializer = self.map()

        fist_array = variables[0].value
        shape = fist_array.shape
        for variable in variables:
            if variable.value.shape != shape:
                raise NotImplementedError("broadcasting (%s, %s)" %
                                                (shape, variable.value.shape))

        args = [fist_array.ctypes.shape]
        for variable in variables:
            if variable.type.is_array:
                numpy_array = variable.value
                data_pointer = get_data_pointer(numpy_array, variable.type)
                args.append(data_pointer)
                if not specializer.is_contig_specializer:
                    args.append(numpy_array.ctypes.strides)
            else:
                raise NotImplementedError

        return ctypes_func(*args)

    def __add__(self, other):
        return Binop("+", self, lazy_array(other))

class Binop(Lazy):

    def __init__(self, op, lhs, rhs):
        super(Binop, self).__init__()
        self.op = op
        self.lhs, self.rhs = lhs, rhs
        lhs.parent, rhs.parent = self, self

    def _map(self, variables):
        lhs, rhs = self.lhs._map(variables), self.rhs._map(variables)
        assert lhs.type == rhs.type # no promotion

        if self.op == '=':
            return b.assign(lhs, rhs)
        else:
            return b.binop(lhs.type, self.op, lhs, rhs)

class LazyArray(Lazy):
    def __init__(self, numpy_array):
        super(LazyArray, self).__init__()
        self.numpy_array = numpy_array

    def _map(self, variables):
        minidtype = minitypes.map_dtype(self.numpy_array.dtype)
        array_type = minidtype[(slice(None),) * self.numpy_array.ndim]
        variable = b.variable(array_type, 'op%d' % len(variables))
        variables.append(variable)
        variable.value = self.numpy_array
        return variable

    def __setitem__(self, item, value):
        if not isinstance(item, tuple):
            item = (item,)

        for s in item:
            if not isinstance(s, slice):
                raise NotImplementedError("Only full slices are supported")
            elif s.start is not None or s.stop is not None or s.step is not None:
                raise NotImplementedError("Only full slice assignment is supported")

        lazy_result = Binop('=', self, value)
        return lazy_result.eval()

def lazy_array(numpy_array):
    if isinstance(numpy_array, Lazy):
        return numpy_array
    return LazyArray(numpy_array)

def test():
    """
    >>> test()
    [[   0.    2.    4.    6.    8.   10.   12.   14.   16.   18.]
     [  20.   22.   24.   26.   28.   30.   32.   34.   36.   38.]
     [  40.   42.   44.   46.   48.   50.   52.   54.   56.   58.]
     [  60.   62.   64.   66.   68.   70.   72.   74.   76.   78.]
     [  80.   82.   84.   86.   88.   90.   92.   94.   96.   98.]
     [ 100.  102.  104.  106.  108.  110.  112.  114.  116.  118.]
     [ 120.  122.  124.  126.  128.  130.  132.  134.  136.  138.]
     [ 140.  142.  144.  146.  148.  150.  152.  154.  156.  158.]
     [ 160.  162.  164.  166.  168.  170.  172.  174.  176.  178.]
     [ 180.  182.  184.  186.  188.  190.  192.  194.  196.  198.]]
    """
    a = np.arange(100, dtype=np.double).reshape(10, 10)
    lazy_a = lazy_array(a)
    lazy_a[:, :] = lazy_a + lazy_a
    print a

if __name__ == '__main__':
    test()
    #import doctest
    #doctest.testmod()