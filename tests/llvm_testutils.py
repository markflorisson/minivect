import numpy as np

import testutils
from testutils import *

llvm_context = context = get_llvm_context()
b = context.astbuilder

def get_array(shape=(130, 160), dtype=np.float32, order='C'):
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape, order=order)

def specialize(specializer_cls, ast, context=None, print_tree=False):
    return testutils.specialize(specializer_cls, ast,
                                context=context or llvm_context,
                                print_tree=print_tree)

def invoke(specializer_name, minifunc, *args):
    sp = sps[specializer_name]
    _, (ctypes_func, lfunc) = specialize(sp, minifunc)

class MiniFunction(miniutils.MiniFunction):
    def __init__(self, sp_name, variables, expr, name=None, context=None):
        context = context or llvm_context
        specializer = sps[sp_name]
        super(MiniFunction, self).__init__(context, specializer, variables,
                                           expr, name)