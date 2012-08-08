"""
Convert a minivect type to a ctypes type and an llvm function to a
ctypes function.
"""

import math
import ctypes

def convert_to_ctypes(type):
    "Convert the minitype to a ctypes type"
    if type.is_pointer:
        return ctypes.POINTER(convert_to_ctypes(type.base_type))
    elif type.is_object or type.is_array:
        return ctypes.py_object
    elif type.is_float:
        if type.itemsize == 4:
            return ctypes.c_float
        elif type.itemsize == 8:
            return ctypes.c_double
        else:
            return ctypes.c_longdouble
    elif type.is_int:
        item_idx = int(math.log(type.itemsize))
        if type.is_signed:
            values = [ctypes.c_int8, ctypes.c_int16, ctypes.c_int32,
                      ctypes.c_int64]
        else:
            values = [ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32,
                      ctypes.c_uint64]
        return values[item_idx]
    elif type.is_complex:
        if type.itemsize == 8:
            return Complex64
        elif type.itemsize == 16:
            return Complex128
        else:
            return Complex256
    elif type.is_c_string:
        return ctypes.c_char_p
    elif type.is_function:
        return_type = convert_to_ctypes(type.return_type)
        arg_types = tuple(convert_to_ctypes(arg_type)
                              for arg_type in type.args)
        return ctypes.CFUNCTYPE(return_type, *arg_types)
    elif type.is_py_ssize_t:
        return getattr(ctypes, 'c_uint%d' % (_ext.sizeof_py_ssize_t() * 8))
    elif type.is_void:
        return None
    elif type.is_carray:
        return convert_to_ctypes(type.base_type) * type.size
    else:
        raise NotImplementedError(type)

def get_ctypes_func(func, llvm_func, llvm_execution_engine, context):
    "Get a ctypes function from an llvm function"
    ctypes_func_type = convert_to_ctypes(func.type)
    p = llvm_execution_engine.get_pointer_to_function(llvm_func)
    return ctypes_func_type(p)