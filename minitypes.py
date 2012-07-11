"""
This module provides a minimal type system, and ways to promote types, as
well as ways to convert to an LLVM type system. A set of predefined types are
defined. Types may be sliced to turn them into array types, in the same way
as the memoryview syntax.

>>> char
char
>>> int8[:, :, :]
int8[:, :, :]
>>> int8.signed
True
>>> uint8
uint8
>>> uint8.signed
False

>>> char.pointer()
char *
>>> int_[:, ::1]
int[:, ::1]
>>> int_[::1, :]
int[::1, :]
>>> double[:, ::1, :]
Traceback (most recent call last):
   ...
InvalidTypeSpecification: Step may only be provided once, and only in the first or last dimension.
"""

__all__ = ['Py_ssize_t', 'void', 'char', 'uchar', 'int_', 'long_', 'bool_', 'object_',
           'float_', 'double', 'longdouble', 'float32', 'float64', 'float128',
           'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
           'complex64', 'complex128', 'complex256']

try:
    import llvm.core as lc
except ImportError:
    lc = None

import miniutils
import minierror

class TypeMapper(object):
    """
    >>> import miniast
    >>> context = miniast.Context()
    >>> miniast.typemapper = TypeMapper(context)
    >>> tm = context.typemapper

    >>> tm.promote_types(int8, double)
    double
    >>> tm.promote_types(int8, uint8)
    uint8
    >>> tm.promote_types(int8, complex128)
    complex128
    >>> tm.promote_types(int8, object_)
    PyObject *

    >>> tm.promote_types(int64, float32)
    float
    >>> tm.promote_types(int64, complex64)
    complex64
    >>> tm.promote_types(float32, float64)
    double
    >>> tm.promote_types(float32, complex64)
    complex64
    >>> tm.promote_types(complex64, complex128)
    complex128
    >>> tm.promote_types(complex256, object_)
    PyObject *

    >>> tm.promote_types(float32.pointer(), Py_ssize_t)
    float *
    >>> tm.promote_types(float32.pointer(), Py_ssize_t)
    float *
    >>> tm.promote_types(float32.pointer(), uint8)
    float *

    >>> tm.promote_types(float32.pointer(), float64.pointer())
    Traceback (most recent call last):
        ...
    UnpromotableTypeError: (float *, double *)

    >>> tm.promote_types(float32[:, ::1], float32[:, ::1])
    float[:, ::1]
    >>> tm.promote_types(float32[:, ::1], float64[:, ::1])
    double[:, ::1]
    >>> tm.promote_types(float32[:, ::1], float64[::1, :])
    double[:, :]
    >>> tm.promote_types(float32[:, :], complex128[:, :])
    complex128[:, :]
    >>> tm.promote_types(int_[:, :], object_[:, ::1])
    PyObject *[:, :]
    """
    def __init__(self, context):
        self.context = context

    def map_type(self, opaque_type):
        if opaque_type.is_int:
            return int_
        elif opaque_type.is_float:
            return float_
        elif opaque_type.is_double:
            return double
        elif opaque_type.is_pointer:
            return PointerType(self.map_type(opaque_type.base_type))
        elif opaque_type.is_py_ssize_t:
            return Py_ssize_t
        elif opaque_type.is_char:
            return char
        else:
            raise minierror.UnmappableTypeError(opaque_type)

    def to_llvm(self, type):
        "Return an LLVM type for the given type."
        raise NotImplementedError

    def from_python(self, value):
        "Get a type from a python value"
        if isinstance(value, float):
            return double
        elif isinstance(value, (int, long)):
            return int_
        elif isinstance(value, complex):
            return complex128
        else:
            return object_
            # raise minierror.UnmappableTypeError(type(value))

    def promote_numeric(self, type1, type2):
        "Promote two numeric types"
        return max([type1, type2], key=lambda type: type.rank)

    def promote_arrays(self, type1, type2):
        "Promote two array types in an expression to a new array type"
        equal_ndim = type1.ndim == type2.ndim
        return ArrayType(self.promote_types(type1.dtype, type2.dtype),
                         ndim=max(type1.ndim, type2.ndim),
                         is_c_contig=(equal_ndim and type1.is_c_contig and
                                      type2.is_c_contig),
                         is_f_contig=(equal_ndim and type1.is_f_contig and
                                      type2.is_f_contig))

    def promote_types(self, type1, type2):
        "Promote two arbitrary types"
        if type1.is_pointer and type2.is_int_like:
            return type1
        elif type2.is_pointer and type2.is_int_like:
            return type2
        elif type1.is_object or type2.is_object:
            return object_
        elif type1.is_numeric and type2.is_numeric:
            return self.promote_numeric(type1, type2)
        elif type1.is_array and type2:
            return self.promote_arrays(type1, type2)
        else:
            raise minierror.UnpromotableTypeError((type1, type2))


class Type(miniutils.ComparableObjectMixin):
    """
    Base class for all types.

    .. attribute:: subtypes

        The list of subtypes to allow comparing and hashing them recursively
    """

    is_array = False
    is_pointer = False
    is_typewrapper = False

    is_bool = False
    is_numeric = False
    is_py_ssize_t = False
    is_char = False
    is_int = False
    is_float = False
    is_c_string = False
    is_object = False
    is_function = False
    is_int_like = False
    is_complex = False
    is_void = False

    subtypes = []

    def __init__(self, **kwds):
        vars(self).update(kwds)
        self.qualifiers = kwds.get('qualifiers', frozenset())

    def qualify(self, *qualifiers):
        "Qualify this type with a qualifier such as ``const`` or ``restrict``"
        qualifiers = list(qualifiers)
        qualifiers.extend(self.qualifiers)
        attribs = dict(vars(self), qualifiers=qualifiers)
        return type(self)(**attribs)

    def unqualify(self, *unqualifiers):
        "Remove the given qualifiers from the type"
        unqualifiers = set(unqualifiers)
        qualifiers = [q for q in self.qualifiers if q not in unqualifiers]
        attribs = dict(vars(self), qualifiers=qualifiers)
        return type(self)(**attribs)

    def pointer(self):
        "Get a pointer to this type"
        return PointerType(self)

    @property
    def subtype_list(self):
        return [getattr(self, subtype) for subtype in self.subtypes]

    @property
    def comparison_type_list(self):
        return self.subtype_list

    def __eq__(self, other):
        # Don't use isinstance here, compare on exact type to be consistent
        # with __hash__. Override where sensible
        return (type(self) is type(other) and
                self.comparison_type_list == other.comparison_type_list)

    def __hash__(self):
        h = hash(type(self))
        for subtype in self.comparison_type_list:
            h = h ^ hash(subtype)

        return h

    def __getitem__(self, item):
        assert isinstance(item, (tuple, slice))

        def verify_slice(s):
            if s.start or s.stop or s.step not in (None, 1):
                raise minierror.InvalidTypeSpecification(
                    "Only a step of 1 may be provided to indicate C or "
                    "Fortran contiguity")

        if isinstance(item, tuple):
            step_idx = None
            for idx, s in enumerate(item):
                verify_slice(s)
                if s.step and (step_idx or idx not in (0, len(item) - 1)):
                    raise minierror.InvalidTypeSpecification(
                        "Step may only be provided once, and only in the "
                        "first or last dimension.")

                if s.step == 1:
                    step_idx = idx

            return ArrayType(self, len(item),
                             is_c_contig=step_idx == len(item) - 1,
                             is_f_contig=step_idx == 0)
        else:
            verify_slice(item)
            return ArrayType(self, 1, is_c_contig=bool(item.step))

    def to_llvm(self, context):
        "Get a corresponding llvm type from this type"
        return context.to_llvm(self)

    def __getattr__(self, attr):
        if attr.startswith('is_'):
            return False
        return getattr(type(self), attr)

class ArrayType(Type):

    is_array = True
    subtypes = ['dtype']

    def __init__(self, dtype, ndim, is_c_contig=False, is_f_contig=False,
                 inner_contig=False):
        super(ArrayType, self).__init__()
        self.dtype = dtype
        self.ndim = ndim
        self.is_c_contig = is_c_contig
        self.is_f_contig = is_f_contig
        self.inner_contig = inner_contig or is_c_contig or is_f_contig

    @property
    def comparison_type_list(self):
        return [self.dtype, self.is_c_contig, self.is_f_contig, self.inner_contig]

    def pointer(self):
        raise Exception("You probably want a pointer type to the dtype")

    def to_llvm(self, context):
        raise Exception("Obtain a pointer to the dtype and convert that "
                        "to an LLVM type")

    def __repr__(self):
        axes = [":"] * self.ndim
        if self.is_c_contig:
            axes[-1] = "::1"
        elif self.is_f_contig:
            axes[0] = "::1"

        return "%s[%s]" % (self.dtype, ", ".join(axes))

class PointerType(Type):
    is_pointer = True
    subtypes = ['base_type']

    def __init__(self, base_type, **kwds):
        super(PointerType, self).__init__(**kwds)
        self.base_type = base_type

    def __repr__(self):
        return "%s *%s" % (self.base_type, " ".join(self.qualifiers))

    def to_llvm(self, context):
        return lc.Type.pointer(self.base_type.to_llvm(context))

class CArrayType(Type):
    is_carray = True
    subtypes = ['base_type']

    def __init__(self, base_type, size, **kwds):
        super(CArrayType, self).__init__(**kwds)
        self.base_type = base_type
        self.size = size

    def __repr__(self):
        return "%s[%d]" % (self.base_type, self.length)

    def to_llvm(self, context):
        return lc.Type.array(self.base_type.to_llvm(context), self.size)

class TypeWrapper(Type):
    is_typewrapper = True
    subtypes = ['opaque_type']

    def __init__(self, opaque_type, context, **kwds):
        super(TypeWrapper, self).__init__(**kwds)
        self.opaque_type = opaque_type
        self.context = context

    def __repr__(self):
        return self.context.declare_type(self)

    def __deepcopy__(self, memo):
        return self

class NamedType(Type):
    name = None
    def __eq__(self, other):
        return isinstance(other, NamedType) and self.name == other.name

    def __repr__(self):
        if self.qualifiers:
            return "%s %s" % (self.name, " ".join(self.qualifiers))
        return self.name

class BoolType(NamedType):
    is_bool = True
    name = "bool"

    def __repr__(self):
        return "int %s" % " ".join(self.qualifiers)

    def to_llvm(self, context):
        return int8.to_llvm(context)

class NumericType(NamedType):
    """
    Base class for numeric types.

    .. attribute:: name

        name of the type

    .. attribute:: itemsize

        sizeof(type)

    .. attribute:: rank

        ordering of numeric types
    """
    is_numeric = True

class IntType(NumericType):
    is_int = True
    is_int_like = True
    name = "int"
    signed = True
    rank = 4
    itemsize = 4

    def to_llvm(self, context):
        if self.itemsize == 1:
            return lc.Type.int(8)
        elif self.itemsize == 2:
            return lc.Type.int(16)
        elif self.itemsize == 4:
            return lc.Type.int(32)
        else:
            assert self.itemsize == 8, self
            return lc.Type.int(64)

class FloatType(NumericType):
    is_float = True

    def to_llvm(self, context):
        if self.itemsize == 4:
            return lc.Type.float()
        elif self.itemsize == 8:
            return lc.Type.double()
        else:
            # Note: what about fp80/fp96?
            assert self.itemsize == 16
            return lc.Type.fp128()

class ComplexType(NumericType):
    is_complex = True
    subtypes = ['base_type']

class Py_ssize_t_Type(IntType):
    is_py_ssize_t = True
    name = "Py_ssize_t"
    rank = 9
    signed = True

class CharType(IntType):
    is_char = True
    name = "char"
    rank = 1
    signed = True

    def to_llvm(self, context):
        return lc.Type.int(8)

class CStringType(Type):
    is_c_string = True

    def __repr__(self):
        return "const char *"

    def to_llvm(self, context):
        return char.pointer().to_llvm(context)

class VoidType(NamedType):
    is_void = True
    name = "void"

    def to_llvm(self, context):
        return lc.Type.void()

class ObjectType(Type):
    is_object = True

    def __repr__(self):
        return "PyObject *"

class FunctionType(Type):
    subtypes = ['return_type', 'args']
    is_function = True
    is_vararg = False

    def to_llvm(self, context):
        return lc.Type.function(self.return_type.to_llvm(context),
                                [arg_type.to_llvm(context)
                                    for arg_type in self.args],
                                self.is_vararg)

#
### Internal types
#
c_string_type = CStringType()
void = VoidType()

#
### Public types
#
Py_ssize_t = Py_ssize_t_Type()
size_t = IntType(name="size_t", rank=8.5, itemsize=8, signed=False)
char = CharType(name="char")
short = IntType(name="short", rank=2, itemsize=2)
int_ = IntType(name="int", rank=4, itemsize=4)
long_ = IntType(name="long", rank=5, itemsize=4)
longlong = IntType(name="PY_LONG_LONG", rank=8, itemsize=8)

uchar = CharType(name="unsigned char", signed=False)
ushort = IntType(name="unsigned short", rank=2.5, itemsize=2, signed=False)
uint = IntType(name="unsigned int", rank=4.5, itemsize=4, signed=False)
ulong = IntType(name="unsigned long", rank=5.5, itemsize=4, signed=False)
ulonglong = IntType(name="unsigned PY_LONG_LONG", rank=8.5, itemsize=8,
                    signed=False)

bool_ = BoolType()
object_ = ObjectType()

int8 = IntType(name="int8", rank=1, itemsize=1)
int16 = IntType(name="int16", rank=2, itemsize=2)
int32 = IntType(name="int32", rank=4, itemsize=4)
int64 = IntType(name="int64", rank=8, itemsize=8)

uint8 = IntType(name="uint8", rank=1.5, signed=False, itemsize=1)
uint16 = IntType(name="uint16", rank=2.5, signed=False, itemsize=2)
uint32 = IntType(name="uint32", rank=4.5, signed=False, itemsize=4)
uint64 = IntType(name="uint64", rank=8.5, signed=False, itemsize=8)

float32 = float_ = FloatType(name="float", rank=10, itemsize=4)
float64 = double = FloatType(name="double", rank=12, itemsize=8)
float128 = longdouble = FloatType(name="long double", rank=14, itemsize=16)

complex64 = ComplexType(name="complex64", base_type=float32,
                        rank=16, itemsize=8)
complex128 = ComplexType(name="complex128", base_type=float64,
                         rank=18, itemsize=16)
complex256 = ComplexType(name="complex256", base_type=float128,
                         rank=20, itemsize=32)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
