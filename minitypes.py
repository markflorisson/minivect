"""
Minimal type system. Don't call this module types, to avoid 'from .' imports
and ensure 2.4 compatibility.


>>> char
char
>>> int8[:, :]
int8[:, :]
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

__all__ = ['Py_ssize_t', 'void', 'char', 'uchar', 'int_', 'bool_', 'object_',
           'float_', 'double', 'longdouble', 'float32', 'float64', 'float128',
           'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
           'complex64', 'complex128', 'complex256']

try:
    import llvm.core as lc
except ImportError:
    lc = None

import miniutils
import minierror

#rank_to_type_name = (
#    "char",
#    "short",
#    "int",
#    "long",
#    "Py_ssize_t",
#    "PY_LONG_LONG",
#    "float",
#    "double",
#    "long double",
#)
#typename_to_rank = dict(
#    (name, idx) for idx, name in enumerate(rank_to_type_name))

def promote(type1, type2):
    if type1.is_pointer:
        return type1
    elif type2.is_pointer:
        return type2
    elif type1.is_numeric and type2.is_numeric:
        return max([type1, type2], key=lambda type: type.rank)
    else:
        raise minierror.UnpromotableTypeError((type1, type2))

class TypeMapper(object):
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
        if isinstance(value, float):
            return double
        elif isinstance(value, (int, long)):
            return int_
        elif isinstance(value, complex):
            return complex128
        else:
            return object_
            # raise minierror.UnmappableTypeError(type(value))

class Type(miniutils.ComparableObjectMixin):
    is_array = False
    is_pointer = False
    is_typewrapper = False

    is_bool = False
    is_numeric = False
    is_py_ssize_t = False
    is_char = False
    is_int = False
    is_float = False
    is_double = False
    is_c_string = False
    is_object = False
    is_function = False
    is_int_like = False
    is_complex = False

    subtypes = []

    def __init__(self, **kwds):
        vars(self).update(kwds)
        self.qualifiers = set()

    def pointer(self):
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
                if (s.step and step_idx) or idx not in (0, len(item) - 1):
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
            return ArrayType(self, 1, is_c_contig=item.step)

    def to_llvm(self, context):
        return context.to_llvm(self)

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
        return [self.dtype, self.axes]

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

    def __init__(self, base_type):
        super(PointerType, self).__init__()
        self.base_type = base_type

    def tostring(self, qualifiers=None):
        if qualifiers is None:
            qualifiers = self.qualifiers
        return "%s *%s" % (self.base_type, " ".join(qualifiers))

    def __repr__(self):
        return self.tostring()

    def to_llvm(self, context):
        return lc.Type.pointer(self.base_type.to_llvm())

class MutablePointerType(Type):
    subtypes = ['pointer_type']

    def __init__(self, pointer_type):
        super(MutablePointerType, self).__init__()
        self.pointer_type = pointer_type

    def tostring(self):
        qualifiers = self.pointer_type.qualifiers - set(['const'])
        return self.pointer_type.tostring(qualifiers=qualifiers)

    def __getattr__(self, attr):
        return getattr(self.pointer_type, attr)

class CArrayType(Type):
    is_carray = True
    subtypes = ['base_type']

    def __init__(self, base_type, size):
        super(CArrayType, self).__init__()
        self.base_type = base_type
        self.size = size

    def __repr__(self):
        return "%s[%d]" % (self.base_type, self.length)

    def to_llvm(self, context):
        return lc.Type.array(self.base_type.to_llvm(), self.size)

class TypeWrapper(Type):
    is_typewrapper = True
    subtypes = ['opaque_type']

    def __init__(self, opaque_type, context):
        super(TypeWrapper, self).__init__()
        self.opaque_type = opaque_type
        self.context = context

    def __repr__(self):
        return self.context.declare_type(self)

    def __deepcopy__(self, memo):
        return self

class NamedType(Type):
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
        return int8.to_llvm()

class NumericType(NamedType):
    is_numeric = True

class IntLike(NumericType):
    is_int_like = True

class IntType(IntLike):
    is_int = True
    name = "int"
    signed = True
    rank = 4

    def to_llvm(self, context):
        if self.rank == 1:
            return lc.Type.int(8)
        elif self.rank == 2:
            return lc.Type.int(16)
        elif self.rank == 4:
            return lc.Type.int(32)
        else:
            assert self.rank == 8
            return lc.Type.int(64)

class FloatType(NumericType):
    is_float = True
    name = "float"
    rank = 4

    def to_llvm(self, context):
        if self.rank == 4:
            return lc.Type.float()
        elif self.rank == 8:
            return lc.Type.double()
        else:
            # Note: what about fp80/fp96?
            assert self.rank == 16
            return lc.Type.fp128()

class DoubleType(NumericType):
    is_double = True
    name = "double"
    rank = 8

class ComplexType(NumericType):
    is_complex = True

class Py_ssize_t_Type(IntLike):
    is_py_ssize_t = True
    name = "Py_ssize_t"
    rank = 9

class CharType(IntLike):
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
        return char.pointer().to_llvm()

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
        return lc.Type.function(self.return_type.to_llvm(),
                                [arg_type.to_llvm() for arg_type in self.args],
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
char = CharType()
uchar = CharType(signed=False)
int_ = IntType()
bool_ = BoolType()
object_ = ObjectType()

float32 = float_ = FloatType()
float64 = double = DoubleType()
float128 = longdouble = FloatType(rank=16)

int8 = IntType(name="int8", rank=1)
int16 = IntType(name="int16", rank=2)
int32 = IntType(name="int32", rank=4)
int64 = IntType(name="int64", rank=8)

uint8 = IntType(name="uint8", rank=1, signed=False)
uint16 = IntType(name="int16", rank=2, signed=False)
uint32 = IntType(name="int32", rank=4, signed=False)
uint64 = IntType(name="int64", rank=8, signed=False)

complex64 = ComplexType(name="complex64", rank=8)
complex128 = ComplexType(name="complex128", rank=16)
complex256 = ComplexType(name="complex256", rank=32)

if __name__ == '__main__':
    import doctest
    doctest.testmod()