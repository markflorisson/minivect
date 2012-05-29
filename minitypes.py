"""
Minimal type system. Don't call this module types, to avoid 'from .' imports
and ensure 2.4 compatibility.
"""

import miniutils
import minierror

rank_to_type_name = (
    "char",
    "short",
    "int",
    "long",
    "Py_ssize_t",
    "PY_LONG_LONG",
    "float",
    "double",
    "long double",
)
typename_to_rank = dict(
    (name, idx) for idx, name in enumerate(rank_to_type_name))

def rank(type):
    assert type.is_numeric
    return typename_to_rank[type.name]

def promote(type1, type2):
    if type1.is_pointer:
        return type1
    elif type2.is_pointer:
        return type2
    elif type1.is_numeric and type2.is_numeric:
        return max([type1, type2], key=rank)
    else:
        raise minierror.UnpromotableTypeError((type1, type2))

class TypeMapper(object):
    def map_type(self, opaque_type):
        if opaque_type.is_int:
            return IntType()
        elif opaque_type.is_float:
            return FloatType()
        elif opaque_type.is_double:
            return DoubleType()
        elif opaque_type.is_pointer:
            return PointerType(self.map_type(opaque_type.base_type))
        elif opaque_type.is_py_ssize_t:
            return Py_ssize_t
        elif opaque_type.is_char:
            return c_char_t
        else:
            raise minierror.UnmappableTypeError(opaque_type)

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

    subtypes = []

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

    def tostring(self, context):
        return str(self)

class ArrayType(Type):

    is_array = True
    subtypes = ['dtype']

    def __init__(self, dtype, ndim, is_c_contig=False, is_f_contig=False,
                 inner_contig=False):
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

class PointerType(Type):
    is_pointer = True
    subtypes = ['base_type']

    def __init__(self, base_type):
        self.base_type = base_type

    def tostring(self, context):
        return "%s *" % context.declare_type(self.base_type)

class TypeWrapper(Type):
    is_typewrapper = True
    subtypes = ['opaque_type']

    def __init__(self, opaque_type):
        self.opaque_type = opaque_type

class NamedType(Type):
    def __str__(self):
        return self.name

class BoolType(NamedType):
    is_bool = True
    name = "bool"

class NumericType(NamedType):
    is_numeric = True

class IntType(NumericType):
    is_int = True
    name = "int"

class FloatType(NumericType):
    is_float = True
    name = "float"

class DoubleType(NumericType):
    is_double = True
    name = "double"

class Py_ssize_t_Type(NumericType):
    is_py_ssize_t = True
    name = "Py_ssize_t"

class CharType(NumericType):
    is_char = True
    name = "char"

Py_ssize_t = Py_ssize_t_Type()
c_char_t = CharType()
bool = BoolType()