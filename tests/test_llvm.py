from testutils import *

class Context(miniast.LLVMContext):
    def pipeline(self, specializer_class):
        return [specializer_class(self)]

context = Context()

class S(specializers.Specializer):
    def visit_FunctionNode(self, node):
        node.specializer = self
        node.specialization_name = 'Test'
        return node

pointer_type = convert_to_ctypes(minitypes.Py_ssize_t.pointer())
NULL = pointer_type()

def test_llvm():
    """
    >>> test_llvm()
    """
    type1 = Py_ssize_t[:, :]
    type2 = double[:, :]
    type1.broadcasting = (False, False)
    type2.broadcasting = (False, True)

    var1, var2 = vars = build_vars(type1, type2)
    expr = b.add(var1, b.add(var2, var2))
    expr = b.add(var1, var2)
    body = b.assign(var1, expr)

    #vars = build_vars(type1)
    #body = b.return_(b.constant(0))
    func = build_function(vars, body)
    print func.type
    print func
    result_ast, (lfunc, ctypes_func) = specialize(S, func, context)

    print lfunc
    ctypes_func(NULL, NULL, NULL)
    # result_ast, code_output = specialize(cinner, func, context)
    # print code_output[0]
    # print code_output[1]


#if __name__ == '__main__':
#    import doctest
#    doctest.testmod()