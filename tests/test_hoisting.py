from testutils import *

def test_hoist():
    """
    >>> result_ast, code_output = test_hoist()
    >> result_ast.print_tree(context)
    >>> print code_output[1]
    """
    type1 = double[:, :]
    type2 = double[:]
    type1.broadcasting = (False, True)
    type2.broadcastig = (False, False)

    var1, var2 = vars = build_vars(type1, type2)
    body = b.assign(var1, b.add(var1, var2))
    func = build_function(vars, body)

    specializer_cls = specializers.StridedFortranInnerContigSpecializer
    #specializer_cls = specializers.CTiledStridedSpecializer
    result_ast, code_output = specialize(specializer_cls, func)
    return result_ast, code_output


if __name__ == '__main__':
    import doctest
    doctest.testmod()