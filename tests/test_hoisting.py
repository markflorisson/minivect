from testutils import *

def test_hoist():
    """
    >>> test_hoist()
    """
    type1 = double[:, :]
    type2 = double[:, :]
    type1.broadcasting = (False, False)
    type2.broadcasting = (False, True)

    var1, var2 = vars = build_vars(type1, type2)
    expr = b.add(var1, b.add(var2, var2))
    expr = b.add(var1, var2)
    body = b.assign(var1, expr)
    func = build_function(vars, body)

    result_ast, code_output = specialize(cinner, func)
    e = toxml(result_ast)
    assert e.xpath('not(//NDIterate)')

    # Check the loop level of the hoisted expression
    op1, op2 = e.xpath(
        '//FunctionNode//ArrayFunctionArgument/DataPointer/@value')
    broadcasting_pointer_temp, = e.xpath(
        '//AssignmentExpr[./rhs/DataPointer[@value="%s"]]/lhs/TempNode/@value' % op2)

    q = '//ForNode[.//AssignmentExpr/rhs//TempNode[@value="%s"]]/@loop_level'
    loop_level, = e.xpath(q % broadcasting_pointer_temp)
    assert loop_level == "0", loop_level

if __name__ == '__main__':
    import doctest
    doctest.testmod()