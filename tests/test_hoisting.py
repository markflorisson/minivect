from testutils import *

def test_hoist():
    """
    >>> result_ast, code_output = test_hoist()
    >>> result_ast.for_loops[0].body.stats[-3].print_tree(context)
    >>> print code_output[1]
    """
    type1 = double[:, :, :]
    type2 = double[:, :, :]
    type1.broadcasting = (False, True)
    type2.broadcastig = (False, False)

    var1, var2 = vars = build_vars(type1, type2)
    body = b.assign(var1, b.add(var1, var2))
    func = build_function(vars, body)

    # result_ast, code_output = specialize(cinner_sse, func)
    for result_ast, code_output in run([cinner_sse], func):
        e = toxml(result_ast)
        assert e.xpath('not(//NDIterate)')
        print code_output

    return 1, 2
    # return result_ast, code_output


if __name__ == '__main__':
    import doctest
    doctest.testmod()