import pytest

from llvm_testutils import *

def build_expr(type):
    out, v1, v2, v3 = vars = build_vars(type, type, type, type)
    expr = b.assign(out, b.add(v1, b.mul(v2, v3)))
    return vars, expr

def build_kernels(specialization_name, **kw):
    funcs = []
    for ndim in range(1, 4):
        vars, expr = build_expr(minitypes.ArrayType(float_, ndim, **kw))
        funcs.append(MiniFunction(specialization_name, vars, expr))

    return funcs

arrays2d = [get_array(), get_array(), get_array()]
arrays1d = [a[0] for a in arrays2d]
arrays3d = [a[:, None, :] for a in arrays2d]
arrays = [(arrays1d, arrays2d, arrays3d)]

@pytest.mark.parametrize(("arrays", "specializer_names"),
                         [(arrays, ['contig'])])
def test_specializations(arrays, specializer_names):
    """
    >>> test_specializations(arrays, ['contig'])
    contig
    """
    for name in specializer_names:
        print name
        f1d, f2d, f3d = build_kernels(name)
        for (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) in arrays:
            assert np.all(f1d(x1, y1, z1) == x1 + y1 * z1)
            assert np.all(f2d(x2, y2, z2) == x2 + y2 * z2)
            assert np.all(f3d(x3, y3, z3) == x3 + y3 * z3)

