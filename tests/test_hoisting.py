from testutils import *

def test_hoist():
    """
    >> test_hoist()
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

def test_hoist_3d():
    """
    >>> test_hoist_3d()
    static int __mini_mangle_functioninner_contig_c(Py_ssize_t * __mini_mangle_shape, double * op0_data, Py_ssize_t * op0_strides, double * op1_data, Py_ssize_t * op1_strides, double * op2_data, Py_ssize_t * op2_strides, double * op3_data, Py_ssize_t * op3_strides) {
        Py_ssize_t const __mini_mangle_op1_stride6 = (op1_strides[0] / sizeof(double));
        Py_ssize_t const __mini_mangle_op1_stride7 = (op1_strides[1] / sizeof(double));
        Py_ssize_t const __mini_mangle_op1_stride8 = (op1_strides[2] / sizeof(double));
        double * __mini_mangle_temp9 = op1_data;
        Py_ssize_t const __mini_mangle_op2_stride10 = (op2_strides[0] / sizeof(double));
        Py_ssize_t const __mini_mangle_op2_stride11 = (op2_strides[1] / sizeof(double));
        double * __mini_mangle_temp12 = op2_data;
        Py_ssize_t const __mini_mangle_op0_stride13 = (op0_strides[0] / sizeof(double));
        Py_ssize_t const __mini_mangle_op0_stride14 = (op0_strides[1] / sizeof(double));
        double * __mini_mangle_temp15 = op0_data;
        Py_ssize_t const __mini_mangle_op3_stride17 = (op3_strides[0] / sizeof(double));
        Py_ssize_t const __mini_mangle_op3_stride18 = (op3_strides[1] / sizeof(double));
        double * __mini_mangle_temp19 = op3_data;
        Py_ssize_t __mini_mangle_temp0 = ((__mini_mangle_shape[0] * __mini_mangle_shape[1]) * __mini_mangle_shape[2]);
        Py_ssize_t __mini_mangle_temp3;
        #ifdef _OPENMP
        #pragma omp parallel for if((__mini_mangle_temp0 > 1024)) lastprivate(__mini_mangle_temp3)
        #endif
        for (__mini_mangle_temp3 = 0; __mini_mangle_temp3 < __mini_mangle_shape[0]; __mini_mangle_temp3++) {
            double * __mini_mangle_temp16 = __mini_mangle_temp15;
            double * __mini_mangle_temp20 = __mini_mangle_temp19;
            double __mini_mangle_hoisted_temp4 = (*__mini_mangle_temp9);
            Py_ssize_t __mini_mangle_temp2;
            for (__mini_mangle_temp2 = 0; __mini_mangle_temp2 < __mini_mangle_shape[1]; __mini_mangle_temp2++) {
                double __mini_mangle_hoisted_temp5 = (__mini_mangle_hoisted_temp4 * __mini_mangle_temp12[__mini_mangle_temp2]);
                Py_ssize_t __mini_mangle_temp1;
                #ifdef __INTEL_COMPILER
                #pragma simd
                #endif
                for (__mini_mangle_temp1 = 0; __mini_mangle_temp1 < __mini_mangle_shape[2]; __mini_mangle_temp1++) {
                    __mini_mangle_temp16[__mini_mangle_temp1] = (__mini_mangle_hoisted_temp5 * __mini_mangle_temp20[__mini_mangle_temp1]);
                }
                __mini_mangle_temp16 += __mini_mangle_op0_stride14;
                __mini_mangle_temp20 += __mini_mangle_op3_stride18;
            }
            __mini_mangle_temp9 += __mini_mangle_op1_stride6;
            __mini_mangle_temp12 += __mini_mangle_op2_stride10;
            __mini_mangle_temp15 += __mini_mangle_op0_stride13;
            __mini_mangle_temp19 += __mini_mangle_op3_stride17;
        }
        return 0;
    }
    <BLANKLINE>
    """
    type1 = double[:, :, :]
    type2 = double[:, :, :]
    type3 = double[:, :, :]
    type1.broadcasting = (False, True, True)
    type2.broadcasting = (True, False, True)
    type3.broadcasting = (True, True, False)

    out_type = double[:, :, :]

    out, var1, var2, var3 = vars = build_vars(out_type, type1, type2, type3)
    expr = b.mul(b.mul(var1, var2), var3)
    body = b.assign(out, expr)
    func = build_function(vars, body)

    result_ast, code_output = specialize(cinner, func)
    print code_output


if __name__ == '__main__':
    import doctest
    doctest.testmod()