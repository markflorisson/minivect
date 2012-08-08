from testutils import *

context = get_llvm_context()

def test_llvm():
    """
    >>> test_llvm()
    <BLANKLINE>
    define i32 @functioninner_contig_c(i32*, double*, i32*, double*, i32*) {
    entry_0:
      %5 = bitcast i32* %4 to i32*
      %6 = load i32* %5
      %7 = udiv i32 %6, 8
      %8 = bitcast i32* %2 to i32*
      %9 = load i32* %8
      %10 = udiv i32 %9, 8
      %temp8 = alloca double*
      store double* %1, double** %temp8
      br label %for.cond_1
    <BLANKLINE>
    for.cond_1:                                       ; preds = %for.incr_2, %entry_0
      %temp6.0 = phi double* [ %3, %entry_0 ], [ %31, %for.incr_2 ]
      %temp2.0 = phi i32 [ 0, %entry_0 ], [ %14, %for.incr_2 ]
      %11 = bitcast i32* %0 to i32*
      %12 = load i32* %11
      %13 = icmp slt i32 %temp2.0, %12
      br i1 %13, label %for.body_3, label %for.exit_4
    <BLANKLINE>
    for.incr_2:                                       ; preds = %for.exit_8
      %14 = add i32 %temp2.0, 1
      br label %for.cond_1
    <BLANKLINE>
    for.body_3:                                       ; preds = %for.cond_1
      %hoisted_temp3 = alloca double
      %15 = bitcast double* %temp6.0 to double*
      %16 = load double* %15
      store double %16, double* %hoisted_temp3
      %temp1 = alloca i32
      store i32 0, i32* %temp1
      br label %for.cond_5
    <BLANKLINE>
    for.exit_4:                                       ; preds = %for.cond_1
      ret i32 0
    <BLANKLINE>
    for.cond_5:                                       ; preds = %for.incr_6, %for.body_3
      %17 = load i32* %temp1
      %18 = getelementptr i32* %0, i32 1
      %19 = load i32* %18
      %20 = icmp slt i32 %17, %19
      br i1 %20, label %for.body_7, label %for.exit_8
    <BLANKLINE>
    for.incr_6:                                       ; preds = %for.body_7
      %21 = load i32* %temp1
      %22 = add i32 %21, 1
      store i32 %22, i32* %temp1
      br label %for.cond_5
    <BLANKLINE>
    for.body_7:                                       ; preds = %for.cond_5
      %23 = getelementptr double** %temp8, i32* %temp1
      %24 = load double** %23
      %25 = load double** %temp8
      %26 = load i32* %temp1
      %27 = getelementptr double* %25, i32 %26
      %28 = load double* %27
      %29 = load double* %hoisted_temp3
      %30 = fadd double %28, %29
      store double %30, double* %24
      br label %for.incr_6
    <BLANKLINE>
    for.exit_8:                                       ; preds = %for.cond_5
      %31 = getelementptr double* %temp6.0, i32 %7
      %32 = load double** %temp8
      %33 = getelementptr double* %32, i32 %10
      store double* %33, double** %temp8
      br label %for.incr_2
    }
    <BLANKLINE>
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

    result_ast, code_output = specialize(cinner, func, context) #, print_tree=True)
    print code_output[0]
    # print code_output[1]


if __name__ == '__main__':
    import doctest
    doctest.testmod()