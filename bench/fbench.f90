subroutine aplusb_ff(a, b, size1, size2) BIND(C, name="aplusb_ff")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2
    REAL(C_DOUBLE), dimension(size1, size2), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1, size2), intent(in) :: b

    a(:, :) = a + b
end subroutine

subroutine aplusb_fc(a, b, size1, size2) BIND(C, name="aplusb_fc")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2
    REAL(C_DOUBLE), dimension(size1, size2), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1, size2), intent(in) :: b

    a(:, :) = a + transpose(b)
end subroutine

subroutine aplusb_strided_cc(a, b, size1, size2, stride) BIND(C, name="aplusb_strided_cc")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2, stride

    REAL(C_DOUBLE), dimension(size2 * stride, size1), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size2 * stride, size1), intent(in) :: b

    ! same as _ff
    ! instead, slice in the first dimension
     a(::stride, :) = a(::stride, :) + b(::stride, :)
end subroutine

subroutine aplusb_strided_fc(a, b, size1, size2, stride) BIND(C, name="aplusb_strided_fc")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2, stride
    REAL(C_DOUBLE), dimension(size1 * stride, size2), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1 * stride, size2), intent(in) :: b

    a(::stride, :) = a(::stride, :) + transpose(b(::stride, :))
end subroutine
