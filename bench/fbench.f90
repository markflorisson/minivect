subroutine aplusb_ff(a, b, size1, size2) BIND(C, name="aplusb_ff")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2
    REAL(C_DOUBLE), dimension(size1, size2), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1, size2), intent(in) :: b

    a(:, :) = a + b
end subroutine

subroutine aplusb_fcf(a, b, c, size1, size2) BIND(C, name="aplusb_fcf")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2
    REAL(C_DOUBLE), dimension(size1, size2), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1, size2), intent(in) :: b, c

    a(:, :) = transpose(b) + c
end subroutine

! "a.T[:, :] = a + b.T + c + d.T + e + f.T" 
subroutine aplusb_cfcfcf(a, b, c, d, e, f, size1, size2, stride) BIND(C, name="aplusb_cfcfcf")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2, stride
    REAL(C_DOUBLE), dimension(size1 * stride, size2 * stride), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1 * stride, size2 * stride), intent(in) :: b, c, d, e, f

    a(::2, ::2) = b(::2, ::2) + &
                  transpose(c(::2, ::2)) + d(::2, ::2) + &
                  transpose(e(::2, ::2)) + transpose(f(::2, ::2))
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

subroutine innercontig2d(a, b, c, d, size1, size2, stride) BIND(C, name="innercontig2d")
    use, intrinsic :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE, intent(in) :: size1, size2, stride
    REAL(C_DOUBLE), dimension(size1, size2 * stride), intent(inout) :: a
    REAL(C_DOUBLE), dimension(size1, size2 * stride), intent(in) :: b, c, d

    a(:, ::stride) = a(:, ::stride) + b(:, ::stride) + &
                     c(:, ::stride) + d(:, ::stride)
end subroutine
