program main
    implicit none
    include 'omp_lib.h'
    integer :: i
    integer, parameter :: S = 2000
    double precision, pointer, dimension(:, :) :: a, b, c, d, e

    double precision t

    allocate(a(S, S), b(S, S), c(S, S), d(S, S), e(S, S))
    a = 0.0
    b = 0.0
    c = 0.0
    d = 0.0
    e = 0.0

    t = omp_get_wtime()
    do i = 1, 100
        call expr1(a, b)
    end do
    print *, "expr1", omp_get_wtime() - t

    t = omp_get_wtime()
    do i = 1, 100
        call expr2(a, b, c, d, e)
    end do
    print *, "expr2", omp_get_wtime() - t

    t = omp_get_wtime()
    do i = 1, 100
        call expr3(a, b, c, d, e)
    end do
    print *, "expr3", omp_get_wtime() - t

    print *, sum(a)

contains
    subroutine expr1(a, b)
        implicit none
        double precision, dimension(:, :) :: a, b
        a(:, :) = a(:, :) + transpose(b(:, :))
    end subroutine

    subroutine expr2(a, b, c, d, e)
        implicit none
        double precision, dimension(:, :) :: a, b, c, d, e
        a(:, :) = a + transpose(b) + c + transpose(d) + e
    end subroutine

    subroutine expr3(a, b, c, d, e)
        implicit none
        double precision, dimension(:, :) :: a, b, c, d, e
        a(:, :) = transpose(a) + b + transpose(c) + &
                  d + transpose(e)
    end subroutine
 end program
