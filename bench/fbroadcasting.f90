program main
    use omp_lib
    implicit none

    integer, parameter :: N = 200
    double precision, dimension(N) :: a, b, c
    double precision :: output(N, N, N)
    integer :: i, j, k
    integer :: bounds(6)
    double precision :: t
    integer :: P(2, 2)

    do i = 1, N
        a(i) = i - 1
        b(i) = i - 1
        c(i) = i - 1
    end do

    bounds = (/ 1, 2, 1, 3, 2, 3 /)
    !output(:, :, :) = spread(spread(a(1, 1, :), 1, N) * spread(b(1, :, 1), 2, N), 1, N) &
    !                          * spread(spread(c(:, 1, 1), 2, N), 3, N)
    !output(:, :, :) = spread(spread(a, 1, N), 2, N) * spread(spread(b, 1, N), 3, N) &
    !                          * spread(spread(c, 2, N), 3, N)
    !output(:, :, :) = spread(spread(a, 1, N) * spread(b, 2, N), 1, N) &
    !                          * spread(spread(c, 2, N), 3, N)
    !do i = 1, N
    !    do j = 1, N
    !        do k = 1, N
    !            if (output(i, j, k) .ne. (i-1) * (j-1) * (k-1)) then
    !                print *, "not equal", i-1, j-1, k-1, output(i, j, k), &
    !                                      (i-1) * (j-1) * (k-1)
    !                stop
    !            end if
    !        end do
    !    end do
    !end do

    !do i = 1, 100
    !    output(:, :, :) = spread(spread(a, 1, N) * spread(b, 2, N), 1, N) &
    !                          * spread(spread(c, 2, N), 3, N)
    !end do

    t = omp_get_wtime()
    do i = 1, 10
        output(:, :, :) = spread(spread(a, bound(1), N), bound(2), N) * &
                                 spread(spread(b, bound(3), N), bound(4), N) * &
                                 spread(spread(c, bound(5), N), bound(6), N)
    end do
    t = omp_get_wtime() - t
    print *, t

contains
    integer function bound(i)
        implicit  none
        integer :: i
        bound = bounds(i)
    end function
end program main
