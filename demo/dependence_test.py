"""
Determine data dependences between two arbitrary numpy arrays.
See thesis/thesis.pdf section 5.2.
"""

import fractions

import numpy as np

a = np.empty((10, 10))

def get_start_end(a):
    start = end = a.ctypes.data
    for i in range(a.ndim):
        if a.shape[i] == 0:
            # Empty array, no overlap possible
            return 0, 0
        else:
            offset = a.strides[i] * (a.shape[i] - 1)
            if a.strides[i] > 0:
                end += offset
            else:
                start += offset

    return start, end

def overlap(a, b):
    """
    >>> overlap(a, a)
    True
    >>> overlap(a[:5], a[5:])
    False
    >>> overlap(a[:6], a[5:])
    True
    >>> overlap(a[2:, 5:], a[5:, :6])
    True
    >>> overlap(a[2:3, 5:6], a[2:, 6:])
    False
    >>> overlap(a[2:3, 5:6], a[2:, 5:])
    True
    """
    start1, end1 = get_start_end(a)
    start2, end2 = get_start_end(b)

    if start1 < start2:
        return end1 > start2
    else:
        return end2 > start1

def verify_containment(base, a):
    base_start, base_end = get_start_end(base)
    a_start, a_end = get_start_end(a)
    assert a_start >= base_start and a_end <= base_end

def verify_base(base, a, b):
    # verify ndim
    assert base.ndim == a.ndim

    # verify pointers
    verify_containment(base, a)
    verify_containment(base, b)

    # verify strides
    negative = base.strides[0] < 0
    for i in range(base.ndim):
        # base strides may not be 0, and must be either all positive, or all negative
        assert base.strides[i] != 0
        assert (base.strides[i] < 0) == negative

        assert a.strides[i] % base.strides[i] == 0, (a.strides[i], base.strides[i])
        assert b.strides[i] % base.strides[i] == 0, (b.strides[i], base.strides[i])

def array_order(a, is_base=False):
    """
    Return the dimension indices corresponding to ascending stride order

    >>> a = np.empty((10, 10, 10), order='C')
    >>> array_order(a)
    [2, 1, 0]
    >>> array_order(a[::-1, ::-1])
    [2, 1, 0]
    >>> array_order(a.T)
    [0, 1, 2]
    >>> array_order(a.swapaxes(0, 1))
    [2, 0, 1]
    """
    stride_index_pairs = zip((abs(stride) for stride in a.strides), range(a.ndim))
    stride_index_pairs.sort()
    order = [index for stride, index in stride_index_pairs]
    if is_base and a.strides[0] < 0:
        return order[::-1]
    return order

def dimensional_independence(a):
    """
    >>> dimensional_independence(a)
    >>> dimensional_independence(a[:1, ::2])
    >>> dimensional_independence(a[::-2, ::-2])
    >>> d = a.copy()
    >>> d.strides = (10, 2) # 2 * 10 = 20, so columns overlap with rows
    >>> dimensional_independence(d)
    Traceback (most recent call last):
       ...
    AssertionError: (10, 18)
    """
    order = array_order(a)
    for previous_dim, dim in zip(order, order[1:]):
        extent = a.strides[previous_dim] * (a.shape[previous_dim] - 1)
        assert abs(a.strides[dim]) >= abs(extent), (a.strides[dim], extent)

def verify_dimensional_constraints(base, a, offsets, steps):
    for i, (offset, step) in enumerate(zip(offsets, steps)):
        assert 0 <= offset + (a.shape[i] - 1) * step < base.shape[i]

def offsets(base, a):
    """
    >>> offsets(a, a[2:, 4:])
    (2, 4)
    >>> offsets(a, a[::-1, ::-1])
    (9, 9)
    >>> offsets(a, a[::-1])
    (9, 0)
    >>> offsets(a, a[:, ::-1])
    (0, 9)
    """
    # assert array_order(base) == array_order(a)
    offsets = [None] * base.ndim
    distance = a.ctypes.data - base.ctypes.data
    for dim in reversed(array_order(base, is_base=True)):
        offsets[dim] = distance // base.strides[dim]
        assert 0 <= offsets[dim] < base.shape[dim], (offsets[dim], base.shape[dim])
        distance = distance % base.strides[dim]

    assert distance == 0

    return tuple(offsets)

def steps(base, a):
    steps = []
    for i in range(base.ndim):
        steps.append(a.strides[i] // base.strides[i])
    return steps

def verify_solution(base, a, offsets_a, steps_a):
    slices = []
    for offset, step, extent in zip(offsets_a, steps_a, a.shape):
        slices.append(slice(offset, offset + extent * step, step))

    np.all(base[tuple(slices)] == a)

def verify(base, a, b):
    """
    >>> verify(a, a[::2, ::2], a[1::2, 1::2])
    ((0, 0), [2, 2], (1, 1), [2, 2])
    >>> verify(a, a, a[::-1, ::-1])
    ((0, 0), [1, 1], (9, 9), [-1, -1])
    >>> verify(a, a, a[:, ::-1])
    ((0, 0), [1, 1], (0, 9), [1, -1])
    >>> verify(a, a[:, ::-1], a[::-1, :])
    ((0, 9), [1, -1], (9, 0), [-1, 1])
    >>> verify(a, a[:, 2::2], a[:, 8:0:-2])
    ((0, 2), [1, 2], (0, 8), [1, -2])
    """
    assert a.shape == b.shape

    verify_base(base, a, b)

    dimensional_independence(base)
    dimensional_independence(a)
    dimensional_independence(b)


    offsets_a = offsets(base, a)
    steps_a = steps(base, a)
    verify_dimensional_constraints(base, a, offsets_a, steps_a)

    offsets_b = offsets(base, b)
    steps_b = steps(base, b)
    verify_dimensional_constraints(base, b, offsets_b, steps_b)

    verify_solution(base, a, offsets_a, steps_a)
    verify_solution(base, b, offsets_b, steps_b)

    return (offsets_a, steps_a, offsets_b, steps_b)

def siv(base, a, b, offsets_a, steps_a, offsets_b, steps_b):
    direction_vector = []
    for offset_a, offset_b, step_a, step_b, extent in zip(offsets_a, offsets_b,
                                                          steps_a, steps_b, a.shape):
        distance = offset_b - offset_a
        if step_a == step_b:
            # SIV test
            if distance % step_a == 0 and abs(distance) < extent:
                # dependence
                if distance < 0:
                    direction_vector.append('>')
                elif distance == 0:
                    direction_vector.append('=')
                else:
                    direction_vector.append('<')
            else:
                # independence
                return None
        elif distance % fractions.gcd(step_a, step_b) != 0:
            # independence
            return None
        elif abs(step_a) == abs(step_b):
            # Weak crossing SIV test
            i = distance / (2*step_a)
            if i % 0.5 == 0 and 0 <= i < extent:
                # dependence
                direction_vector.append('*')
            else:
                # independence
                return None
        else:
            # possible dependence, further checking is needed (try banerjee). Remain conservative
            direction_vector.append('*')

    return tuple(direction_vector)

def direction_vector(a, b):
    """
    Returns False in case of independence, or a direction vector if no
    independence is proven

    >>> [direction_vector(*slice_pair) for slice_pair in slice_pairs]
    [False, False, False, False, False, False, ('*', '=')]

    # Dependent, test direction vectors
    >>> direction_vector(a[::2], a[8::-2])
    ('*', '=')
    >>> direction_vector(a[::2, 1:5], a[8::-2, 0:4])
    ('*', '>')
    >>> direction_vector(a[::2, 1:5], a[8::-2, 2:6])
    ('*', '<')
    """
    if not overlap(a, b):
        return False

    assert a.shape == b.shape

    base = a
    while base.base is not None:
        base = base.base

    base_b = b
    while base_b.base is not None:
        base_b = base_b.base

    if not (base.ctypes.data == base_b.ctypes.data and
            base.shape == base_b.shape and
            base.strides == base_b.strides):
        return True # assume dependence

    offsets_steps = verify(base, a, b)

    result = siv(base, a, b, *offsets_steps)
    if result is None:
        return False # no dependence

    return result # direction vector

def general_gcd_test(a, b):
    """
    Returns False in case of independence, otherwise there may or may not
    be a dependence

    >>> [general_gcd_test(*slice_pair) for slice_pair in slice_pairs]
    [False, True, False, False, True, True, True]
    """
    # perform the GCD test without prior transformation
    if overlap(a, b):
        gcd = fractions.gcd(reduce(fractions.gcd, a.strides),
                            reduce(fractions.gcd, a.strides))
        return (a.ctypes.data - b.ctypes.data) % gcd == 0

    return False


slice_pairs = [
    # independent
    (a[::2, ::2], a[1::2, 1::2]),
    (a[::2], a[::-2]),
    (a[:, ::2], a[:, ::-2]),
    (a[:5], a[5:]),
    (a[:, :5], a[:, 5:]),
    (a[1:3, 4:8], a[2:4, 0:4]),
    (a[0:4:3], a[2:5:2]), # rows (0, 3) and (2, 4)
]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
