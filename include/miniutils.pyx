cdef public int broadcast_arrays(list arrays, tuple broadcast_shape, int ndim,
                                 cnp.npy_intp **shape_out,
                                 cnp.npy_intp **strides_out) except -1:
    """
    Broadcast the given arrays. Returns the total broadcast shape of size
    ndim in shape_out. Returns the strides for each array in strides_out
    in an array of size len(arrays) * ndim.

    Note: Leading broadcasting dimensions should not have strides if the
          minivect function was compiled using an array type with less
          dimensions than `ndim`. Instead, the actual strides should be given
          without leading zero strides.
    """
    cdef cnp.ndarray array
    cdef int i, j, start, stop

    cdef cnp.npy_intp *shape = <cnp.npy_intp *> stdlib.malloc(
                                              ndim * sizeof(cnp.npy_intp))
    cdef cnp.npy_intp *strides_list = <cnp.npy_intp *> stdlib.malloc(
                                len(arrays) * ndim * sizeof(cnp.npy_intp))

    if shape == NULL or strides_list == NULL:
        raise MemoryError

    # Build a shape list of size ndim
    for i in range(ndim):
        shape[i] = broadcast_shape[i]

    # Build a strides list for all arrays of size ndim
    for i, array in enumerate(arrays):
        start = i * ndim

        if array.ndim < ndim:
            # broadcast leading dimensions
            for j in range(start, start + ndim - array.ndim):
                strides_list[j] = 0
            start = j

        for j in range(ndim):
            if array.shape[j] == 1:
                strides_list[start + j] = 0
            else:
                strides_list[start + j] = cnp.PyArray_STRIDE(array, j)

    # for j in range(ndim):
    #     print 'shape%d:' % j, shape[j]

    # for i in range(len(arrays)):
    #     for j in range(ndim):
    #         print 'stride%d:' % j, strides_list[i * ndim + j]
    #     print

    shape_out[0] = shape
    strides_out[0] = strides_list
    return 0

cdef public bint is_broadcasting(list arrays, broadcast) except -1:
    for array in arrays:
        if broadcast.nd != array.ndim or array.shape != broadcast.shape:
            return True

    return False

cdef public inline int build_dynamic_args(
        list arrays, cnp.npy_intp *strides_list,
        void ***data_pointers_out, cnp.npy_intp ***strides_list_out,
        int ndim) except -1:

    cdef void **data_pointers = <void **> stdlib.malloc(len(arrays) *
                                                        sizeof(void **))
    cdef cnp.npy_intp **strides_pointers = <cnp.npy_intp **> stdlib.malloc(
                                        len(arrays) * sizeof(cnp.npy_intp **))
    if data_pointers == NULL or strides_pointers == NULL:
        raise MemoryError

    cdef cnp.ndarray array
    cdef int i

    for i, array in enumerate(arrays):
        data_pointers[i] = cnp.PyArray_DATA(array)
        strides_pointers[i] = &strides_list[i * ndim]

    data_pointers_out[0] = data_pointers
    strides_list_out[0] = strides_pointers
    return 0