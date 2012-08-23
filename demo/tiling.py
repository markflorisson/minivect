"""
Simple algorithm to compute tiling parameters. Based on the paper
"Tile Size Selection Using Cache Organization and Data Layout".
"""

import math

import numpy as np

def computerows(colsize, CS, N):
    cols_per_set = q1 = CS / N

    r1 = CS % N
    setdiff = N - r1
    cols_per_n = N / setdiff
    gap = N % setdiff

    if colsize == N:
        return cols_per_set
    elif colsize == r1 and colsize > setdiff:
        return cols_per_set + 1
    else:
        cols_per_setdiff = math.floor(setdiff / colsize)
        cols_per_gap = math.floor(gap / colsize)
        return int(cols_per_setdiff * cols_per_n * cols_per_set +
                   cols_per_gap * cols_per_set +
                   cols_per_setdiff * math.floor(r1 / setdiff) + cols_per_gap)

def find_tile_sizes(arrays, cache_size):
    for array in arrays:
        colsize = N = max(array.strides) / array.dtype.itemsize
        rowsize = cache_size / N
        r = cache_size % colsize
        candidates = [(rowsize, colsize)]
        while colsize > 4 and r != 0:
            tmp = colsize
            colsize = r
            r = tmp % r
            rowsize = computerows(colsize, cache_size, N)
            candidates.append((rowsize, colsize))

        if array.strides[0] < array.strides[1]:
            candidates = [c[::-1] for c in candidates]

        print array.strides, candidates
        yield candidates

def maxarea(arrays, CS):
    list_of_candidate_sets = list(find_tile_sizes(arrays, CS))

    area = 0
    tile = None
    for T1 in list_of_candidate_sets[0]:
        x0, y0 = T1
        for candidates in list_of_candidate_sets[1:]:
            curarea = 0
            for T2 in candidates:
                x1, y1 = T2
                if min(x0, x1) * min(y0, y1) > curarea:
                    cur_x = min(x0, x1)
                    cur_y = min(y0, y1)
                    curarea = cur_x * cur_y

            x0, y0 = cur_x, cur_y

        if x0 * y0 > area:
            tile = x0, y0
            area = x0 * y0

    return tile

CS = 4096

dtype = np.float32

a = np.empty((200, 200), dtype)
b = np.empty((200, 200), dtype)
arrays = a, a.T, b, b.T
print maxarea(arrays, CS / 4)

a = np.empty((160, 280), dtype)
b = np.empty((162, 282), dtype)[1:-1, 1:-1]

arrays = a, a.T, b, b.T
arrays = [a[:160, :160] for a in arrays]

print maxarea(arrays, CS / 4)

def occupy_cache(m, n, N, CS):
    result = set()
    location = 0
    for i in range(m):
        for j in range(m, m + n):
            datum = (i * N + j) % CS
            result.add(datum)

    return len(result), m * n

