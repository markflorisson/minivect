#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4096

const int associativity = 4;
const int cache_set_size = 32768; /* 32kb */

void *align_p(void *unaligned, size_t alignment) {
    uintptr_t aligned = (uintptr_t) unaligned;
    int offset = aligned % alignment;
    if (offset > 0)
        offset = alignment - offset;
    return (void *) (unaligned + offset);
}

void init(float *p) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            *p++ = i * N + j;
        }
    }
}

void
blocked(
    float *restrict a,
    float *restrict b,
    float *restrict c,
    float *restrict d,
    float *restrict e,
    float *restrict f,
    float *restrict g,
    float *restrict h,
    float *restrict k,
    float *restrict l,
    float *restrict m,
    float *restrict n,
    float *restrict o,
    float *restrict p,
    float *restrict q,
    float *restrict r)
{
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[j] = a[j] + b[j] + c[j] + d[j];
        }
        for (j = 0; j < N; j++) {
            a[j] = a[j] + e[j] + f[j] + g[j];
        }
        for (j = 0; j < N; j++) {
            a[j] = a[j] + h[j] + k[j] + l[j];
        }
        for (j = 0; j < N; j++) {
            a[j] = a[j] + m[j] + n[j] + o[j];
        }
        for (j = 0; j < N; j++) {
            a[j] = a[j] + p[j] + q[j] + r[j];
        }

        a += N;
        b += N;
        c += N;
        d += N;
        e += N;
        f += N;
        g += N;
        h += N;
        k += N;
        l += N;
        m += N;
        n += N;
        o += N;
        p += N;
        q += N;
        r += N;
    }
}

void
unblocked(
    float *restrict a,
    float *restrict b,
    float *restrict c,
    float *restrict d,
    float *restrict e,
    float *restrict f,
    float *restrict g,
    float *restrict h,
    float *restrict k,
    float *restrict l,
    float *restrict m,
    float *restrict n,
    float *restrict o,
    float *restrict p,
    float *restrict q,
    float *restrict r)
{
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[j] = (a[j] + b[j] + c[j] + d[j] + e[j] +
                    f[j] + g[j] + h[j] +
                    k[j] + l[j] + m[j] + n[j] + o[j] +
                    p[j] + q[j] + r[j]);
        }
        a += N;
        b += N;
        c += N;
        d += N;
        e += N;
        f += N;
        g += N;
        h += N;
        k += N;
        l += N;
        m += N;
        n += N;
        o += N;
        p += N;
        q += N;
        r += N;
    }
}

int main(void) {
    float *arrays[16];

    int i;
    double t;

    for (i = 0; i < 16; i++) {
        float *p = malloc(sizeof(float) * N * N + cache_set_size);
        if (!p) {
            perror("Out of memory");
            exit(1);
        }
        arrays[i] = align_p(p, cache_set_size);
        init(arrays[i]);
    }

    blocked(arrays[0], arrays[1], arrays[2],
            arrays[3], arrays[4], arrays[5],
            arrays[6], arrays[7], arrays[8],
            arrays[9], arrays[10], arrays[11],
            arrays[12], arrays[13], arrays[14],
            arrays[15]);

    t = omp_get_wtime();
    for (i = 0; i < 100; i++)
        blocked(arrays[0], arrays[1], arrays[2],
                arrays[3], arrays[4], arrays[5],
                arrays[6], arrays[7], arrays[8],
                arrays[9], arrays[10], arrays[11],
                arrays[12], arrays[13], arrays[14],
                arrays[15]);
    printf("blocked: %lf\n", omp_get_wtime() - t);

    t = omp_get_wtime();
    for (i = 0; i < 100; i++)
        unblocked(arrays[0], arrays[1], arrays[2],
                  arrays[3], arrays[4], arrays[5],
                  arrays[6], arrays[7], arrays[8],
                  arrays[9], arrays[10], arrays[11],
                  arrays[12], arrays[13], arrays[14],
                  arrays[15]);
    printf("unblocked: %lf\n", omp_get_wtime() - t);

    return 0;
}
