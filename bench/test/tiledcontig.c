#include <xmmintrin.h>
#include <mmintrin.h>
#include <smmintrin.h>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1600

static void
transpose(float *const restrict arg1, const float *const restrict arg2, const int blocksize)
{
    int i, j, ti, tj;
    const int fstride1 = N;
    const int cstride0 = N;
    for (ti = 0; ti < N; ti+=blocksize) {
        for (tj = 0; tj < N; tj+=blocksize) {
            int upper_i = ti + blocksize;
            int upper_j = tj + blocksize;
            for (i = ti; i < upper_i; i+=4) {
                float *restrict a = arg1 + i * N;
                float *restrict b = arg2 + i;
                for (j = tj; j < upper_j; j+=4) {

                    __m128 col1 = _mm_load_ps(b + j * fstride1);
                    __m128 col2 = _mm_load_ps(b + (j+1) * fstride1);
                    __m128 col3 = _mm_load_ps(b + (j+2) * fstride1);
                    __m128 col4 = _mm_load_ps(b + (j+3) * fstride1);

                    __m128 a1 = _mm_unpacklo_ps(col1, col2);
                    __m128 b1 = _mm_unpacklo_ps(col3, col4);
                    __m128 c1 = _mm_unpackhi_ps(col1, col2);
                    __m128 d1 = _mm_unpackhi_ps(col3, col4);

                    __m128 newrow1 = _mm_shuffle_ps(a1, b1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128 newrow2 = _mm_shuffle_ps(a1, b1, _MM_SHUFFLE(3, 2, 3, 2));
                    __m128 newrow3 = _mm_shuffle_ps(c1, d1, _MM_SHUFFLE(1, 0, 1, 0));
                    __m128 newrow4 = _mm_shuffle_ps(c1, d1, _MM_SHUFFLE(3, 2, 3, 2));

                    __m128 row1 = _mm_load_ps(a + j);
                    __m128 row2 = _mm_load_ps(a + j + cstride0);
                    __m128 row3 = _mm_load_ps(a + j + 2 * cstride0);
                    __m128 row4 = _mm_load_ps(a + j + 3 * cstride0);

                    row1 = _mm_add_ps(row1, newrow1);
                    row2 = _mm_add_ps(row2, newrow2);
                    row3 = _mm_add_ps(row3, newrow3);
                    row4 = _mm_add_ps(row4, newrow4);

                    _mm_store_ps(a + j, row1);
                    _mm_store_ps(a + j + cstride0, row2);
                    _mm_store_ps(a + j + 2 * cstride0, row3);
                    _mm_store_ps(a + j + 3 * cstride0, row4);
                }
            }
        }
    }
}

static void
simple_transpose(float *const restrict arg1, const float *const restrict arg2, const int blocksize)
{
    int i, j, ti, tj;
    for (ti = 0; ti < N; ti+=blocksize) {
        for (tj = 0; tj < N; tj+=blocksize) {
            int upper_i = ti + blocksize;
            int upper_j = tj + blocksize;
            for (i = ti; i < upper_i; i++) {
                float *restrict a = arg1 + i * N;
                float *restrict b = arg2 + i;
                for (j = tj; j < upper_j; j++) {
                    a[j] += b[j * N];
                }
            }
        }
    }
}

void *align_p(void *unaligned) {
    uintptr_t aligned = (uintptr_t) unaligned;
    int offset = aligned % 16;
    if (offset > 0)
        offset = 16 - offset;
    return (void *) (unaligned + offset);
}

static void init(float *a, float *b) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            a[i * N + j] = i * N + j;
            b[i + j * N] = i * N + j;
        }
    }
}

static void verify(float *a) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (a[i * N + j] != (i * N + j) * 2) {
                abort();
            }
        }
    }
}

int main(void) {
    float *a = malloc(sizeof(float) * N * N + 15);
    float *b = malloc(sizeof(float) * N * N + 15);
    float *c = malloc(sizeof(float) * N * N + 15);
    float *d = malloc(sizeof(float) * N * N + 15);

    a = align_p(a);
    b = align_p(b);
    c = align_p(c);
    d = align_p(d);

    int i;
    double t;

    init(c, d);
    simple_transpose(c, d, 64);
    verify(c);

    t = omp_get_wtime();
    for (i = 0; i < 500; i++)
        simple_transpose(c, d, 64);
    printf("normal: %lf\n", omp_get_wtime() - t);

    init(a, b);
    transpose(a, b, 64);
    verify(a);

    t = omp_get_wtime();
    for (i = 0; i < 500; i++)
        transpose(a, b, 64);
    printf("sse: %lf\n", omp_get_wtime() - t);

    return 0;
}
