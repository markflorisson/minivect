#include "Python.h"

#ifndef CYTHON_RESTRICT
  #if defined(__GNUC__)
    #define CYTHON_RESTRICT __restrict__
  #elif defined(_MSC_VER)
    #define CYTHON_RESTRICT __restrict
  #elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define CYTHON_RESTRICT restrict
  #else
    #define CYTHON_RESTRICT
  #endif
#endif

static int __mini_mangle___pyx_array_expression17tiled_c(Py_ssize_t const *const CYTHON_RESTRICT __mini_mangle___pyx_shape, double *const CYTHON_RESTRICT __pyx_op1_data, Py_ssize_t const *const CYTHON_RESTRICT __pyx_op1_strides, double const *const CYTHON_RESTRICT __pyx_op2_data, Py_ssize_t const *const CYTHON_RESTRICT __pyx_op2_strides, double const *const CYTHON_RESTRICT __pyx_op3_data, Py_ssize_t const *const CYTHON_RESTRICT __pyx_op3_strides, double const *const CYTHON_RESTRICT __pyx_op4_data, Py_ssize_t const *const CYTHON_RESTRICT __pyx_op4_strides, double const *const CYTHON_RESTRICT __pyx_op5_data, Py_ssize_t const *const CYTHON_RESTRICT __pyx_op5_strides, double const *const CYTHON_RESTRICT __pyx_op6_data, Py_ssize_t const *const CYTHON_RESTRICT __pyx_op6_strides, Py_ssize_t const __mini_mangle_blocksize, Py_ssize_t const __mini_mangle_omp_size) {
    Py_ssize_t __mini_mangle_temp0;
    Py_ssize_t __mini_mangle_temp1;
    __mini_mangle_temp0 = ((__mini_mangle___pyx_shape[0]) * (__mini_mangle___pyx_shape[1]));
    #ifdef _OPENMP
    #pragma omp parallel for if((__mini_mangle_temp0 > __mini_mangle_omp_size))
    #endif
    for (__mini_mangle_temp1 = 0; __mini_mangle_temp1 < (__mini_mangle___pyx_shape[0]); __mini_mangle_temp1 = (__mini_mangle_temp1 + __mini_mangle_blocksize)) {
        Py_ssize_t __mini_mangle_temp2;
        for (__mini_mangle_temp2 = 0; __mini_mangle_temp2 < (__mini_mangle___pyx_shape[1]); __mini_mangle_temp2 = (__mini_mangle_temp2 + __mini_mangle_blocksize)) {
            Py_ssize_t __mini_mangle_temp3;
            Py_ssize_t __mini_mangle_temp4;
            Py_ssize_t __mini_mangle_temp5;
            __mini_mangle_temp3 = (((__mini_mangle_temp1 + __mini_mangle_blocksize) < (__mini_mangle___pyx_shape[1])) ? (__mini_mangle_temp1 + __mini_mangle_blocksize) : (__mini_mangle___pyx_shape[1]));
            __mini_mangle_temp4 = (((__mini_mangle_temp2 + __mini_mangle_blocksize) < (__mini_mangle___pyx_shape[0])) ? (__mini_mangle_temp2 + __mini_mangle_blocksize) : (__mini_mangle___pyx_shape[0]));
            for (__mini_mangle_temp5 = __mini_mangle_temp2; __mini_mangle_temp5 < __mini_mangle_temp4; __mini_mangle_temp5++) {
                double *CYTHON_RESTRICT __mini_mangle_temp6;
                double const *CYTHON_RESTRICT __mini_mangle_temp7;
                double const *CYTHON_RESTRICT __mini_mangle_temp8;
                double const *CYTHON_RESTRICT __mini_mangle_temp9;
                double const *CYTHON_RESTRICT __mini_mangle_temp10;
                double const *CYTHON_RESTRICT __mini_mangle_temp11;
                Py_ssize_t __mini_mangle_temp12;
                __mini_mangle_temp6 = ((double *) (((char *) __pyx_op1_data) + (__mini_mangle_temp5 * (__pyx_op1_strides[0]))));
                __mini_mangle_temp7 = ((double *) (((char *) __pyx_op2_data) + (__mini_mangle_temp5 * (__pyx_op2_strides[0]))));
                __mini_mangle_temp8 = ((double *) (((char *) __pyx_op3_data) + (__mini_mangle_temp5 * (__pyx_op3_strides[0]))));
                __mini_mangle_temp9 = ((double *) (((char *) __pyx_op4_data) + (__mini_mangle_temp5 * (__pyx_op4_strides[0]))));
                __mini_mangle_temp10 = ((double *) (((char *) __pyx_op5_data) + (__mini_mangle_temp5 * (__pyx_op5_strides[0]))));
                __mini_mangle_temp11 = ((double *) (((char *) __pyx_op6_data) + (__mini_mangle_temp5 * (__pyx_op6_strides[0]))));
                #ifdef __INTEL_COMPILER
                #pragma simd
                #endif
                for (__mini_mangle_temp12 = __mini_mangle_temp1; __mini_mangle_temp12 < __mini_mangle_temp3; __mini_mangle_temp12++) {
                    (*((double *CYTHON_RESTRICT) (((char *) __mini_mangle_temp6) + (__mini_mangle_temp12 * (__pyx_op1_strides[1]))))) = (((((*((double const *CYTHON_RESTRICT) (((char *) __mini_mangle_temp7) + (__mini_mangle_temp12 * (__pyx_op2_strides[1]))))) + (*((double const *CYTHON_RESTRICT) (((char *) __mini_mangle_temp8) + (__mini_mangle_temp12 * (__pyx_op3_strides[1])))))) + (*((double const *CYTHON_RESTRICT) (((char *) __mini_mangle_temp9) + (__mini_mangle_temp12 * (__pyx_op4_strides[1])))))) + (*((double const *CYTHON_RESTRICT) (((char *) __mini_mangle_temp10) + (__mini_mangle_temp12 * (__pyx_op5_strides[1])))))) + (*((double const *CYTHON_RESTRICT) (((char *) __mini_mangle_temp11) + (__mini_mangle_temp12 * (__pyx_op6_strides[1]))))));
                }
            }
        }
    }
    return 0;
}



void *list[] = {
    __mini_mangle___pyx_array_expression17tiled_c,
};

