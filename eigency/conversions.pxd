cimport numpy as np
from libc.stdint cimport (int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                          uint32_t, uint64_t)

import numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# read https://numpy.org/devdocs/numpy_2_0_migration_guide.html#the-pyarray-descr-struct-has-been-changed
# for more information about this lien
cdef extern from "npy2_compat.h":
    pass

# floats
cdef api np.ndarray[float, ndim=2] ndarray_float()
cdef api np.ndarray[float, ndim=2] ndarray_float_C(float *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[float, ndim=2] ndarray_float_F(float *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[float, ndim=2] ndarray_copy_float_C(const float *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[float, ndim=2] ndarray_copy_float_F(const float *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[double, ndim=2] ndarray_double()
cdef api np.ndarray[double, ndim=2] ndarray_double_C(double *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[double, ndim=2] ndarray_double_F(double *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[double, ndim=2] ndarray_copy_double_C(const double *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[double, ndim=2] ndarray_copy_double_F(const double *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[long double, ndim=2] ndarray_long_double()
cdef api np.ndarray[long double, ndim=2] ndarray_long_double_C(long double *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[long double, ndim=2] ndarray_long_double_F(long double *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[long double, ndim=2] ndarray_copy_long_double_C(const long double *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[long double, ndim=2] ndarray_copy_long_double_F(const long double *data, long rows, long cols, long outer_stride, long inner_stride)


# int

cdef api np.ndarray[int8_t, ndim=2] ndarray_int8()
cdef api np.ndarray[int8_t, ndim=2] ndarray_int8_C(int8_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int8_t, ndim=2] ndarray_int8_F(int8_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int8_t, ndim=2] ndarray_copy_int8_C(const int8_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int8_t, ndim=2] ndarray_copy_int8_F(const int8_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[int16_t, ndim=2] ndarray_int16()
cdef api np.ndarray[int16_t, ndim=2] ndarray_int16_C(int16_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int16_t, ndim=2] ndarray_int16_F(int16_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int16_t, ndim=2] ndarray_copy_int16_C(const int16_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int16_t, ndim=2] ndarray_copy_int16_F(const int16_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[int32_t, ndim=2] ndarray_int32()
cdef api np.ndarray[int32_t, ndim=2] ndarray_int32_C(int32_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int32_t, ndim=2] ndarray_int32_F(int32_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int32_t, ndim=2] ndarray_copy_int32_C(const int32_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int32_t, ndim=2] ndarray_copy_int32_F(const int32_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[int64_t, ndim=2] ndarray_int64()
cdef api np.ndarray[int64_t, ndim=2] ndarray_int64_C(int64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int64_t, ndim=2] ndarray_int64_F(int64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int64_t, ndim=2] ndarray_copy_int64_C(const int64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[int64_t, ndim=2] ndarray_copy_int64_F(const int64_t *data, long rows, long cols, long outer_stride, long inner_stride)


# uint

cdef api np.ndarray[uint8_t, ndim=2] ndarray_uint8()
cdef api np.ndarray[uint8_t, ndim=2] ndarray_uint8_C(uint8_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint8_t, ndim=2] ndarray_uint8_F(uint8_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint8_t, ndim=2] ndarray_copy_uint8_C(const uint8_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint8_t, ndim=2] ndarray_copy_uint8_F(const uint8_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[uint16_t, ndim=2] ndarray_uint16()
cdef api np.ndarray[uint16_t, ndim=2] ndarray_uint16_C(uint16_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint16_t, ndim=2] ndarray_uint16_F(uint16_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint16_t, ndim=2] ndarray_copy_uint16_C(const uint16_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint16_t, ndim=2] ndarray_copy_uint16_F(const uint16_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[uint32_t, ndim=2] ndarray_uint32()
cdef api np.ndarray[uint32_t, ndim=2] ndarray_uint32_C(uint32_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint32_t, ndim=2] ndarray_uint32_F(uint32_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint32_t, ndim=2] ndarray_copy_uint32_C(const uint32_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint32_t, ndim=2] ndarray_copy_uint32_F(const uint32_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[uint64_t, ndim=2] ndarray_uint64()
cdef api np.ndarray[uint64_t, ndim=2] ndarray_uint64_C(uint64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint64_t, ndim=2] ndarray_uint64_F(uint64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint64_t, ndim=2] ndarray_copy_uint64_C(const uint64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[uint64_t, ndim=2] ndarray_copy_uint64_F(const uint64_t *data, long rows, long cols, long outer_stride, long inner_stride)


"""
cdef api np.ndarray[np.npy_clongdouble, ndim=2] ndarray_complex_long_double()
cdef api np.ndarray[np.npy_clongdouble, ndim=2] ndarray_complex_long_double_C(np.npy_clongdouble *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.npy_clongdouble, ndim=2] ndarray_complex_long_double_F(np.npy_clongdouble *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.npy_clongdouble, ndim=2] ndarray_copy_complex_long_double_C(const np.npy_clongdouble *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.npy_clongdouble, ndim=2] ndarray_copy_complex_long_double_F(const np.npy_clongdouble *data, long rows, long cols, long outer_stride, long inner_stride)
"""

cdef api np.ndarray[np.complex128_t, ndim=2] ndarray_complex_double()
cdef api np.ndarray[np.complex128_t, ndim=2] ndarray_complex_double_C(np.complex128_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.complex128_t, ndim=2] ndarray_complex_double_F(np.complex128_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.complex128_t, ndim=2] ndarray_copy_complex_double_C(const np.complex128_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.complex128_t, ndim=2] ndarray_copy_complex_double_F(const np.complex128_t *data, long rows, long cols, long outer_stride, long inner_stride)

cdef api np.ndarray[np.complex64_t, ndim=2] ndarray_complex_float()
cdef api np.ndarray[np.complex64_t, ndim=2] ndarray_complex_float_C(np.complex64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.complex64_t, ndim=2] ndarray_complex_float_F(np.complex64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.complex64_t, ndim=2] ndarray_copy_complex_float_C(const np.complex64_t *data, long rows, long cols, long outer_stride, long inner_stride)
cdef api np.ndarray[np.complex64_t, ndim=2] ndarray_copy_complex_float_F(const np.complex64_t *data, long rows, long cols, long outer_stride, long inner_stride)
