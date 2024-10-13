cimport cython

cimport numpy as np
import numpy as np
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

from numpy.lib.stride_tricks import as_strided

#
# long double (float128)
#

@cython.boundscheck(False)
cdef np.ndarray[long double, ndim=2] ndarray_long_double():
    return np.empty((0,0), dtype='longdouble')

@cython.boundscheck(False)
cdef np.ndarray[long double, ndim=2] ndarray_long_double_C(long double *data, long rows, long cols, long row_stride, long col_stride):
    cdef long double[:,:] mem_view = <long double[:rows,:cols]>data
    dtype = 'longdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[long double, ndim=2] ndarray_long_double_F(long double *data, long rows, long cols, long row_stride, long col_stride):
    cdef long double[::1,:] mem_view = <long double[:rows:1,:cols]>data
    dtype = 'longdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[long double, ndim=2] ndarray_copy_long_double_C(const long double *data, long rows, long cols, long row_stride, long col_stride):
    cdef long double[:,:] mem_view = <long double[:rows,:cols]>data
    dtype = 'longdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[long double, ndim=2] ndarray_copy_long_double_F(const long double *data, long rows, long cols, long row_stride, long col_stride):
    cdef long double[::1,:] mem_view = <long double[:rows:1,:cols]>data
    dtype = 'longdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# double (float64)
#

@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2] ndarray_double():
    return np.empty((0,0), dtype='double')

@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2] ndarray_double_C(double *data, long rows, long cols, long row_stride, long col_stride):
    cdef double[:,:] mem_view = <double[:rows,:cols]>data
    dtype = 'double'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2] ndarray_double_F(double *data, long rows, long cols, long row_stride, long col_stride):
    cdef double[::1,:] mem_view = <double[:rows:1,:cols]>data
    dtype = 'double'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2] ndarray_copy_double_C(const double *data, long rows, long cols, long row_stride, long col_stride):
    cdef double[:,:] mem_view = <double[:rows,:cols]>data
    dtype = 'double'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[double, ndim=2] ndarray_copy_double_F(const double *data, long rows, long cols, long row_stride, long col_stride):
    cdef double[::1,:] mem_view = <double[:rows:1,:cols]>data
    dtype = 'double'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# float
#

@cython.boundscheck(False)
cdef np.ndarray[float, ndim=2] ndarray_float():
    return np.empty((0,0), dtype='float')

@cython.boundscheck(False)
cdef np.ndarray[float, ndim=2] ndarray_float_C(float *data, long rows, long cols, long row_stride, long col_stride):
    cdef float[:,:] mem_view = <float[:rows,:cols]>data
    dtype = 'float'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[float, ndim=2] ndarray_float_F(float *data, long rows, long cols, long row_stride, long col_stride):
    cdef float[::1,:] mem_view = <float[:rows:1,:cols]>data
    dtype = 'float'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[float, ndim=2] ndarray_copy_float_C(const float *data, long rows, long cols, long row_stride, long col_stride):
    cdef float[:,:] mem_view = <float[:rows,:cols]>data
    dtype = 'float'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[float, ndim=2] ndarray_copy_float_F(const float *data, long rows, long cols, long row_stride, long col_stride):
    cdef float[::1,:] mem_view = <float[:rows:1,:cols]>data
    dtype = 'float'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# long
#

@cython.boundscheck(False)
cdef np.ndarray[int32_t, ndim=2] ndarray_int32():
    return np.empty((0,0), dtype='int32')

@cython.boundscheck(False)
cdef np.ndarray[int32_t, ndim=2] ndarray_int32_C(int32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int32_t[:,:] mem_view = <int32_t[:rows,:cols]>data
    dtype = 'int32'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int32_t, ndim=2] ndarray_int32_F(int32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int32_t[::1,:] mem_view = <int32_t[:rows:1,:cols]>data
    dtype = 'int32'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int32_t, ndim=2] ndarray_copy_int32_C(const int32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int32_t[:,:] mem_view = <int32_t[:rows,:cols]>data
    dtype = 'int32'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[int32_t, ndim=2] ndarray_copy_int32_F(const int32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int32_t[::1,:] mem_view = <int32_t[:rows:1,:cols]>data
    dtype = 'int32'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# int64_t
#

@cython.boundscheck(False)
cdef np.ndarray[int64_t, ndim=2] ndarray_int64():
    return np.empty((0,0), dtype=np.int64)

@cython.boundscheck(False)
cdef np.ndarray[int64_t, ndim=2] ndarray_int64_C(int64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int64_t[:,:] mem_view = <int64_t[:rows,:cols]>data
    dtype = np.dtype(np.int64)
    cdef np.npy_intp itemsize = dtype.itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int64_t, ndim=2] ndarray_int64_F(int64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int64_t[::1,:] mem_view = <int64_t[:rows:1,:cols]>data
    dtype = np.dtype(np.int64)
    cdef np.npy_intp itemsize = dtype.itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int64_t, ndim=2] ndarray_copy_int64_C(const int64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int64_t[:,:] mem_view = <int64_t[:rows,:cols]>data
    dtype = np.dtype(np.int64)
    cdef np.npy_intp itemsize = dtype.itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[int64_t, ndim=2] ndarray_copy_int64_F(const int64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int64_t[::1,:] mem_view = <int64_t[:rows:1,:cols]>data
    dtype = np.dtype(np.int64)
    cdef np.npy_intp itemsize = dtype.itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# unsigned int 32
#

@cython.boundscheck(False)
cdef np.ndarray[uint32_t, ndim=2] ndarray_uint32():
    return np.empty((0,0), dtype='uint')

@cython.boundscheck(False)
cdef np.ndarray[uint32_t, ndim=2] ndarray_uint32_C(uint32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef uint32_t[:,:] mem_view = <uint32_t[:rows,:cols]>data
    dtype = 'uint32'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[uint32_t, ndim=2] ndarray_uint32_F(uint32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.uint32_t[::1,:] mem_view = <np.uint32_t[:rows:1,:cols]>data
    dtype = 'uint'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[uint32_t, ndim=2] ndarray_copy_uint32_C(const uint32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.uint32_t[:,:] mem_view = <np.uint32_t[:rows,:cols]>data
    dtype = 'uint'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[uint32_t, ndim=2] ndarray_copy_uint32_F(const uint32_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.uint32_t[::1,:] mem_view = <np.uint32_t[:rows:1,:cols]>data
    dtype = 'uint'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))


#
# int
#
"""
@cython.boundscheck(False)
cdef np.ndarray[int, ndim=2] ndarray_int():
    return np.empty((0,0), dtype='int')

@cython.boundscheck(False)
cdef np.ndarray[int, ndim=2] ndarray_int_C(int *data, long rows, long cols, long row_stride, long col_stride):
    cdef int[:,:] mem_view = <int[:rows,:cols]>data
    dtype = 'int'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int, ndim=2] ndarray_int_F(int *data, long rows, long cols, long row_stride, long col_stride):
    cdef int[::1,:] mem_view = <int[:rows:1,:cols]>data
    dtype = 'int'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int, ndim=2] ndarray_copy_int_C(const int *data, long rows, long cols, long row_stride, long col_stride):
    cdef int[:,:] mem_view = <int[:rows,:cols]>data
    dtype = 'int'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[int, ndim=2] ndarray_copy_int_F(const int *data, long rows, long cols, long row_stride, long col_stride):
    cdef int[::1,:] mem_view = <int[:rows:1,:cols]>data
    dtype = 'int'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))
"""
#
# unsigned int
#

@cython.boundscheck(False)
cdef np.ndarray[uint64_t, ndim=2] ndarray_uint64():
    return np.empty((0,0), dtype='uint64')

@cython.boundscheck(False)
cdef np.ndarray[uint64_t, ndim=2] ndarray_uint64_C(uint64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef uint64_t[:,:] mem_view = <uint64_t[:rows,:cols]>data
    dtype = 'uint64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[uint64_t, ndim=2] ndarray_uint64_F(uint64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef uint64_t[::1,:] mem_view = <uint64_t[:rows:1,:cols]>data
    dtype = 'uint64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[uint64_t, ndim=2] ndarray_copy_uint64_C(const uint64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef uint64_t[:,:] mem_view = <uint64_t[:rows,:cols]>data
    dtype = 'uint64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[uint64_t, ndim=2] ndarray_copy_uint64_F(const uint64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef uint64_t[::1,:] mem_view = <uint64_t[:rows:1,:cols]>data
    dtype = 'uint64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# short
#

@cython.boundscheck(False)
cdef np.ndarray[int16_t, ndim=2] ndarray_int16():
    return np.empty((0,0), dtype='int16')

@cython.boundscheck(False)
cdef np.ndarray[int16_t, ndim=2] ndarray_int16_C(int16_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int16_t[:,:] mem_view = <int16_t[:rows,:cols]>data
    dtype = 'int16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int16_t, ndim=2] ndarray_int16_F(int16_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int16_t[::1,:] mem_view = <int16_t[:rows:1,:cols]>data
    dtype = 'int16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[int16_t, ndim=2] ndarray_copy_int16_C(const int16_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int16_t[:,:] mem_view = <int16_t[:rows,:cols]>data
    dtype = 'int16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[int16_t, ndim=2] ndarray_copy_int16_F(const int16_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef int16_t[::1,:] mem_view = <int16_t[:rows:1,:cols]>data
    dtype = 'int16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# unsigned short
#

@cython.boundscheck(False)
cdef np.ndarray[unsigned short, ndim=2] ndarray_uint16():
    return np.empty((0,0), dtype='uint16')

@cython.boundscheck(False)
cdef np.ndarray[unsigned short, ndim=2] ndarray_uint16_C(unsigned short *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned short[:,:] mem_view = <unsigned short[:rows,:cols]>data
    dtype = 'uint16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[unsigned short, ndim=2] ndarray_uint16_F(unsigned short *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned short[::1,:] mem_view = <unsigned short[:rows:1,:cols]>data
    dtype = 'uint16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[unsigned short, ndim=2] ndarray_copy_uint16_C(const unsigned short *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned short[:,:] mem_view = <unsigned short[:rows,:cols]>data
    dtype = 'uint16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[unsigned short, ndim=2] ndarray_copy_uint16_F(const unsigned short *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned short[::1,:] mem_view = <unsigned short[:rows:1,:cols]>data
    dtype = 'uint16'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# signed char
#

@cython.boundscheck(False)
cdef np.ndarray[signed char, ndim=2] ndarray_int8():
    return np.empty((0,0), dtype='int8')

@cython.boundscheck(False)
cdef np.ndarray[signed char, ndim=2] ndarray_int8_C(signed char *data, long rows, long cols, long row_stride, long col_stride):
    cdef signed char[:,:] mem_view = <signed char[:rows,:cols]>data
    dtype = 'int8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[signed char, ndim=2] ndarray_int8_F(signed char *data, long rows, long cols, long row_stride, long col_stride):
    cdef signed char[::1,:] mem_view = <signed char[:rows:1,:cols]>data
    dtype = 'int8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[signed char, ndim=2] ndarray_copy_int8_C(const signed char *data, long rows, long cols, long row_stride, long col_stride):
    cdef signed char[:,:] mem_view = <signed char[:rows,:cols]>data
    dtype = 'int8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[signed char, ndim=2] ndarray_copy_int8_F(const signed char *data, long rows, long cols, long row_stride, long col_stride):
    cdef signed char[::1,:] mem_view = <signed char[:rows:1,:cols]>data
    dtype = 'int8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# unsigned char
#

@cython.boundscheck(False)
cdef np.ndarray[unsigned char, ndim=2] ndarray_uint8():
    return np.empty((0,0), dtype='uint8')

@cython.boundscheck(False)
cdef np.ndarray[unsigned char, ndim=2] ndarray_uint8_C(unsigned char *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned char[:,:] mem_view = <unsigned char[:rows,:cols]>data
    dtype = 'uint8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[unsigned char, ndim=2] ndarray_uint8_F(unsigned char *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned char[::1,:] mem_view = <unsigned char[:rows:1,:cols]>data
    dtype = 'uint8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[unsigned char, ndim=2] ndarray_copy_uint8_C(const unsigned char *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned char[:,:] mem_view = <unsigned char[:rows,:cols]>data
    dtype = 'uint8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[unsigned char, ndim=2] ndarray_copy_uint8_F(const unsigned char *data, long rows, long cols, long row_stride, long col_stride):
    cdef unsigned char[::1,:] mem_view = <unsigned char[:rows:1,:cols]>data
    dtype = 'uint8'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# complex long double
#

"""
@cython.boundscheck(False)
cdef np.ndarray[np.npy_clongdouble, ndim=2] ndarray_complex_long_double():
    return np.empty((0,0), dtype='clongdouble')


@cython.boundscheck(False)
cdef np.ndarray[np.npy_clongdouble, ndim=2] ndarray_complex_long_double_C(np.npy_clongdouble *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.npy_clongdouble[:,:] mem_view = <np.npy_clongdouble[:rows,:cols]>data
    dtype = 'clongdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[np.npy_clongdouble, ndim=2] ndarray_complex_long_double_F(np.npy_clongdouble *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.npy_clongdouble[::1,:] mem_view = <np.npy_clongdouble[:rows:1,:cols]>data
    dtype = 'clongdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[np.npy_clongdouble, ndim=2] ndarray_copy_complex_long_double_C(const np.npy_clongdouble *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.npy_clongdouble[:,:] mem_view = <np.npy_clongdouble[:rows,:cols]>data
    dtype = 'clongdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[np.npy_clongdouble, ndim=2] ndarray_copy_complex_long_double_F(const np.npy_clongdouble *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.npy_clongdouble[::1,:] mem_view = <np.npy_clongdouble[:rows:1,:cols]>data
    dtype = 'clongdouble'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))
"""
#
# complex double
#

@cython.boundscheck(False)
cdef np.ndarray[np.complex128_t, ndim=2] ndarray_complex_double():
    return np.empty((0,0), dtype='complex128')

@cython.boundscheck(False)
cdef np.ndarray[np.complex128_t, ndim=2] ndarray_complex_double_C(np.complex128_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex128_t[:,:] mem_view = <np.complex128_t[:rows,:cols]>data
    dtype = 'complex128'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[np.complex128_t, ndim=2] ndarray_complex_double_F(np.complex128_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex128_t[::1,:] mem_view = <np.complex128_t[:rows:1,:cols]>data
    dtype = 'complex128'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[np.complex128_t, ndim=2] ndarray_copy_complex_double_C(const np.complex128_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex128_t[:,:] mem_view = <np.complex128_t[:rows,:cols]>data
    dtype = 'complex128'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[np.complex128_t, ndim=2] ndarray_copy_complex_double_F(const np.complex128_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex128_t[::1,:] mem_view = <np.complex128_t[:rows:1,:cols]>data
    dtype = 'complex128'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))

#
# complex float
#

@cython.boundscheck(False)
cdef np.ndarray[np.complex64_t, ndim=2] ndarray_complex_float():
    return np.empty((0,0), dtype='complex64')

@cython.boundscheck(False)
cdef np.ndarray[np.complex64_t, ndim=2] ndarray_complex_float_C(np.complex64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex64_t[:,:] mem_view = <np.complex64_t[:rows,:cols]>data
    dtype = 'complex64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[np.complex64_t, ndim=2] ndarray_complex_float_F(np.complex64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex64_t[::1,:] mem_view = <np.complex64_t[:rows:1,:cols]>data
    dtype = 'complex64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize])

@cython.boundscheck(False)
cdef np.ndarray[np.complex64_t, ndim=2] ndarray_copy_complex_float_C(const np.complex64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex64_t[:,:] mem_view = <np.complex64_t[:rows,:cols]>data
    dtype = 'complex64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="C"), strides=[row_stride*itemsize, col_stride*itemsize]))

@cython.boundscheck(False)
cdef np.ndarray[np.complex64_t, ndim=2] ndarray_copy_complex_float_F(const np.complex64_t *data, long rows, long cols, long row_stride, long col_stride):
    cdef np.complex64_t[::1,:] mem_view = <np.complex64_t[:rows:1,:cols]>data
    dtype = 'complex64'
    cdef np.npy_intp itemsize = np.dtype(dtype).itemsize
    return np.copy(as_strided(np.asarray(mem_view, dtype=dtype, order="F"), strides=[row_stride*itemsize, col_stride*itemsize]))
