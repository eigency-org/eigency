#ifndef TEST1_H
#define TEST1_H

#include "Eigen/Core"
#include "eigency.h"

long function_w_vec_arg(Eigen::Map<Eigen::VectorXd> &vec);

long function_w_1darr_arg(Eigen::Map<Eigen::ArrayXi> &arr);

void function_w_vec_arg_no_map1(Eigen::VectorXd vec);

void function_w_vec_arg_no_map2(const Eigen::VectorXd &vec);

void function_w_mat_arg(Eigen::Map<Eigen::MatrixXd> &mat);

void function_w_ld_mat_arg(Eigen::Map<eigency::MatrixXld> &mat);

void function_w_complex_mat_arg(Eigen::Map<Eigen::MatrixXcd> &mat);

void function_w_complex_ld_mat_arg(Eigen::Map<eigency::MatrixXcld> &mat);

void function_w_fullspec_arg(Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1> > &vec);

Eigen::VectorXd function_w_vec_retval();

Eigen::Matrix3d function_w_mat_retval();

Eigen::MatrixXd function_w_empty_mat_retval();

Eigen::Matrix<double, 2, 4> function_w_mat_retval_full_spec();

Eigen::Map<Eigen::ArrayXXd> &function_filter1(Eigen::Map<Eigen::ArrayXXd> &);

typedef Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >RowMajorArrayMap;
RowMajorArrayMap &function_filter2(RowMajorArrayMap &);

typedef Eigen::Map<Eigen::ArrayXXd, Eigen::Unaligned, Eigen::Stride<1, Eigen::Dynamic> > CustomStrideMap;
CustomStrideMap &function_filter3(CustomStrideMap &);

Eigen::ArrayXXd function_type_double(Eigen::Map<Eigen::ArrayXXd> &array);
Eigen::ArrayXXf function_type_float(Eigen::Map<Eigen::ArrayXXf> &array);

Eigen::Array<int8_t,  Eigen::Dynamic, Eigen::Dynamic> function_type_int8(Eigen::Map<Eigen::Array<int8_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);
Eigen::Array<int16_t, Eigen::Dynamic, Eigen::Dynamic> function_type_int16(Eigen::Map<Eigen::Array<int16_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);
Eigen::Array<int32_t, Eigen::Dynamic, Eigen::Dynamic> function_type_int32(Eigen::Map<Eigen::Array<int32_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);
Eigen::Array<int64_t, Eigen::Dynamic, Eigen::Dynamic> function_type_int64(Eigen::Map<Eigen::Array<int64_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);

Eigen::Array<uint8_t,  Eigen::Dynamic, Eigen::Dynamic> function_type_uint8(Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);
Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic> function_type_uint16(Eigen::Map<Eigen::Array<uint16_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);
Eigen::Array<uint32_t, Eigen::Dynamic, Eigen::Dynamic> function_type_uint32(Eigen::Map<Eigen::Array<uint32_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);
Eigen::Array<uint64_t, Eigen::Dynamic, Eigen::Dynamic> function_type_uint64(Eigen::Map<Eigen::Array<uint64_t, Eigen::Dynamic, Eigen::Dynamic> > &mat);

Eigen::ArrayXXi function_type_int(Eigen::Map<Eigen::ArrayXXi> &array);
Eigen::ArrayXXcd function_type_complex_double(Eigen::Map<Eigen::ArrayXXcd> &array);
Eigen::ArrayXXcf function_type_complex_float(Eigen::Map<Eigen::ArrayXXcf> &array);

Eigen::Map<Eigen::ArrayXXd> function_single_col_matrix(Eigen::Map<Eigen::ArrayXXd> &array);


class FixedMatrixClass {
public:
    FixedMatrixClass():
        matrix(Eigen::Matrix3d::Constant(4.)) {
    }
    Eigen::Matrix3d &get_matrix() {
        return this->matrix;
    }
    const Eigen::Matrix3d &get_const_matrix() {
        return this->matrix;
    }
private:
    Eigen::Matrix3d matrix;
};

class DynamicArrayClass {
public:
    DynamicArrayClass(const Eigen::Map<Eigen::ArrayXXd> &array):
        array(array) {
    }
    Eigen::Map<Eigen::ArrayXXd> get_array() {
        return this->array;
    }

    Eigen::ArrayXXd get_array_copy() {
        return this->array;
    }
private:
    Eigen::Map<Eigen::ArrayXXd> array;
};

class DynamicRowMajorArrayClass {
public:
    DynamicRowMajorArrayClass(const Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > &array):
        array(array) {
    }
    Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > get_array() {
        return this->array;
    }

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> get_array_copy() {
        return this->array;
    }
private:
    Eigen::Map<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > array;
};

#endif
