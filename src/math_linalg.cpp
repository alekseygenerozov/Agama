#include "math_linalg.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>
#include <gsl/gsl_linalg.h>

#ifdef HAVE_EIGEN

// skip assert-based boundary checks
#define NDEBUG
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#if EIGEN_VERSION_AT_LEAST(3,3,0)
typedef Eigen::BDCSVD<math::Matrix<double>::Type> SVDecompImpl;
#else
//#include <unsupported/Eigen/SVD>
typedef Eigen::JacobiSVD<math::Matrix<double>::Type> SVDecompImpl;
#endif

#else

#include <gsl/gsl_blas.h>
#include <gsl/gsl_version.h>
#if GSL_MAJOR_VERSION >= 2
#define HAVE_GSL_SPARSE
#include <gsl/gsl_spblas.h>
#else
#warning "Sparse matrix support is not available, replaced by dense matrices"
#endif

#endif

namespace math{

namespace {
// wrappers for GSL vector and matrix views (access the data arrays without copying)
struct Vec {
    explicit Vec(std::vector<double>& vec) :
        v(gsl_vector_view_array(&vec.front(), vec.size())) {}
    operator gsl_vector* () { return &v.vector; }
private:
    gsl_vector_view v;
};

struct VecC {
    explicit VecC(const std::vector<double>& vec) :
        v(gsl_vector_const_view_array(&vec.front(), vec.size())) {}
    operator const gsl_vector* () const { return &v.vector; }
private:
    gsl_vector_const_view v;
};

#ifndef HAVE_EIGEN
struct Mat {
    explicit Mat(Matrix<double>& mat) :
        m(gsl_matrix_view_array(mat.data(), mat.rows(), mat.cols())) {}
    operator gsl_matrix* () { return &m.matrix; }
private:
    gsl_matrix_view m;
};

struct MatC {
    explicit MatC(const Matrix<double>& mat) :
        m(gsl_matrix_const_view_array(mat.data(), mat.rows(), mat.cols())) {}
    operator const gsl_matrix* () const { return &m.matrix; }
private:
    gsl_matrix_const_view m;
};
#endif
} // internal namespace

// ------ utility routines with common implementations for both EIGEN and GSL ------ //

void eliminateNearZeros(std::vector<double>& vec, double threshold)
{
    double mag=0;
    for(unsigned int t=0; t<vec.size(); t++)
        mag = fmax(mag, fabs(vec[t]));
    mag *= threshold;
    for(unsigned int t=0; t<vec.size(); t++)
        if(fabs(vec[t]) <= mag)
            vec[t]=0;
}

void eliminateNearZeros(Matrix<double>& mat, double threshold)
{
    double mag=0;
    for(unsigned int i=0; i<mat.rows(); i++)
        for(unsigned int j=0; j<mat.cols(); j++)
            mag = fmax(mag, fabs(mat(i,j)));
    mag *= threshold;
    for(unsigned int i=0; i<mat.rows(); i++)
        for(unsigned int j=0; j<mat.cols(); j++)
            if(fabs(mat(i,j)) <= mag)
                mat(i,j)=0;
}

template<> void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y)
{
    unsigned int size = X.size();
    if(size!=Y.size())
        throw std::invalid_argument("blas_daxpy: invalid size of input arrays");
    if(alpha==0) return;
    for(unsigned int i=0; i<size; i++)
        Y[i] += alpha*X[i];
}

double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y)
{
    unsigned int size = X.size();
    if(size!=Y.size())
        throw std::invalid_argument("blas_ddot: invalid size of input arrays");
    double result = 0;
    for(unsigned int i=0; i<size; i++)
        result += X[i]*Y[i];
    return result;
}

void linearSystemSolveTridiag(const std::vector<double>& diag, const std::vector<double>& aboveDiag,
    const std::vector<double>& belowDiag, const std::vector<double>& y, std::vector<double>& x)
{
    x.resize(diag.size());
    gsl_linalg_solve_tridiag(VecC(diag), VecC(aboveDiag), VecC(belowDiag), VecC(y), Vec(x));
}

void linearSystemSolveTridiagSymm(const std::vector<double>& diag, const std::vector<double>& offDiag,
    const std::vector<double>& y, std::vector<double>& x)
{
    x.resize(diag.size());
    gsl_linalg_solve_symm_tridiag(VecC(diag), VecC(offDiag), VecC(y), Vec(x));
}

#ifdef HAVE_EIGEN
// --------- EIGEN-BASED IMPLEMENTATIONS --------- //

// non-inlined sparse matrix methods

template<typename NumT>
SpMatrix<NumT>::SpMatrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    impl(nRows, nCols)
{
    impl.setFromTriplets(values.begin(), values.end());
}

template<typename NumT>
NumT SpMatrix<NumT>::elem(const unsigned int index, unsigned int &row, unsigned int &col) const
{
    if(static_cast<int>(index) >= impl.nonZeros())
        throw std::range_error("SpMatrix: element index out of range");
    row = impl.innerIndexPtr()[index];
    col = binSearch(static_cast<typename Type::Index>(index), impl.outerIndexPtr(), impl.cols()+1);
    return impl.valuePtr()[index];
}

template<typename NumT>
std::vector<Triplet> SpMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    result.reserve(impl.nonZeros());
    for(int j=0; j<impl.cols(); ++j)
        for(typename Type::InnerIterator i(impl,j); i; ++i)
            result.push_back(Triplet(i.row(), i.col(), i.value()));
    return result;
}

/// convert the result of Eigen operation into a std::vector
template<typename T>
inline std::vector<double> toStdVector(const T& src) {
    Eigen::VectorXd vec(src);
    return std::vector<double>(vec.data(), vec.data()+vec.size());
}

/// wrap std::vector into an Eigen-compatible interface
inline Eigen::Map<const Eigen::VectorXd> toEigenVector(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(&v.front(), v.size());
}

// ------ wrappers for BLAS routines ------ //
template<> void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y)
{
    if(alpha!=0)
        Y.impl += alpha * X.impl;
}

template<typename MatrixType>
void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const MatrixType& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    Eigen::VectorXd v;
    if(TransA==CblasNoTrans)
        v = alpha * A.impl * toEigenVector(X);
    else
        v = alpha * A.impl.transpose() * toEigenVector(X);
    if(beta!=0)
        v += beta * toEigenVector(Y);
    Y.assign(v.data(), v.data()+v.size());
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X)
{
    Eigen::VectorXd v;
    if(Uplo==CblasUpper && Diag==CblasNonUnit) {
        if(TransA == CblasNoTrans)
            v = A.impl.triangularView<Eigen::Upper>() * toEigenVector(X);
        else
            v = A.impl.triangularView<Eigen::Upper>().transpose() * toEigenVector(X);
    } else if(Uplo==CblasUpper && Diag==CblasUnit) {
        if(TransA == CblasNoTrans)
            v = A.impl.triangularView<Eigen::UnitUpper>() * toEigenVector(X);
        else
            v = A.impl.triangularView<Eigen::UnitUpper>().transpose() * toEigenVector(X);
    } else if(Uplo==CblasLower && Diag==CblasNonUnit) {
        if(TransA == CblasNoTrans)
            v = A.impl.triangularView<Eigen::Lower>() * toEigenVector(X);
        else
            v = A.impl.triangularView<Eigen::Lower>().transpose() * toEigenVector(X);
    } else if(Uplo==CblasLower && Diag==CblasUnit) {
        if(TransA == CblasNoTrans)
            v = A.impl.triangularView<Eigen::UnitLower>() * toEigenVector(X);
        else
            v = A.impl.triangularView<Eigen::UnitLower>().transpose() * toEigenVector(X);
    } else
        throw std::invalid_argument("blas_dtrmv: invalid operation mode");
    X.assign(v.data(), v.data()+v.size());
}

template<typename MatrixType>
void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const MatrixType& A, const MatrixType& B, double beta, MatrixType& C)
{
    if(TransA == CblasNoTrans) {
        if(TransB == CblasNoTrans) {
            if(beta==0)
                C.impl = alpha * A.impl * B.impl;
            else
                C.impl = alpha * A.impl * B.impl + beta * C.impl;
        } else {
            if(beta==0)
                C.impl = alpha * A.impl * B.impl.transpose();
            else
                C.impl = alpha * A.impl * B.impl.transpose() + beta * C.impl;
        }
    } else {
        if(TransB == CblasNoTrans) {
            if(beta==0)
                C.impl = alpha * A.impl.transpose() * B.impl;
            else
                C.impl = alpha * A.impl.transpose() * B.impl + beta * C.impl;
        } else {
            if(beta==0)
                C.impl = alpha * A.impl.transpose() * B.impl.transpose();
            else  // in this rare case, and if MatrixType==SpMatrix, we need to convert the storage order
                C.impl = typename MatrixType::Type(alpha * A.impl.transpose() * B.impl.transpose())
                + beta * C.impl;
        }
    }
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B)
{
    if(alpha!=1) {
        B.impl *= alpha;
        blas_dtrsm(Side, Uplo, TransA, Diag, 1, A, B);
        return;
    }
    if(Uplo==CblasUpper && Diag==CblasNonUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheLeft>(B.impl);
            else
                A.impl.triangularView<Eigen::Upper>().transpose().solveInPlace<Eigen::OnTheLeft>(B.impl);
        } else {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(B.impl);
            else
                A.impl.triangularView<Eigen::Upper>().transpose().solveInPlace<Eigen::OnTheRight>(B.impl);
        }
    } else if(Uplo==CblasUpper && Diag==CblasUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::UnitUpper>().solveInPlace<Eigen::OnTheLeft>(B.impl);
            else
                A.impl.triangularView<Eigen::UnitUpper>().transpose().solveInPlace<Eigen::OnTheLeft>(B.impl);
        } else {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::UnitUpper>().solveInPlace<Eigen::OnTheRight>(B.impl);
            else
                A.impl.triangularView<Eigen::UnitUpper>().transpose().solveInPlace<Eigen::OnTheRight>(B.impl);
        }
    } else if(Uplo==CblasLower && Diag==CblasNonUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheLeft>(B.impl);
            else
                A.impl.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheLeft>(B.impl);
        } else {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheRight>(B.impl);
            else
                A.impl.triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(B.impl);
        }
    } else if(Uplo==CblasLower && Diag==CblasUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::UnitLower>().solveInPlace<Eigen::OnTheLeft>(B.impl);
            else
                A.impl.triangularView<Eigen::UnitLower>().transpose().solveInPlace<Eigen::OnTheLeft>(B.impl);
        } else {
            if(TransA == CblasNoTrans)
                A.impl.triangularView<Eigen::UnitLower>().solveInPlace<Eigen::OnTheRight>(B.impl);
            else
                A.impl.triangularView<Eigen::UnitLower>().transpose().solveInPlace<Eigen::OnTheRight>(B.impl);
        }
    } else
        throw std::invalid_argument("blas_dtrsm: invalid operation mode");
}


// ------ linear algebra routines ------ //    

/// LU decomposition for dense matrices
typedef Eigen::PartialPivLU<Matrix<double>::Type> LUDecompImpl;

/// LU decomposition for sparse matrices
typedef Eigen::SparseLU<SpMatrix<double>::Type> SpLUDecompImpl;

LUDecomp::LUDecomp(const Matrix<double>& M) :
    sparse(false), impl(new LUDecompImpl(M.impl)) {}

LUDecomp::LUDecomp(const SpMatrix<double>& M) :
    sparse(true), impl(NULL)
{
    SpLUDecompImpl* LU = new SpLUDecompImpl();
    LU->compute(M.impl);
    if(LU->info() != Eigen::Success) {
        delete LU;
        throw std::runtime_error("Sparse LUDecomp failed");
    }
    impl = LU;
}

LUDecomp::LUDecomp(const LUDecomp& src) :
    sparse(src.sparse), impl(NULL)
{
    if(sparse)  // copy constructor not supported by Eigen
        throw std::runtime_error("Cannot copy Sparse LUDecomp");
    else
        impl = new LUDecompImpl(*static_cast<const LUDecompImpl*>(src.impl));
}

LUDecomp& LUDecomp::operator=(const LUDecomp& src)
{
    if(this == &src)
        return *this;
    void* dest = NULL;
    if(src.sparse)  // assignment not supported by Eigen
        throw std::runtime_error("Cannot assign Sparse LUDecomp");
    else
        dest = new LUDecompImpl(*static_cast<const LUDecompImpl*>(src.impl));
    if(sparse)
        delete static_cast<const SpLUDecompImpl*>(impl);
    else
        delete static_cast<const LUDecompImpl*>(impl);
    sparse = src.sparse;
    impl = dest;
    return *this;
}

LUDecomp::~LUDecomp()
{
    if(sparse)
        delete static_cast<const SpLUDecompImpl*>(impl);
    else
        delete static_cast<const LUDecompImpl*>(impl);
}

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("LUDecomp not initialized");
    if(sparse)
        return toStdVector(static_cast<const SpLUDecompImpl*>(impl)->solve(toEigenVector(rhs)));
    else
        return toStdVector(static_cast<const LUDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}


/// Cholesky decomposition for dense matrices

typedef Eigen::LLT<Matrix<double>::Type, Eigen::Lower> CholeskyDecompImpl;

CholeskyDecomp::CholeskyDecomp(const Matrix<double>& M) :
    impl(new CholeskyDecompImpl(M.impl)) 
{
    if(static_cast<const CholeskyDecompImpl*>(impl)->info() != Eigen::Success)
        throw std::domain_error("CholeskyDecomp failed");
}

CholeskyDecomp::~CholeskyDecomp() { delete static_cast<const CholeskyDecompImpl*>(impl); }

CholeskyDecomp::CholeskyDecomp(const CholeskyDecomp& src) :
    impl(new CholeskyDecompImpl(*static_cast<const CholeskyDecompImpl*>(src.impl))) {}

CholeskyDecomp& CholeskyDecomp::operator=(const CholeskyDecomp& src)
{
    if(this == &src)
        return *this;
    void* dest = new CholeskyDecompImpl(*static_cast<const CholeskyDecompImpl*>(src.impl));
    delete static_cast<const CholeskyDecompImpl*>(impl);
    impl = dest;
    return *this;
}

Matrix<double> CholeskyDecomp::L() const { 
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    Matrix<double> L;
    L.impl = static_cast<const CholeskyDecompImpl*>(impl)->matrixL();
    return L;
}

std::vector<double> CholeskyDecomp::solve(const std::vector<double>& rhs) const {
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    return toStdVector(static_cast<const CholeskyDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}

/// Singular-value decomposition for dense matrices

SVDecomp::SVDecomp(const Matrix<double>& M) :
    impl(new SVDecompImpl(M.impl, Eigen::ComputeThinU | Eigen::ComputeThinV)) {}

SVDecomp::~SVDecomp() { delete static_cast<SVDecompImpl*>(impl); }

SVDecomp::SVDecomp(const SVDecomp& src) :
    impl(new SVDecompImpl(*static_cast<const SVDecompImpl*>(src.impl))) {}

SVDecomp& SVDecomp::operator=(const SVDecomp& src)
{
    if(this == &src)
        return *this;
    void* dest = new SVDecompImpl(*static_cast<const SVDecompImpl*>(src.impl));
    delete static_cast<const SVDecompImpl*>(impl);
    impl = dest;
    return *this;
}

Matrix<double> SVDecomp::U() const { 
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    Matrix<double> U;
    U.impl = static_cast<const SVDecompImpl*>(impl)->matrixU();
    return U;
}

Matrix<double> SVDecomp::V() const { 
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    Matrix<double> V;
    V.impl = static_cast<const SVDecompImpl*>(impl)->matrixV();
    return V;
}

std::vector<double> SVDecomp::S() const { 
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    return toStdVector(static_cast<const SVDecompImpl*>(impl)->singularValues());
}

std::vector<double> SVDecomp::solve(const std::vector<double>& rhs) const {
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    return toStdVector(static_cast<const SVDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}

// --------- END OF EIGEN-BASED IMPLEMENTATIONS --------- //
#else
// --------- GSL-BASED IMPLEMENTATIONS --------- //

// GSL sparse matrices are implemented only in version >= 2, and only with numerical format = double
#ifdef HAVE_GSL_SPARSE

template<typename NumT>
SpMatrix<NumT>::SpMatrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values)
{
    unsigned int size = values.size();
    gsl_spmatrix* sp  = gsl_spmatrix_alloc_nzmax(nRows, nCols, size, GSL_SPMATRIX_TRIPLET);
    for(unsigned int k=0; k<size; k++)
        gsl_spmatrix_set(sp, values[k].i, values[k].j, values[k].v);
    impl = gsl_spmatrix_compcol(sp);
    gsl_spmatrix_free(sp);
}

template<typename NumT>
SpMatrix<NumT>::SpMatrix(const SpMatrix<NumT>& srcObj)
{
    const gsl_spmatrix* src = static_cast<const gsl_spmatrix*>(srcObj.impl);
    gsl_spmatrix* dest = gsl_spmatrix_alloc_nzmax(src->size1, src->size2, src->nz, GSL_SPMATRIX_CCS);
    gsl_spmatrix_memcpy(dest, src);
    impl = dest;
}

template<typename NumT>
SpMatrix<NumT>& SpMatrix<NumT>::operator=(const SpMatrix<NumT>& srcObj)
{
    if(this == &srcObj)
        return *this;
    const gsl_spmatrix* src = static_cast<const gsl_spmatrix*>(srcObj.impl);
    gsl_spmatrix* dest = gsl_spmatrix_alloc_nzmax(src->size1, src->size2, src->nz, GSL_SPMATRIX_CCS);
    gsl_spmatrix_memcpy(dest, src);
    gsl_spmatrix_free(static_cast<gsl_spmatrix*>(impl));
    impl = dest;
    return *this;
}

template<typename NumT>
SpMatrix<NumT>::~SpMatrix() {
    gsl_spmatrix_free(static_cast<gsl_spmatrix*>(impl)); }

template<typename NumT>
NumT SpMatrix<NumT>::operator() (unsigned int row, unsigned int col) const {
    return static_cast<NumT>(gsl_spmatrix_get(static_cast<const gsl_spmatrix*>(impl), row, col)); }

template<typename NumT>
NumT SpMatrix<NumT>::elem(const unsigned int index, unsigned int &row, unsigned int &col) const
{
    const gsl_spmatrix* sp = static_cast<const gsl_spmatrix*>(impl);
    if(index >= sp->nz)
        throw std::range_error("SpMatrix: element index out of range");
    row = sp->i[index];
    col = binSearch(static_cast<size_t>(index), sp->p, sp->size2+1);
    return static_cast<NumT>(sp->data[index]);
}

template<typename NumT>
unsigned int SpMatrix<NumT>::rows() const { return static_cast<const gsl_spmatrix*>(impl)->size1; }

template<typename NumT>
unsigned int SpMatrix<NumT>::cols() const { return static_cast<const gsl_spmatrix*>(impl)->size2; }

template<typename NumT>
unsigned int SpMatrix<NumT>::size() const { return static_cast<const gsl_spmatrix*>(impl)->nz; }

template<typename NumT>
std::vector<Triplet> SpMatrix<NumT>::values() const
{
    const gsl_spmatrix* sp = static_cast<const gsl_spmatrix*>(impl);
    std::vector<Triplet> result;
    result.reserve(sp->nz);
    unsigned int col = 0;
    for(unsigned int k=0; k<sp->nz; k++) {
        while(col<sp->size2 && sp->p[col+1]<=k)
            col++;
        result.push_back(Triplet(sp->i[k], col, sp->data[k]));
    }
    return result;
}

#else
// no GSL support for sparse matrices - implement them as dense matrices
template<typename NumT>
SpMatrix<NumT>::SpMatrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    impl(new Matrix<NumT>(nRows, nCols, values)) {}

template<typename NumT>
SpMatrix<NumT>::SpMatrix(const SpMatrix<NumT>& src) :
    impl(new Matrix<NumT>(*static_cast<const Matrix<NumT>*>(src.impl))) {}

template<typename NumT>
SpMatrix<NumT>& SpMatrix<NumT>::operator=(const SpMatrix<NumT>& src)
{
    if(this == &src)
        return *this;
    Matrix<NumT> *newMat = new Matrix<NumT>(*static_cast<const Matrix<NumT>*>(src.impl));
    delete static_cast<const Matrix<NumT>*>(impl);
    impl = newMat;
    return *this;
}

template<typename NumT>
SpMatrix<NumT>::~SpMatrix() {
    delete static_cast<const Matrix<NumT>*>(impl); }

template<typename NumT>
NumT SpMatrix<NumT>::operator() (unsigned int row, unsigned int col) const {
    return static_cast<const Matrix<NumT>*>(impl)->operator()(row, col); }

template<typename NumT>
NumT SpMatrix<NumT>::elem(const unsigned int index, unsigned int &row, unsigned int &col) const
{
    const Matrix<NumT>* sp = static_cast<const Matrix<NumT>*>(impl);
    row = index / sp->cols();
    col = index % sp->cols();
    if(row >= sp->rows())
        throw std::range_error("SpMatrix: element index out of range");
    return sp->data()[index];
}

template<typename NumT>
unsigned int SpMatrix<NumT>::rows() const { return static_cast<const Matrix<double>*>(impl)->rows(); }

template<typename NumT>
unsigned int SpMatrix<NumT>::cols() const { return static_cast<const Matrix<double>*>(impl)->cols(); }

template<typename NumT>
unsigned int SpMatrix<NumT>::size() const { return rows()*cols(); }

template<typename NumT>
std::vector<Triplet> SpMatrix<NumT>::values() const
{
    const Matrix<NumT>* sp = static_cast<const Matrix<NumT>*>(impl);
    std::vector<Triplet> result;
    unsigned int nCols=sp->cols(), size = nCols * sp->rows();
    const NumT* data = sp->data();
    for(unsigned int k=0; k<size; k++)
        if(data[k]!=0)
            result.push_back(Triplet(k / nCols, k % nCols, data[k]));
    return result;
}
    
#endif

// ------ wrappers for BLAS routines ------ //

template<> void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y)
{
    if(X.rows() != Y.rows() || X.cols() != Y.cols())
        throw std::invalid_argument("blas_daxpy: incompatible sizes of input arrays");
    if(alpha==0) return;
    unsigned int size = X.rows() * X.cols();
    const double* arrX = X.data();
    double* arrY = Y.data();
    for(unsigned int k=0; k<size; k++)
        arrY[k] += alpha*arrX[k];
}

template<> void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const Matrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y) {
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, MatC(A), VecC(X), beta, Vec(Y));
}

template<> void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const SpMatrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y) {
#ifdef HAVE_GSL_SPARSE
    gsl_spblas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, static_cast<const gsl_spmatrix*>(A.impl),
        VecC(X), beta, Vec(Y));
#else
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha,
        MatC(*static_cast<const Matrix<double>*>(A.impl)), VecC(X), beta, Vec(Y));
#endif
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X) {
    gsl_blas_dtrmv((CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, MatC(A), Vec(X));
}

template<> void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta, Matrix<double>& C) {
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, 
        alpha, MatC(A), MatC(B), beta, Mat(C));
}

template<> void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const SpMatrix<double>& A, const SpMatrix<double>& B, double beta, SpMatrix<double>& C) {
#ifdef HAVE_GSL_SPARSE
    if(beta!=0)
        throw std::runtime_error("blas_dgemm: beta!=0 not implemented");
    const gsl_spmatrix* spA = static_cast<const gsl_spmatrix*>(A.impl);
    const gsl_spmatrix* spB = static_cast<const gsl_spmatrix*>(B.impl);
    gsl_spmatrix *trA = NULL, *trB = NULL;
    if(TransA!=CblasNoTrans) {
        trA = gsl_spmatrix_alloc_nzmax(spA->size2, spA->size1, spA->nz, GSL_SPMATRIX_CCS);
        gsl_spmatrix_transpose_memcpy(trA, spA);
        spA = trA;
    }
    if(TransB!=CblasNoTrans) {
        trB = gsl_spmatrix_alloc_nzmax(spB->size2, spB->size1, spB->nz, GSL_SPMATRIX_CCS);
        gsl_spmatrix_transpose_memcpy(trB, spB);
        spB = trB;
    }
    gsl_spblas_dgemm(alpha, spA, spB, static_cast<gsl_spmatrix*>(C.impl));
    if(TransA!=CblasNoTrans)
        gsl_spmatrix_free(trA);
    if(TransB!=CblasNoTrans)
        gsl_spmatrix_free(trB);
#else
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, alpha,
        MatC(*static_cast<const Matrix<double>*>(A.impl)),
        MatC(*static_cast<const Matrix<double>*>(B.impl)),
        beta, Mat(*static_cast< Matrix<double>*>(C.impl)));
#endif
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B) {
    gsl_blas_dtrsm((CBLAS_SIDE_t)Side, (CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, 
        alpha, MatC(A), Mat(B));
}

// ----- Linear algebra routines ----- //

/// LU decomposition implementation for GSL
struct LUDecompImpl {
    gsl_matrix* LU;
    gsl_permutation* perm;
    LUDecompImpl(const Matrix<double>& M) {
        LU = gsl_matrix_alloc(M.rows(), M.cols());
        perm = gsl_permutation_alloc(M.rows());
        if(!LU || !perm) {
            gsl_permutation_free(perm);
            gsl_matrix_free(LU);
            throw std::bad_alloc();
        }
        int dummy;
        gsl_matrix_memcpy(LU, MatC(M));
        gsl_linalg_LU_decomp(LU, perm, &dummy);
    }
    LUDecompImpl(const LUDecompImpl& src) {
        LU = gsl_matrix_alloc(src.LU->size1, src.LU->size2);
        perm = gsl_permutation_alloc(src.LU->size1);
        if(!LU || !perm) {
            gsl_permutation_free(perm);
            gsl_matrix_free(LU);
            throw std::bad_alloc();
        }
        gsl_matrix_memcpy(LU, src.LU);
        gsl_permutation_memcpy(perm, src.perm);
    }
    ~LUDecompImpl() {
        gsl_permutation_free(perm);
        gsl_matrix_free(LU);
    }
private:
    LUDecompImpl& operator=(const LUDecompImpl&);
};

LUDecomp::LUDecomp(const Matrix<double>& M) :
    sparse(false), impl(new LUDecompImpl(M)) {}

// GSL does not offer LU decomposition of sparse matrices, so they are converted to dense ones
LUDecomp::LUDecomp(const SpMatrix<double>& M) :
    sparse(false),
#ifdef HAVE_GSL_SPARSE
    impl(new LUDecompImpl(Matrix<double>(M)))
#else
    impl(new LUDecompImpl(*static_cast<const Matrix<double>*>(M.impl)))
#endif
{}

LUDecomp::~LUDecomp() { delete static_cast<LUDecompImpl*>(impl); }

LUDecomp::LUDecomp(const LUDecomp& src) :
    sparse(false), impl(new LUDecompImpl(*static_cast<const LUDecompImpl*>(src.impl))) {}

LUDecomp& LUDecomp::operator=(const LUDecomp& src)
{
    if(this == &src)
        return *this;
    LUDecompImpl* dest = new LUDecompImpl(*static_cast<const LUDecompImpl*>(src.impl));
    if(!dest)
        throw std::bad_alloc();
    delete static_cast<LUDecompImpl*>(impl);
    impl = dest;
    return *this;
}

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("LUDecomp not initialized");
    if(rhs.size() != static_cast<const LUDecompImpl*>(impl)->LU->size1)
        throw std::invalid_argument("LUDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    gsl_linalg_LU_solve(static_cast<const LUDecompImpl*>(impl)->LU,
        static_cast<const LUDecompImpl*>(impl)->perm, VecC(rhs), Vec(result));
    return result;
}
    
/// Cholesky decomposition implementation for GSL
CholeskyDecomp::CholeskyDecomp(const Matrix<double>& M) :
    impl(NULL)
{
    gsl_matrix* L = gsl_matrix_alloc(M.rows(), M.cols());
    if(!L)
        throw std::bad_alloc();
    gsl_matrix_memcpy(L, MatC(M));
    try{
        gsl_linalg_cholesky_decomp(L);
    }
    catch(std::domain_error&) {
        gsl_matrix_free(L);
        throw std::domain_error("CholeskyDecomp failed");
    }
    impl=L;
}

CholeskyDecomp::~CholeskyDecomp()
{
    gsl_matrix_free(static_cast<gsl_matrix*>(impl));
}

CholeskyDecomp::CholeskyDecomp(const CholeskyDecomp& srcObj) :
    impl(NULL)
{
    const gsl_matrix* src = static_cast<const gsl_matrix*>(srcObj.impl);
    gsl_matrix* L = gsl_matrix_alloc(src->size1, src->size2);
    if(!L)
        throw std::bad_alloc();
    gsl_matrix_memcpy(L, src);
    impl = L;
}

CholeskyDecomp& CholeskyDecomp::operator=(const CholeskyDecomp& srcObj)
{
    if(this == &srcObj)
        return *this;
    const gsl_matrix* src = static_cast<const gsl_matrix*>(srcObj.impl);
    gsl_matrix* L = gsl_matrix_alloc(src->size1, src->size2);
    if(!L)
        throw std::bad_alloc();
    gsl_matrix_free(static_cast<gsl_matrix*>(impl));
    gsl_matrix_memcpy(L, src);
    impl = L;
    return *this;
}

Matrix<double> CholeskyDecomp::L() const
{
    const gsl_matrix* M = static_cast<const gsl_matrix*>(impl);
    if(!M || M->size1!=M->size2)
        throw std::runtime_error("CholeskyDecomp not initialized");
    Matrix<double> L(M->size1, M->size2);
    for(size_t i=0; i<M->size1; i++)
        for(size_t j=0; j<=i; j++)
            L(i,j) = M->data[i*M->size2+j];
    return L;
}

std::vector<double> CholeskyDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    if(rhs.size() != static_cast<const gsl_matrix*>(impl)->size1)
        throw std::invalid_argument("CholeskyDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    gsl_linalg_cholesky_solve(static_cast<const gsl_matrix*>(impl), VecC(rhs), Vec(result));
    return result;
}

/// singular-value decomposition implementation for GSL
struct SVDecompImpl {
    Matrix<double> U, V;
    std::vector<double> S;
};
    
SVDecomp::SVDecomp(const Matrix<double>& M) :
    impl(new SVDecompImpl())
{
    SVDecompImpl* sv = static_cast<SVDecompImpl*>(impl);
    sv->U = M;
    sv->V.resize(M.cols(), M.cols());
    sv->S.resize(M.cols());
    std::vector<double> temp(M.cols());
    if(M.rows() >= M.cols()*5) {   // use a modified algorithm for very 'elongated' matrices
        Matrix<double> tempmat(M.cols(), M.cols());
        gsl_linalg_SV_decomp_mod(Mat(sv->U), Mat(tempmat), Mat(sv->V), Vec(sv->S), Vec(temp));
    } else
        gsl_linalg_SV_decomp(Mat(sv->U), Mat(sv->V), Vec(sv->S), Vec(temp));
    // chop off excessively small singular values which may destabilize solution of linear system
    double minSV = sv->S[0] * 2e-16 * std::max<unsigned int>(sv->S.size(), 10);
    for(unsigned int k=0; k<sv->S.size(); k++)
        if(sv->S[k] < minSV)
            sv->S[k] = 0;
}

SVDecomp::~SVDecomp() { delete static_cast<SVDecompImpl*>(impl); }
    
SVDecomp::SVDecomp(const SVDecomp& src) :
    impl(new SVDecompImpl(*static_cast<const SVDecompImpl*>(src.impl))) {}

SVDecomp& SVDecomp::operator=(const SVDecomp& src)
{
    if(this == &src)
        return *this;
    SVDecompImpl* dest = new SVDecompImpl(*static_cast<const SVDecompImpl*>(src.impl));
    if(!dest)
        throw std::bad_alloc();
    delete static_cast<SVDecompImpl*>(impl);
    impl = dest;
    return *this;
}

Matrix<double> SVDecomp::U() const { return static_cast<const SVDecompImpl*>(impl)->U; }

Matrix<double> SVDecomp::V() const { return static_cast<const SVDecompImpl*>(impl)->V; }

std::vector<double> SVDecomp::S() const { return static_cast<const SVDecompImpl*>(impl)->S; }

std::vector<double> SVDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    const SVDecompImpl* sv = static_cast<const SVDecompImpl*>(impl);
    if(rhs.size() != sv->U.rows())
        throw std::invalid_argument("SVDecomp: incorrect size of RHS vector");
    std::vector<double> result(sv->U.cols());
    gsl_linalg_SV_solve(MatC(sv->U), MatC(sv->V), VecC(sv->S), VecC(rhs), Vec(result));
    return result;
}

#endif

// template instantiations to be compiled (both for Eigen and GSL)
template struct SpMatrix<float>;
template struct SpMatrix<double>;
template void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y);
template void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y);
template void blas_dgemv(CBLAS_TRANSPOSE, double, const Matrix<double>&,
    const std::vector<double>&, double, std::vector<double>&);
template void blas_dgemv(CBLAS_TRANSPOSE, double, const SpMatrix<double>&,
    const std::vector<double>&, double, std::vector<double>&);
template void blas_dgemm(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    double, const Matrix<double>&, const Matrix<double>&, double, Matrix<double>&);
template void blas_dgemm(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    double, const SpMatrix<double>&, const SpMatrix<double>&, double, SpMatrix<double>&);

}  // namespace
