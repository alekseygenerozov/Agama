#include "math_linalg.h"
#include <cmath>
#include <stdexcept>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#ifdef HAVE_EIGEN
#include <Eigen/Dense>
#endif

namespace math{

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

namespace {
// wrappers for GSL vector and matrix views (access the data arrays without copying) and permutations
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

} // internal namespace

// ------ wrappers for BLAS routines ------ //

void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y) {
    gsl_blas_daxpy(alpha, VecC(X), Vec(Y));
}

double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y) {
    double result;
    gsl_blas_ddot(VecC(X), VecC(Y), &result);
    return result;
}

double blas_dnrm2(const std::vector<double>& X) {
    return gsl_blas_dnrm2(VecC(X));
}

void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const Matrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y) {
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, MatC(A), VecC(X), beta, Vec(Y));
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X) {
    gsl_blas_dtrmv((CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, MatC(A), Vec(X));
}

void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta, Matrix<double>& C) {
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, 
        alpha, MatC(A), MatC(B), beta, Mat(C));
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B) {
    gsl_blas_dtrsm((CBLAS_SIDE_t)Side, (CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, 
        alpha, MatC(A), Mat(B));
}

// ------ linear algebra routines ------ //
    
#ifdef HAVE_EIGEN
/// convert the result of Eigen operation into a std::vector
template<typename T>
inline std::vector<double> toStdVector(const Eigen::ReturnByValue<T>& src) {
    Eigen::VectorXd vec(src);
    return std::vector<double>(vec.data(), vec.data()+vec.size());
}

/// LU decomposition implementation for Eigen
typedef Eigen::PartialPivLU<Matrix<double>::type> LUDecompImpl;

LUDecomp::LUDecomp(const Matrix<double>& M) :
    impl(new LUDecompImpl(M.impl)) {}

LUDecomp::~LUDecomp() { delete static_cast<const LUDecompImpl*>(impl); }

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const {
    return toStdVector(static_cast<const LUDecompImpl*>(impl)->solve(
        Eigen::Map<const Eigen::VectorXd>(&rhs.front(), rhs.size())));
}

#else

/// LU decomposition implementation for GSL
struct LUDecompImpl {
    Matrix<double> LU;
    gsl_permutation* perm;
    LUDecompImpl(const Matrix<double>& M) : LU(M), perm(gsl_permutation_alloc(M.rows())) {
        int dummy;
        gsl_linalg_LU_decomp(Mat(LU), perm, &dummy);
    }
    ~LUDecompImpl() { gsl_permutation_free(perm); }
};

LUDecomp::LUDecomp(const Matrix<double>& M) :
    impl(new LUDecompImpl(M)) {}

LUDecomp::~LUDecomp() { delete static_cast<const LUDecompImpl*>(impl); }

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const
{
    if(rhs.size() != static_cast<const LUDecompImpl*>(impl)->LU.rows())
        throw std::invalid_argument("LUDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    gsl_linalg_LU_solve(MatC(static_cast<const LUDecompImpl*>(impl)->LU),
        static_cast<const LUDecompImpl*>(impl)->perm, VecC(rhs), Vec(result));
    return result;
}
#endif

void choleskyDecomp(Matrix<double>& A)
{
    gsl_linalg_cholesky_decomp(Mat(A));
}

void linearSystemSolveCholesky(const Matrix<double>& cholA, const std::vector<double>& y, 
    std::vector<double>& x)
{
    x.resize(y.size());
    gsl_linalg_cholesky_solve(MatC(cholA), VecC(y), Vec(x));
}

void singularValueDecomp(Matrix<double>& A, Matrix<double>& V, std::vector<double>& SV)
{
    V.resize(A.cols(), A.cols());
    SV.resize(A.cols());
    std::vector<double> temp(A.cols());
    if(A.rows() >= A.cols()*5) {   // use a modified algorithm for very 'elongated' matrices
        Matrix<double> tempmat(A.cols(), A.cols());
        gsl_linalg_SV_decomp_mod(Mat(A), Mat(tempmat), Mat(V), Vec(SV), Vec(temp));
    } else
        gsl_linalg_SV_decomp(Mat(A), Mat(V), Vec(SV), Vec(temp));
}

void linearSystemSolveSVD(const Matrix<double>& U, const Matrix<double>& V, const std::vector<double>& SV,
    const std::vector<double>& y, std::vector<double>& x)
{
    x.resize(U.cols());
    gsl_linalg_SV_solve(MatC(U), MatC(V), VecC(SV), VecC(y), Vec(x));
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

}  // namespace
