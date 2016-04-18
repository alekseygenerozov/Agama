/** \file   math_linalg.h
    \brief  linear algebra routines (including BLAS)
    \date   2015-2016
    \author Eugene Vasiliev
*/
#pragma once
#include <vector>
#ifdef HAVE_EIGEN
// don't use internal OpenMP parallelization at the level of internal Eigen routines
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>
#endif

namespace math{

/// \name Matrix class
///@{

#ifdef HAVE_EIGEN

/** class for two-dimensional matrices that is simply a wrapper around the Matrix class from Eigen.
    We can't partially specialize the Eigen matrix template in pre-C++11 standard,
    so need to define a container class that transparently exposes the basic methods of Eigen::Matrix */
template<typename NumT>
struct Matrix {
    /// a workaround for template typedef that can only be defined inside another class
    typedef Eigen::Matrix<NumT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> type;

    /// the actual Eigen matrix object
    type impl;

    /// create an empty matrix
    Matrix() {}

    /// create a matrix of given size (values are not initialized!)
    Matrix(unsigned int nRows, unsigned int nCols) : impl(nRows, nCols) {}

    /// create a matrix of given size and initialize it with the given value
    Matrix(unsigned int nRows, unsigned int nCols, double val) :
        impl(nRows, nCols) { impl.fill(val); }

    /// fill the matrix with the given value
    void fill(const NumT value) { impl.fill(value); };

    /// resize an existing matrix while preserving its existing elements
    void conservativeResize(unsigned int newRows, unsigned int newCols) {
        impl.conservativeResize(newRows, newCols); }

    /// resize matrix without preserving its elements
    void resize(unsigned int newRows, unsigned int newCols) {
        impl.resize(newRows, newCols); }

    /// access the matrix element for reading
    const NumT& operator() (unsigned int row, unsigned int col) const {
        return impl.operator()(row, col); }

    /// access the matrix element for writing
    NumT& operator() (unsigned int row, unsigned int col) {
        return impl.operator()(row, col); }

    /// get the number of matrix rows
    unsigned int rows() const { return impl.rows(); }

    /// get the number of matrix columns
    unsigned int cols() const { return impl.cols(); }

    /// access raw data for reading (2d array in row-major order:
    /// indexing scheme is  `M(row, column) = M.data[ row*M.cols() + column ]` )
    const NumT* data() const { return impl.data(); }

    /// access raw data for writing (2d array in row-major order)
    NumT* data() { return impl.data(); }
};

#else

/** a simple class for two-dimensional matrices with dense storage */
template<typename NumT>
class Matrix {
public:
    /// create an empty matrix
    Matrix() : nRows(0), nCols(0) {};

    /// create a matrix of given size, initialized to the given value (0 by default)
    Matrix(unsigned int _nRows, unsigned int _nCols, double val=0) :
        nRows(_nRows), nCols(_nCols), arr(nRows*nCols, val) {};

    /// create a matrix of given size from a flattened array of values:
    /// M(row, column) = data[ row*nCols + column ]
    /*Matrix(unsigned int _nRows, unsigned int _nCols, double* val) :
        nRows(_nRows), nCols(_nCols), arr(val, val+nRows*nCols) {};*/

    /// fill the matrix with the given value
    void fill(const NumT value) { arr.assign(arr.size(), value); };

    /// resize an existing matrix while preserving its existing elements
    void conservativeResize(unsigned int newRows, unsigned int newCols) {
        nRows = newRows;
        nCols = newCols;
        arr.resize(nRows*nCols);
    }

    /// resize matrix without preserving its elements
    void resize(unsigned int newRows, unsigned int newCols) {
        conservativeResize(newRows, newCols); }

    /// access the matrix element for reading (no bound checks!)
    const NumT& operator() (unsigned int row, unsigned int column) const {
        return arr[row*nCols+column]; }

    /// access the matrix element for writing (no bound checks!)
    NumT& operator() (unsigned int row, unsigned int column) {
        return arr[row*nCols+column]; }

    /// get the number of matrix rows
    unsigned int rows() const { return nRows; }

    /// get the number of matrix columns
    unsigned int cols() const { return nCols; }

    /// access raw data for reading (2d array in row-major order:
    /// indexing scheme is  `M(row, column) = M.data[ row*M.cols() + column ]` )
    const NumT* data() const { return &arr.front(); }

    /// access raw data for writing (2d array in row-major order)
    NumT* data() { return &arr.front(); }
private:
    unsigned int nRows;     ///< number of rows (first index)
    unsigned int nCols;     ///< number of columns (second index)
    std::vector<NumT> arr;  ///< flattened data storage
};
#endif

/** An abstract read-only interface for a matrix (dense or sparse).
    It is used in contexts when only the elementwise access to the matrix is needed
    for some routine, but we do not want to store the entire matrix in memory,
    or this matrix is composed of several concatenated matrices and we don't want
    to create a temporary copy of them.
    May be used to loop over non-zero elements of the original matrix without
    the need for a full two-dimensional loop.
*/
template<typename NumT>
class IMatrix {
public:
    virtual ~IMatrix() {}

    /// overall size of the matrix (number of possibly nonzero elements)
    virtual unsigned int size() const = 0;

    /// number of matrix rows
    virtual unsigned int rows() const = 0;

    /// number of matrix columns
    virtual unsigned int cols() const = 0;

    /// return an element from the matrix at the specified position
    virtual NumT operator() (const unsigned int row, const unsigned int col) const = 0;

    /// return an element at the overall `index` from the matrix (0 <= index < size),
    /// together with its separate row and column indices; used to loop over all nonzero elements
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const = 0;
};

/// The interface for dense matrices
template<typename NumT>
class IMatrixDense: public IMatrix<NumT> {
    const Matrix<NumT>& M;  ///< the actual storage of matrix elements
public:
    explicit IMatrixDense(const Matrix<NumT>& src): M(src) {};
    virtual unsigned int size() const { return M.rows() * M.cols(); }
    virtual unsigned int rows() const { return M.rows(); }
    virtual unsigned int cols() const { return M.cols(); }
    virtual NumT operator() (const unsigned int row, const unsigned int col) const {
        return M(row, col);
    }
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        row = index / M.cols();
        col = index % M.cols();
        return M(row, col);
    }
};

/// The interface for diagonal matrices
template<typename NumT>
class IMatrixDiagonal: public IMatrix<NumT> {
    const std::vector<NumT>& D;  ///< the actual storage of diagonal elements
public:
    explicit IMatrixDiagonal(const std::vector<NumT>& src): D(src) {};
    virtual unsigned int size() const { return D.size(); }
    virtual unsigned int rows() const { return D.size(); }
    virtual unsigned int cols() const { return D.size(); }
    virtual NumT operator() (const unsigned int row, const unsigned int col) const {
        return col==row ? D.at(col) : 0;
    }
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        row = col = index;
        return D.at(index);
    }
};

///@}
/// \name Utility routines
///@{

/** check whether all elements of an array are zeros (return true for an empty array as well) */
template<typename NumT>
bool allZeros(const std::vector<NumT>& vec)
{
    for(unsigned int i=0; i<vec.size(); i++)
        if(vec[i]!=0)
            return false;
    return true;
}

/** check if all elements of a matrix are zeros */
template<typename NumT>
bool allZeros(const Matrix<NumT>& mat)
{
    for(unsigned int i=0; i<mat.rows(); i++)
        for(unsigned int j=0; j<mat.cols(); j++)
            if(mat(i,j) != 0)
                return false;
    return true;
}

/** check if all elements of a matrix accessed through IMatrix interface are zeros */
template<typename NumT>
bool allZeros(const IMatrix<NumT>& mat)
{
    for(unsigned int k=0; k<mat.size(); k++) {
        unsigned int i,j;
        if(mat.elem(k, i, j) != 0)
            return false;
    }
    return true;
}

/** zero out array elements with magnitude smaller than the threshold
    times the maximum element of the array;
*/
void eliminateNearZeros(std::vector<double>& vec, double threshold=1e-15);
void eliminateNearZeros(Matrix<double>& mat, double threshold=1e-15);

///@}
/// \name  BLAS wrappers - same calling conventions as GSL BLAS but with STL vector and our matrix types
///@{

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

/// sum of two vectors:  Y := alpha * X + Y
void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y);

/// dot product of two vectors
double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y);

/// norm of a vector (square root of dot product of a vector by itself)
double blas_dnrm2(const std::vector<double>& X);

/// matrix-vector multiplication:  Y := alpha * A * X + beta * Y
void blas_dgemv(CBLAS_TRANSPOSE TransA,
    double alpha, const Matrix<double>& A, const std::vector<double>& X, double beta,
    std::vector<double>& Y);

/// matrix-vector multiplication for triangular matrix A:  X := A * X
void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X);

/// matrix product:  C := alpha * A * B + beta * C
void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta, Matrix<double>& C);

/// matrix product for triangular matrix A:
/// B := alpha * A^{-1} * B  (if Side=Left)  or  alpha * B * A^{-1}  (if Side=Right)
void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B);

///@}
/// \name  Linear algebra routines
///@{

/** LU decomposition of a generic square matrix M into lower and upper triangular matrices:
    once created, it may be used to solve a linear system `M x = rhs` multiple times
    with different rhs */
class LUDecomp {
    const void* impl;  ///< opaque implementation details
public:
    /// Construct a decomposition for the given matrix M
    LUDecomp(const Matrix<double>& M);
    ~LUDecomp();
    /// Solve the matrix equation `M x = rhs` for x, using the LU decomposition of matrix M
    std::vector<double> solve(const std::vector<double>& rhs) const;
};

/** perform in-place Cholesky decomposition of a symmetric positive-definite matrix A
    into a product of L L^T, where L is a lower triangular matrix.  
    On output, matrix A is replaced by elements of L (stored in the lower triangle)
    and L^T (upper triangle).
*/
void choleskyDecomp(Matrix<double>& A);

/** solve a linear system  A x = y,  using a Cholesky decomposition of matrix A */
void linearSystemSolveCholesky(const Matrix<double>& cholA,
    const std::vector<double>& y, std::vector<double>& x);

/** perform in-place singular value decomposition of a M-by-N matrix A  into a product  U S V^T,
    where U is an orthogonal M-by-N matrix, S is a diagonal N-by-N matrix of singular values,
    and V is an orthogonal N-by-N matrix.
    On output, matrix A is replaced by U, and vector SV contains the elements of diagonal matrix S,
    sorted in decreasing order.
*/
void singularValueDecomp(Matrix<double>& A, Matrix<double>& V, std::vector<double>& SV);

/** solve a linear system  A x = y,  using a singular-value decomposition of matrix A,
    obtained by `singularValueDecomp()`.  The solution is found in the least-square sense.
*/
void linearSystemSolveSVD(const Matrix<double>& U, const Matrix<double>& V, const std::vector<double>& SV,
    const std::vector<double>& y, std::vector<double>& x);

/** solve a tridiagonal linear system  A x = y,  where elements of A are stored in three vectors
    `diag`, `aboveDiag` and `belowDiag` */
void linearSystemSolveTridiag(const std::vector<double>& diag, const std::vector<double>& aboveDiag,
    const std::vector<double>& belowDiag, const std::vector<double>& y, std::vector<double>& x);

/** solve a tridiagonal linear system  A x = y,  where elements of symmetric matrix A are stored 
    in two vectors `diag` and `offDiag` */
void linearSystemSolveTridiagSymm(const std::vector<double>& diag, const std::vector<double>& offDiag,
    const std::vector<double>& y, std::vector<double>& x);

///@}

}  // namespace
