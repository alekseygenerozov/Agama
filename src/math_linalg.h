/** \file   math_linalg.h
    \brief  linear algebra routines (including BLAS)
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include <vector>

namespace math{

/** a simple class for two-dimensional matrices */
template<typename NumT>
class Matrix {
public:
    /// create an empty matrix
    Matrix() : nRows(0), nCols(0) {};

    /// create a matrix of given size
    Matrix(unsigned int _nRows, unsigned int _nCols) :
        nRows(_nRows), nCols(_nCols), data(nRows*nCols) {};

    /// create a matrix of given size and fill it with a value
    Matrix(unsigned int _nRows, unsigned int _nCols, double val) :
        nRows(_nRows), nCols(_nCols), data(nRows*nCols, val) {};

    /// create a matrix of given size from a flattened array of values:
    /// M(row, column) = data[ row*nCols + column ]
    Matrix(unsigned int _nRows, unsigned int _nCols, double* val) :
        nRows(_nRows), nCols(_nCols), data(val, val+nRows*nCols) {};

    /// resize an existing matrix
    void resize(unsigned int newRows, unsigned int newCols) {
        nRows = newRows;
        nCols = newCols;
        data.resize(nRows*nCols);
    }

    /// access the matrix element for reading (no bound checks!)
    const NumT& operator() (unsigned int row, unsigned int column) const {
        return data[row*nCols+column]; }

    /// access the matrix element for writing (no bound checks!)
    NumT& operator() (unsigned int row, unsigned int column) {
        return data[row*nCols+column]; }

    /// number of matrix rows
    unsigned int numRows() const { return nRows; }

    /// number of matrix columns
    unsigned int numCols() const { return nCols; }

    /// access raw data for reading (2d array in row-major order:
    /// indexing scheme is  `M(row, column) = M.data[ row*M.numCols() + column ]` )
    const NumT* getData() const { return &data.front(); }

    /// access raw data for writing (2d array in row-major order)
    NumT* getData() { return &data.front(); }
private:
    unsigned int nRows;      ///< number of rows (first index)
    unsigned int nCols;      ///< number of columns (second index)
    std::vector<NumT> data;  ///< flattened data storage
};

/** zero out array elements with magnitude smaller than the threshold
    times the maximum element of the array;
*/
void eliminateNearZeros(std::vector<double>& vec, double threshold=1e-15);
void eliminateNearZeros(Matrix<double>& mat, double threshold=1e-15);

/** check whether all elements of an array are zeros (return true for an empty array as well) */
bool allZeros(const std::vector<double>& vec);
bool allZeros(const Matrix<double>& mat);

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

/// matrix-vector multiplication:  Y := alpha * A + beta * X
void blas_dgemv(CBLAS_TRANSPOSE TransA,
    double alpha, const Matrix<double>& A, const std::vector<double>& X, double beta,
    std::vector<double>& Y);

/// matrix-vector multiplocation for triangular matrix A:  X := A * X
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

/** perform in-place Cholesky decomposition of a symmetric positive-definite matrix M
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
