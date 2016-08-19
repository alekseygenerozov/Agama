/** \file   math_linalg.h
    \brief  linear algebra routines
    \date   2015-2016
    \author Eugene Vasiliev

    This module defines the matrix class and the interface for linear algebra routines.
    The actual implementation is provided either by GSL or by Eigen.
    Eigen is a highly-optimized linear algebra library that offers excellent performance
    in run-time at the expense of a significantly longer compilation time
    (since it a header-only library entirely based on esoteric template constructions).
    GSL implementations of the same routines are typically several times slower.
    The switch of back-end is transparent for the rest of the code outside this module,
    however, everything should be recompiled as the definition of Matrix class changes.

    Matrices may be defined over any numerical type, although all linear algebra
    routines operate only on matrices of double, and can be either dense or sparse.
    In dense matrices, data is stored in a flattened 1d array in row-major order,
    i.e., a row is a contiguous block in memory. Strides are not supported.
    Sparse matrices provide only a limited functionality -- construction, read-only element
    access, and a couple of most important operations (matrix-vector and matrix-matrix
    multiplications, and solution of square linear systems by LU decomposition).
    Moreover, GSL prior to version 2 does not have sparse matrix support, thus they are
    implemented as dense matrices, which of course may render some operations entirely
    impractical (the use of GSL is not recommended anyway).
    Again the user interface is agnostic to the actual back-end.
*/
#pragma once
#include <vector>
#ifdef HAVE_EIGEN
// don't use internal OpenMP parallelization at the level of internal Eigen routines
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>
#include <Eigen/SparseCore>
#endif

namespace math{

/// \name Matrix-related classes
///@{

/// a triplet of numbers specifying an element in a sparse matrix
struct Triplet {
    int i;    ///< row
    int j;    ///< column
    double v; ///< value
    Triplet() : i(0), j(0), v(0) {}
    Triplet(const int _i, const int _j, const double _v) : i(_i), j(_j), v(_v) {}
    int row() const { return i; }
    int col() const { return j; }
    double value() const { return v; }
};

/** An abstract read-only interface for a matrix.
    Dense, sparse and diagonal matrices implement this interface and add more specific methods.
    It may also be implemented by a user-defined proxy class in the following context:
    suppose we need to provide the elementwise access to the matrix for some routine,
    but do not want to store the entire matrix in memory, or this matrix is composed of
    several concatenated matrices and we don't want to create a temporary copy of them.
    It can be used to loop over non-zero elements of the original matrix without
    the need for a full two-dimensional loop.
*/
template<typename NumT>
struct IMatrix {
    virtual ~IMatrix() {}

    /// overall size of the matrix (number of possibly nonzero elements)
    virtual unsigned int size() const = 0;

    /// number of matrix rows
    virtual unsigned int numRows() const = 0;

    /// number of matrix columns
    virtual unsigned int numCols() const = 0;

    /// return an element from the matrix at the specified position
    virtual NumT at(const unsigned int row, const unsigned int col) const = 0;

    /// return an element at the overall `index` from the matrix (0 <= index < size),
    /// together with its separate row and column indices; used to loop over all nonzero elements
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const = 0;
};

/// The interface for diagonal matrices
template<typename NumT>
struct DiagonalMatrix: public IMatrix<NumT> {
    DiagonalMatrix() {}
    explicit DiagonalMatrix(const std::vector<NumT>& src): D(src) {};
    virtual unsigned int size()    const { return D.size(); }
    virtual unsigned int numRows() const { return D.size(); }
    virtual unsigned int numCols() const { return D.size(); }
    virtual NumT at(const unsigned int row, const unsigned int col) const {
        return col==row ? D.at(col) : 0;
    }
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        row = col = index;
        return D.at(index);
    }
private:
    const std::vector<NumT> D;  ///< the actual storage of diagonal elements
};

#ifdef HAVE_EIGEN

/// Class for read-only sparse matrices
template<typename NumT>
struct SpMatrix: public IMatrix<NumT> {
    typedef Eigen::SparseMatrix<NumT, Eigen::ColMajor> Type;
    
    /// the actual Eigen sparse matrix object, not to be used directly outside this module
    Type impl;

    /// default constructor for an empty matrix
    SpMatrix() {}

    /// create a matrix of given size from a list of triplets (row,column,value)
    SpMatrix(unsigned int nRows, unsigned int nCols,
        const std::vector<Triplet>& values = std::vector<Triplet>());
    
    /// read-only element access
    NumT operator() (unsigned int row, unsigned int col) const { return impl.coeff(row, col); }
    
    /// return all non-zero elements in a single array of triplets (row, column, value)
    std::vector<Triplet> values() const;
    
    /// get the number of matrix rows
    unsigned int rows() const { return impl.rows(); }
    
    /// get the number of matrix columns
    unsigned int cols() const { return impl.cols(); }
    
    /// get the number of nonzero elements
    virtual unsigned int size() const { return impl.nonZeros(); }

    /// return the given nonzero element and store its row and column indices
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const;

    // other virtual functions of IMatrix interface
    virtual NumT at(unsigned int row, unsigned int col) const { return impl.coeff(row, col); }
    virtual unsigned int numRows() const { return impl.rows(); }
    virtual unsigned int numCols() const { return impl.cols(); }
};

/** Class for two-dimensional matrices that is simply a wrapper around the Matrix class from Eigen.
    We can't partially specialize the Eigen matrix template in pre-C++11 standard,
    so need to define a container class that transparently exposes the basic methods of Eigen::Matrix */
template<typename NumT>
struct Matrix: public IMatrix<NumT> {
    /// a workaround for template typedef that can only be defined inside another class
    typedef Eigen::Matrix<NumT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Type;

    /// the actual Eigen matrix object (should not be used directly outside this module)
    Type impl;

    /// create an empty matrix
    Matrix() {}

    /// create a matrix of given size (values are not initialized!)
    Matrix(unsigned int nRows, unsigned int nCols) : impl(nRows, nCols) {}

    /// create a matrix of given size and initialize it with the given value
    Matrix(unsigned int nRows, unsigned int nCols, const NumT value) :
        impl(nRows, nCols) { impl.fill(value); }

    /// create a dense matrix from a sparse one
    explicit Matrix(const SpMatrix<NumT>& src) : impl(src.impl) {}

    /// create a matrix from a list of triplets (row,column,value)
    Matrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
        impl(nRows, nCols)
    {
        impl.fill(0);
        for(unsigned int k=0; k<values.size(); k++)
            impl(values[k].i, values[k].j) = static_cast<NumT>(values[k].v);
    }

    /// fill the matrix with the given value
    void fill(const NumT value) { impl.fill(value); };

    /// resize an existing matrix while preserving its existing elements
    void conservativeResize(unsigned int newRows, unsigned int newCols) {
        impl.conservativeResize(newRows, newCols); }

    /// resize matrix without preserving its elements
    void resize(unsigned int newRows, unsigned int newCols) { impl.resize(newRows, newCols); }

    /// access the matrix element for reading
    const NumT& operator() (unsigned int row, unsigned int col) const { return impl(row, col); }

    /// access the matrix element for writing
    NumT& operator() (unsigned int row, unsigned int col) { return impl(row, col); }

    /// get the number of matrix rows
    unsigned int rows() const { return impl.rows(); }

    /// get the number of matrix columns
    unsigned int cols() const { return impl.cols(); }

    /// access raw data for reading (2d array in row-major order:
    /// indexing scheme is  `M(row, column) = M.data[ row*M.cols() + column ]` )
    const NumT* data() const { return impl.data(); }

    /// access raw data for writing (2d array in row-major order)
    NumT* data() { return impl.data(); }

    /// get the number of elements
    virtual unsigned int size() const { return impl.rows() * impl.cols(); }
    
    /// return the given nonzero element and store its row and column indices
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        row = index / impl.cols();
        col = index % impl.cols();
        return impl(row, col);
    }
    
    // other virtual functions of IMatrix interface
    virtual NumT at(unsigned int row, unsigned int col) const { return impl(row, col); }
    virtual unsigned int numRows() const { return impl.rows(); }
    virtual unsigned int numCols() const { return impl.cols(); }
};

#else
// no EIGEN

/// class for read-only sparse matrices
template<typename NumT>
struct SpMatrix: public IMatrix<NumT> {
    void* impl;  /// implementation details, not to be used directly outside this module

    SpMatrix() : impl(0) {}
    SpMatrix(const SpMatrix<NumT>&);
    SpMatrix& operator=(const SpMatrix<NumT>&);
    virtual ~SpMatrix();
    
    /// create a matrix of given size from a list of triplets (row,column,value)
    SpMatrix(unsigned int nRows, unsigned int nCols,
        const std::vector<Triplet>& values = std::vector<Triplet>());

    /// read-only access to a matrix element
    NumT operator() (unsigned int row, unsigned int col) const;
    
    /// return all non-zero elements in a single array of triplets (row, column, value)
    std::vector<Triplet> values() const;
    
    /// return the number of matrix rows
    unsigned int rows() const;
    
    /// return the number of matrix columns
    unsigned int cols() const;

    /// return the number of nonzero elements
    virtual unsigned int size() const;

    /// return the given nonzero element and store its row and column indices
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const;
    
    // other virtual functions of IMatrix interface
    virtual NumT at(unsigned int row, unsigned int col) const { return operator()(row, col); }
    virtual unsigned int numRows() const { return rows(); }
    virtual unsigned int numCols() const { return cols(); }
};

/** a simple class for two-dimensional matrices with dense storage */
template<typename NumT>
struct Matrix: public IMatrix<NumT> {
    /// create an empty matrix
    Matrix() : nRows(0), nCols(0) {};

    /// create a matrix of given size, initialized to the given value (0 by default)
    Matrix(unsigned int _nRows, unsigned int _nCols, const NumT value=0) :
        nRows(_nRows), nCols(_nCols), arr(nRows*nCols, value) {};

    /// create a dense matrix from a sparse one
    explicit Matrix(const SpMatrix<NumT>& src) :
        nRows(src.rows()), nCols(src.cols()), arr(nRows*nCols) {
        for(unsigned int k=0; k<src.size(); k++) {
            unsigned int i, j;
            NumT v = src.elem(k, i, j);
            arr[i*nCols+j] = v;
        }
    }

    /// create a matrix from a list of triplets (row,column,value)
    Matrix(unsigned int _nRows, unsigned int _nCols, const std::vector<Triplet>& values) :
        nRows(_nRows), nCols(_nCols), arr(nRows*nCols) {
        for(unsigned int k=0; k<values.size(); k++)
            arr[values[k].i*nCols+values[k].j] = static_cast<NumT>(values[k].v);
    }
    
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

    /// get the number of elements
    virtual unsigned int size() const { return arr.size(); }
    
    /// return the given nonzero element and store its row and column indices
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        row = index / nCols;
        col = index % nCols;
        return arr[index];
    }
    
    // other virtual functions of IMatrix interface
    virtual NumT at(unsigned int row, unsigned int col) const { return arr[row*nCols+col]; }
    virtual unsigned int numRows() const { return nRows; }
    virtual unsigned int numCols() const { return nCols; }
    
private:
    unsigned int nRows;     ///< number of rows (first index)
    unsigned int nCols;     ///< number of columns (second index)
    std::vector<NumT> arr;  ///< flattened data storage
};

#endif

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
/// \name  a subset of BLAS routines
///@{

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

/// dot product of two vectors
double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y);

/// sum of two vectors or two matrices:  Y := alpha * X + Y
/// \tparam Type may be std::vector<double> or Matrix<double>
template<typename Type>
void blas_daxpy(double alpha, const Type& X, Type& Y);

/// multiply vector or matrix by a number:  Y := alpha * Y
/// \tparam Type may be std::vector<double> or Matrix<double>
template<typename Type>
void blas_dmul(double alpha, Type& Y) { blas_daxpy(alpha-1, Y, Y); }

/// matrix-vector multiplication:  Y := alpha * A * X + beta * Y
/// \tparam MatrixType is either Matrix<double> or SpMatrix<double>
template<typename MatrixType>
void blas_dgemv(CBLAS_TRANSPOSE TransA,
    double alpha, const MatrixType& A, const std::vector<double>& X, double beta,
    std::vector<double>& Y);

/// matrix-vector multiplication for a triangular matrix A:  X := A * X
void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X);

/// matrix product:  C := alpha * A * B + beta * C
/// \tparam MatrixType is either Matrix<double> or SpMatrix<double>
template<typename MatrixType>
void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const MatrixType& A, const MatrixType& B, double beta, MatrixType& C);

/// matrix product for a triangular matrix A:
/// B := alpha * A^{-1} * B  (if Side=Left)  or  alpha * B * A^{-1}  (if Side=Right)
void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B);

///@}
/// \name  Linear algebra routines
///@{

/** LU decomposition of a generic square positive-definite matrix M into lower and upper
    triangular matrices:  once created, it may be used to solve a linear system `M x = rhs`
    multiple times with different rhs.
*/
class LUDecomp {
    bool sparse; ///< flag specifying whether this is a dense or sparse matrix
    void* impl;  ///< opaque implementation details
public:
    // default constructor and other boilerplate definitions
    LUDecomp() : sparse(false), impl(0) {}
    ~LUDecomp();
    LUDecomp(const LUDecomp&);
    LUDecomp& operator=(const LUDecomp&);

    /// Construct a decomposition for the given dense matrix M
    explicit LUDecomp(const Matrix<double>& M);

    /// Construct a decomposition for the given sparse matrix M
    explicit LUDecomp(const SpMatrix<double>& M);

    /// Solve the matrix equation `M x = rhs` for x, using the LU decomposition of matrix M
    std::vector<double> solve(const std::vector<double>& rhs) const;
};

    
/** Cholesky decomposition of a symmetric positive-definite matrix M
    into a product of L L^T, where L is a lower triangular matrix.
    Once constructed, it be used for solving a linear system `M x = rhs` multiple times with
    different rhs; the cost of construction is roughty twice lower than LUDecomp.
*/
class CholeskyDecomp {
    void* impl;  ///< opaque implementation details
public:
    CholeskyDecomp() : impl(0) {}
    CholeskyDecomp(const CholeskyDecomp&);
    CholeskyDecomp& operator=(const CholeskyDecomp&);
    ~CholeskyDecomp();
    
    /// Construct a decomposition for the given dense matrix M
    /// (only the lower triangular part of M is used).
    /// \throw std::domain_error if M is not positive-definite
    explicit CholeskyDecomp(const Matrix<double>& M);

    /// return the lower triangular matrix L of the decomposition
    Matrix<double> L() const;

    /// Solve the matrix equation `M x = rhs` for x, using the Cholesky decomposition of matrix M
    std::vector<double> solve(const std::vector<double>& rhs) const;
};

/** Singular value decomposition of an arbitrary M-by-N matrix A (M>=N) into a product  U S V^T,
    where U is an orthogonal M-by-N matrix, S is a diagonal N-by-N matrix of singular values,
    and V is an orthogonal N-by-N matrix.
    It can be used to solve linear systems in the same way as LUDecomp or CholeskyDecomp,
    but is applicable also in the case of non-full-rank systems, both when the number of variables
    is larger than the number of equations (in this case it produces a least-square solution),
    and when the nullspace of the system is nontrivial (in this case the solution with the lowest
    norm out of all possible ones is returned).
    It provides both left and right singular vectors and corresponding singular values;
    for a symmetric positive-definite matrix, SVD is equivalent to eigendecomposition (i.e. U=V).
    SVD is also considerably slower than other decompositions.
*/
class SVDecomp {
    void* impl;  ///< opaque implementation details
public:
    SVDecomp() : impl(0) {}
    SVDecomp(const SVDecomp&);
    SVDecomp& operator=(const SVDecomp&);
    ~SVDecomp();

    /// Construct a decomposition for the given matrix M
    explicit SVDecomp(const Matrix<double>& M);

    /// return the orthogonal matrix U that forms part of the decomposition
    Matrix<double> U() const;

    /// return the orthogonal matrix V that forms part of the decomposition
    Matrix<double> V() const;

    /// return the vector of singular values sorted in decreasing order
    std::vector<double> S() const;

    /// Solve the matrix equation `M x = rhs` for x, using the SVD of matrix M:
    /// if the system is overdetermined, the solution is obtained in the least-square sense;
    /// if the system is underdetermined, the solution with the smallest norm is returned.
    std::vector<double> solve(const std::vector<double>& rhs) const;
};


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
