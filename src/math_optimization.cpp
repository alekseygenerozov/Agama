#ifdef HAVE_CVXOPT
#include <cvxopt.h>  // C interface to LP/QP solver written in Python
#endif
#include "math_optimization.h"
#ifdef HAVE_GLPK
#include <glpk.h>    // GNU linear programming kit - optimization problem solver
#include <setjmp.h>  // to handle errors
#endif
#include <stdexcept>
#include <cassert>
#include "math_core.h"

namespace math{

#ifdef HAVE_GLPK
//------- LP solver from the GNU linear programming kit -------//
namespace{
static jmp_buf err_buf;
/// change the default behaviour on GLPK error to something that doesn't crash program right away
void glpk_error_harmless(void*) {
    longjmp(err_buf, 1);
}
}

template<typename NumT>
std::vector<double> linearOptimizationSolve(const IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,  const std::vector<NumT>& L,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    unsigned int numVariables   = A.cols();
    unsigned int numConstraints = A.rows();
    if( rhs.size()!=numConstraints ||
        (!L.empty()    && L.size()!=numVariables) ||
        (!xmin.empty() && xmin.size()!=numVariables) ||
        (!xmax.empty() && xmax.size()!=numVariables) )
        throw std::invalid_argument("linearOptimizationSolve: invalid size of input arrays");

    // error handling
    glp_error_hook(glpk_error_harmless, NULL);
    if(setjmp(err_buf)) { 
        // this trick saves the current execution state on the first call,
        // and if an error occurs inside GLPK then the modified error handler is invoked,
        // and the control is returned here to the following statement
        glp_free_env();
        throw std::runtime_error("Error in linearOptimizationSolve");
    }

    // set up the problem
    glp_prob* problem = glp_create_prob();
    glp_set_obj_dir(problem, GLP_MIN);
    glp_add_rows(problem, numConstraints);
    glp_add_cols(problem, numVariables);
    for(unsigned int c=0; c<numConstraints; c++)
        glp_set_row_bnds(problem, c+1, GLP_FX, rhs[c], rhs[c]);
    for(unsigned int v=0; v<numVariables; v++) {
        double xminval = xmin.empty()? 0 : xmin[v];
        if(!xmax.empty() && isFinite(xmax[v]))
            glp_set_col_bnds(problem, v+1, GLP_DB, xminval, xmax[v]);
        else
            glp_set_col_bnds(problem, v+1, GLP_LO, xminval, 0 /*ignored*/);
        if(!L.empty() && L[v]!=0)
            glp_set_obj_coef(problem, v+1, L[v]);
    }
    // fill the linear matrix using triplets (row, column, value)
    {   // open a block so that temp variables are destroyed after it is closed.
        unsigned int numNonZero=0, numTotal=A.size();
        // count non-zero values in matrix to reserve the right amount of space
        // in vectors containing sparse matrix coefs
        for(unsigned int i=0; i<numTotal; i++) {
            unsigned int c, v;
            double val = A.elem(i, c, v);
            if(val!=0) numNonZero++;
        }
        std::vector<int> ic;
        std::vector<int> iv;
        std::vector<double> mat;
        ic.reserve (numNonZero+1); ic.push_back(0);
        iv.reserve (numNonZero+1); iv.push_back(0);
        mat.reserve(numNonZero+1); mat.push_back(0);
        for(unsigned int i=0; i<numTotal; i++) {
            unsigned int c, v;
            double val = A.elem(i, c, v);
            if(val!=0) {
                ic.push_back(c+1);
                iv.push_back(v+1);
                mat.push_back(val);
            }
        }
        assert(ic.size()==numNonZero+1);
        glp_load_matrix(problem, numNonZero, &ic.front(), &iv.front(), &mat.front());
    }

    // solve the problem using the interior-point method
    int status = glp_interior(problem, NULL);
    if(glp_ipt_status(problem) != GLP_OPT) status=1;  // infeasible

    // retrieve the solution
    std::vector<double> result(numVariables);
    for(unsigned int v=0; v<numVariables; v++)
        result[v] = glp_ipt_col_prim(problem, v+1);
    glp_delete_prob(problem);

    if(status==0)
        return result;  // success
    throw std::runtime_error(
        status==1?           "linearOptimizationSolve: problem is infeasible" :
        status==GLP_EFAIL?   "linearOptimizationSolve: empty problem":
        status==GLP_ENOCVG?  "linearOptimizationSolve: bad convergence":
        status==GLP_EITLIM?  "linearOptimizationSolve: number of iterations exceeded limit":
        status==GLP_EINSTAB? "linearOptimizationSolve: numerical instability":
        "linearOptimizationSolve: unknown error");
}
#else
// GLPK is not available - use the quadratic solver instead (if possible)
template<typename NumT>
std::vector<double> linearOptimizationSolve(const IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,  const std::vector<NumT>& L,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
#ifdef HAVE_CVXOPT
    return quadraticOptimizationSolve(A, rhs, L, IMatrixDiagonal<NumT>(std::vector<NumT>()), xmin, xmax);
#else
    throw std::runtime_error("linearOptimizationSolve not implemented");
#endif
}
#endif

#ifdef HAVE_CVXOPT

/// helper routine to create CVXOPT-compatible dense or sparse matrix
template<typename NumT>
PyObject* initPyMatrix(const IMatrix<NumT>& M)
{
    unsigned int nrows = M.rows(), ncols = M.cols(), ntotal = M.size();
    if(ntotal == nrows*ncols)  //if(M.isDense())
    {   // dense matrices are initialized item-by-item
        PyObject *pyMatrix = (PyObject*)Matrix_New(nrows, ncols, DOUBLE);
        if(!pyMatrix) return NULL;
        for(unsigned int ir=0; ir<nrows; ir++)
            for(unsigned int ic=0; ic<ncols; ic++)
                MAT_BUFD(pyMatrix)[ic*nrows+ir] = M(ir, ic);  // column-major order used in CVXOPT
        return pyMatrix;
    } else {  // To initialize sparse matrices, additional 3 vectors are allocated
              // which hold col, row, and value triplets.
        unsigned int numNonZero=0;
        for(unsigned int i=0; i<ntotal; i++) {
            unsigned int ir, ic;
            double val = M.elem(i, ir, ic);
            if(val!=0) numNonZero++;
        }
        PyObject *rowind = (PyObject*)Matrix_New(numNonZero, 1, INT);
        PyObject *colind = (PyObject*)Matrix_New(numNonZero, 1, INT);
        PyObject *values = (PyObject*)Matrix_New(numNonZero, 1, DOUBLE);
        if(!colind || !rowind || !values) {
            Py_XDECREF(colind);
            Py_XDECREF(rowind);
            Py_XDECREF(values);
            return NULL;
        }
        for(unsigned int i=0,o=0; i<ntotal; i++) {
            unsigned int ir, ic;
            double val = M.elem(i, ir, ic);
            if(val!=0) {
                MAT_BUFI(rowind)[o]=ir;
                MAT_BUFI(colind)[o]=ic;
                MAT_BUFD(values)[o]=val;
                o++;
            }
        }
        PyObject* pyMatrix = (PyObject*)SpMatrix_NewFromIJV(
            (matrix*)rowind, (matrix*)colind, (matrix*)values, nrows, ncols, DOUBLE);
        Py_DECREF(colind);
        Py_DECREF(rowind);
        Py_DECREF(values);
        return pyMatrix;
    }
}

template<typename NumT>
std::vector<double> quadraticOptimizationSolve(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    unsigned int numVariables   = A.cols();
    unsigned int numConstraints = A.rows();
    bool doQP = Q.size()!=0;
    if( rhs.size()!=numConstraints ||
        (!L.empty()    && L.size()!=numVariables) ||
        (!xmin.empty() && xmin.size()!=numVariables) ||
        (!xmax.empty() && xmax.size()!=numVariables) ||
        (doQP && (Q.rows()!=numVariables || Q.cols()!=numVariables)) )
        throw std::invalid_argument("quadraticOptimizationSolve: invalid size of input arrays");

    Py_Initialize();
    PyObject *solvers = import_cvxopt() < 0 ? NULL : PyImport_ImportModule("cvxopt.solvers");
    if(!solvers) {
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: error importing CVXOPT python module");
    }
    PyObject *problem = PyObject_GetAttrString(solvers, doQP ? "qp" : "lp");
    if(!problem) {
        Py_DECREF(solvers);
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: error initializing CVXOPT python module");
    }

    // count inequality constraints: one (lower) or two (lower+upper) constraints per variable
    unsigned int numConsIneq = numVariables;
    for(unsigned int v=0; v<xmax.size(); v++)
        if(isFinite(xmax[v]))
           numConsIneq++;

    // create matrices
    PyObject *objectiveQuad = doQP ? initPyMatrix(Q) : NULL;
    PyObject *coefMatrix    = initPyMatrix(A);
    PyObject *objectiveLin  = (PyObject*)(Matrix_New(numVariables, 1, DOUBLE));
    PyObject *consIneq      = (PyObject*)(SpMatrix_New(numConsIneq, numVariables, numConsIneq, DOUBLE));
    PyObject *consIneqRhs   = (PyObject*)(Matrix_New(numConsIneq, 1, DOUBLE));
    PyObject *coefRhs       = (PyObject*)(Matrix_New(numConstraints, 1, DOUBLE));
    PyObject *pArgs = PyTuple_New(doQP ? 6 : 5);
    if( !objectiveLin  || (doQP && !objectiveQuad) || !consIneq || !consIneqRhs ||
        !coefMatrix || !coefRhs || !pArgs)
    {
        PyErr_Print();
        Py_DECREF(problem);
        Py_DECREF(solvers);
        Py_XDECREF(objectiveLin); 
        Py_XDECREF(objectiveQuad); 
        Py_XDECREF(consIneq); 
        Py_XDECREF(consIneqRhs); 
        Py_XDECREF(coefMatrix); 
        Py_XDECREF(coefRhs); 
        Py_XDECREF(pArgs);
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: error allocating matrices");
    }

    // init objective func, linear part
    for(size_t v=0; v<numVariables; v++)
        MAT_BUFD(objectiveLin)[v] = L.empty()? 0 : L[v];

    // init inequality constraints
    for(size_t v=0, c=0; v<numVariables; v++, c++) {
        SP_ROW(consIneq) [c] = v;
        SP_VALD(consIneq)[c] = -1.;
        SP_COL(consIneq) [v] = c;
        MAT_BUFD(consIneqRhs)[v]= xmin.empty()? 0 : xmin[v];
        if(!xmax.empty() && isFinite(xmax[v])) { // add upper constraint
            SP_ROW(consIneq) [c+1]= c-v+numVariables;
            SP_VALD(consIneq)[c+1]= 1.;
            MAT_BUFD(consIneqRhs)[c-v+numVariables] = xmax[v];
            c++;
        }
    }
    SP_COL(consIneq)[numVariables] = numConsIneq;

    // init rhs
    for(unsigned int c=0; c<numConstraints; c++)
        MAT_BUFD(coefRhs)[c] = rhs[c];

    // pack matrices into an argument tuple
    if(doQP)
        PyTuple_SetItem(pArgs, 0, objectiveQuad);
    PyTuple_SetItem(pArgs, doQP?1:0, objectiveLin);
    PyTuple_SetItem(pArgs, doQP?2:1, consIneq);
    PyTuple_SetItem(pArgs, doQP?3:2, consIneqRhs);
    PyTuple_SetItem(pArgs, doQP?4:3, coefMatrix);
    PyTuple_SetItem(pArgs, doQP?5:4, coefRhs);
    PyObject *solver = PyObject_CallObject(problem, pArgs);
    if(!solver) {
        PyErr_Print();
        Py_DECREF(pArgs);
        Py_DECREF(problem);
        Py_DECREF(solvers);
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: solver returned error");
    }

    PyObject *status = PyDict_GetItemString(solver, "status");
    bool feasible = std::string(PyString_AsString(status)) == "optimal";
    PyObject *sol = PyDict_GetItemString(solver, "x");

    std::vector<double> result(numVariables);
    for(unsigned int v=0; feasible && v<numVariables; v++)
        result[v] = MAT_BUFD(sol)[v];

    Py_DECREF(solver);
    Py_DECREF(pArgs);
    Py_DECREF(problem);
    Py_DECREF(solvers);
    //Py_Finalize();
    if(feasible)
        return result;
    throw std::runtime_error("quadraticOptimizationSolve: problem is infeasible");
}
#else
template<typename NumT>
std::vector<double> quadraticOptimizationSolve(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
#ifdef HAVE_GLPK
    if(Q.size()==0)  // linear problems will be redirected to the appropriate solver
        return linearOptimizationSolve(A, rhs, L, xmin, xmax);
#else
    throw std::runtime_error("quadraticOptimizationSolve not implemented");
#endif
}
#endif

namespace{
/// Matrix augmented with two extra blocks at the right end, containint a diagonal unit matrix
/// and a diagonal negative unit matrix
template<typename NumT>
class AugmentedMatrix: public IMatrix<NumT> {
    const IMatrix<NumT>& M;       ///< the original matrix
    const unsigned int Mcols, Mrows, Nadd;
public:
    AugmentedMatrix(const IMatrix<NumT>& src, unsigned int _Nadd):
        M(src), Mcols(M.cols()), Mrows(M.rows()), Nadd(_Nadd) {};
    virtual unsigned int size() const { return rows() * cols(); }
    virtual unsigned int rows() const { return Mrows; }
    virtual unsigned int cols() const { return Mcols + 2*Nadd; }
    virtual NumT operator() (const unsigned int row, const unsigned int col) const {
        if(col < Mcols)
            return M(row, col);
        if(col == Mcols + row)
            return +1;
        if(col == Mcols + Nadd + row)
            return -1;
        return 0;
    }
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        row = index / (Mcols + 2*Nadd);
        col = index % (Mcols + 2*Nadd);
        return operator()(row, col);
    }
};

/// Matrix of quadratic penalties with extra diagonal terms for slack variables
template<typename NumT>
class AugmentedQuadMatrix: public IMatrix<NumT> {
    const IMatrix<NumT>& Q;          ///< the original matrix
    const std::vector<double>& pen;  ///< additional elements along the main diagonal
    const unsigned int Qsize, Qrows, Nadd;
public:
    AugmentedQuadMatrix(const IMatrix<NumT>& src, const std::vector<NumT>& _pen):
        Q(src), pen(_pen), Qsize(Q.size()), Qrows(Q.rows()), Nadd(pen.size()) {};
    virtual unsigned int size() const { return Qsize + 2*Nadd; }
    virtual unsigned int rows() const { return Qrows + 2*Nadd; }
    virtual unsigned int cols() const { return Qrows + 2*Nadd; }
    virtual NumT operator() (const unsigned int row, const unsigned int col) const {
        if(col < Qrows)
            return Q(row, col);
        if(col == row)
            return pen.at((col-Qrows) % Nadd);
        return 0;
    }
    virtual NumT elem(const unsigned int index, unsigned int &row, unsigned int &col) const {
        if(index<Qsize)
            return Q.elem(index, row, col);
        row = col = index - Qsize + Qrows;
        return pen.at((index-Qsize) % Nadd);
    }
};

}

template<typename NumT>
std::vector<double> linearOptimizationSolveApprox(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const std::vector<NumT>& consPenaltyLin,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    if(allZeros(consPenaltyLin))
        return linearOptimizationSolve(A, rhs, L, xmin, xmax);
    unsigned int numVariables   = A.cols();
    unsigned int numConstraints = A.rows();
    if( rhs.size()!=numConstraints || consPenaltyLin.size()!=numConstraints ||
        (!L.empty() && L.size()!=numVariables) )
        throw std::invalid_argument("linearOptimizationSolveApprox: invalid size of input arrays");
    /// augment the original vectors with extra 2*numConstraints elements for the slack variables
    std::vector<NumT> Laug(numVariables + 2*numConstraints);
    if(!L.empty())   // copy the array of penalty coefs for variables
        std::copy(L.begin(), L.end(), Laug.begin());
    std::copy(consPenaltyLin.begin(), consPenaltyLin.end(), Laug.begin()+numVariables);
    std::copy(consPenaltyLin.begin(), consPenaltyLin.end(), Laug.begin()+numVariables+numConstraints);
    std::vector<NumT> xminaug(xmin);
    if(!xmin.empty())
        xminaug.insert(xminaug.end(), 2*numConstraints, 0);
    std::vector<NumT> xmaxaug(xmin);
    if(!xmax.empty())
        xmaxaug.insert(xmaxaug.end(), 2*numConstraints, INFINITY);
    std::vector<double> result = linearOptimizationSolve(
        AugmentedMatrix<NumT>(A, numConstraints), rhs, Laug, xminaug, xmaxaug);
    result.resize(numVariables);  // chop off extra slack variables
    return result;
}

template<typename NumT>
std::vector<double> quadraticOptimizationSolveApprox(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& consPenaltyLin, const std::vector<NumT>& consPenaltyQuad,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    if(allZeros(consPenaltyLin) && allZeros(consPenaltyQuad))
        return quadraticOptimizationSolve(A, rhs, L, Q, xmin, xmax);
    unsigned int numVariables   = A.cols();
    unsigned int numConstraints = A.rows();
    if( rhs.size()!=numConstraints ||
        (!consPenaltyLin.empty()  && consPenaltyLin. size()!=numConstraints) ||
        (!consPenaltyQuad.empty() && consPenaltyQuad.size()!=numConstraints) ||
        (!L.empty() && L.size()!=numVariables) )
        throw std::invalid_argument("quadraticOptimizationSolveApprox: invalid size of input arrays");
    /// augment the original vectors with extra 2*numConstraints elements for the slack variables
    std::vector<NumT> Laug(numVariables + 2*numConstraints);
    if(!L.empty())   // copy the array of penalty coefs for variables
        std::copy(L.begin(), L.end(), Laug.begin());
    if(!consPenaltyLin.empty()) {
        std::copy(consPenaltyLin.begin(), consPenaltyLin.end(), Laug.begin()+numVariables);
        std::copy(consPenaltyLin.begin(), consPenaltyLin.end(), Laug.begin()+numVariables+numConstraints);
    }
    std::vector<NumT> xminaug(xmin);
    if(!xmin.empty())
        xminaug.insert(xminaug.end(), 2*numConstraints, 0);
    std::vector<NumT> xmaxaug(xmin);
    if(!xmax.empty())
        xmaxaug.insert(xmaxaug.end(), 2*numConstraints, INFINITY);
    bool Qempty = Q.size()==0;
    std::vector<double> result = quadraticOptimizationSolve(
        AugmentedMatrix<NumT>(A, numConstraints), rhs, Laug,
        Qempty && consPenaltyQuad.empty() ?   // in this case don't create any quadratic matrix
            static_cast<const IMatrix<NumT>&>(IMatrixDiagonal<NumT>(std::vector<NumT>())) :
            static_cast<const IMatrix<NumT>&>(AugmentedQuadMatrix<NumT>(Qempty ?
                static_cast<const IMatrix<NumT>&>(IMatrixDiagonal<NumT>(std::vector<NumT>(numVariables, 0))) :
                Q, consPenaltyQuad.empty() ? std::vector<double>(numConstraints) : consPenaltyQuad)),
        xminaug, xmaxaug);
    result.resize(numVariables);  // chop off extra slack variables
    return result;
}

// explicit instantiations
template
std::vector<double> linearOptimizationSolveApprox(const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);

template
std::vector<double> quadraticOptimizationSolveApprox(const IMatrix<double>&, 
    const std::vector<double>&, const std::vector<double>&, const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);
}
