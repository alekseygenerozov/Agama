#include "math_fit.h"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_multimin.h>
#include <stdexcept>

namespace math{

namespace {
// ---- wrappers for GSL vector and matrix views (access the data arrays without copying) ----- //
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
        m(gsl_matrix_view_array(mat.getData(), mat.numRows(), mat.numCols())) {}
    operator gsl_matrix* () { return &m.matrix; }
private:
    gsl_matrix_view m;
};

struct MatC {
    explicit MatC(const Matrix<double>& mat) :
        m(gsl_matrix_const_view_array(mat.getData(), mat.numRows(), mat.numCols())) {}
    operator const gsl_matrix* () const { return &m.matrix; }
private:
    gsl_matrix_const_view m;
};

// ----- wrappers for multidimensional minimization routines ----- //
static double functionWrapperNdim(const gsl_vector* x, void* param) {
    double val;
    static_cast<IFunctionNdim*>(param)->eval(x->data, &val);
    return val;
}

static void functionWrapperNdimDer(const gsl_vector* x, void* param, gsl_vector* df) {
    static_cast<IFunctionNdimDeriv*>(param)->evalDeriv(x->data, NULL, df->data);
}

static void functionWrapperNdimFncDer(const gsl_vector* x, void* param, double* f, gsl_vector* df) {
    static_cast<IFunctionNdimDeriv*>(param)->evalDeriv(x->data, f, df->data);
}

// ----- wrappers for multidimensional nonlinear fitting ----- //
static int functionWrapperNdimMval(const gsl_vector* x, void* param, gsl_vector* f) {
    try{
        static_cast<IFunctionNdimDeriv*>(param)->eval(x->data, f->data);
        return GSL_SUCCESS;
    }
    catch(std::exception&){
        return GSL_FAILURE;
    }
}

static int functionWrapperNdimMvalDer(const gsl_vector* x, void* param, gsl_matrix* df) {
    try{
        static_cast<IFunctionNdimDeriv*>(param)->evalDeriv(x->data, NULL, df->data);
        return GSL_SUCCESS;
    }
    catch(std::exception&){
        return GSL_FAILURE;
    }
}

static int functionWrapperNdimMvalFncDer(const gsl_vector* x, void* param, gsl_vector* f, gsl_matrix* df) {
    try{
        static_cast<IFunctionNdimDeriv*>(param)->evalDeriv(x->data, f->data, df->data);
        return GSL_SUCCESS;
    }
    catch(std::exception&){
        return GSL_FAILURE;
    }
}

}  // internal namespace

// ----- linear least-square fit ------- //
double linearFitZero(const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>* w, double* rms)
{
    if(x.size() != y.size() || (w!=NULL && w->size() != y.size()))
        throw std::invalid_argument("LinearFit: input arrays are not of equal length");
    double c, cov, sumsq;
    if(w==NULL)
        gsl_fit_mul(&x.front(), 1, &y.front(), 1, y.size(), &c, &cov, &sumsq);
    else
        gsl_fit_wmul(&x.front(), 1, &w->front(), 1, &y.front(), 1, y.size(), &c, &cov, &sumsq);
    if(rms!=NULL)
        *rms = sqrt(sumsq/y.size());
    return c;
}

void linearFit(const std::vector<double>& x, const std::vector<double>& y, 
    const std::vector<double>* w, double& slope, double& intercept, double* rms)
{
    if(x.size() != y.size() || (w!=NULL && w->size() != y.size()))
        throw std::invalid_argument("LinearFit: input arrays are not of equal length");
    double cov00, cov11, cov01, sumsq;
    if(w==NULL)
        gsl_fit_linear(&x.front(), 1, &y.front(), 1, y.size(),
            &intercept, &slope, &cov00, &cov01, &cov11, &sumsq);
    else
        gsl_fit_wlinear(&x.front(), 1, &w->front(), 1, &y.front(), 1, y.size(),
            &intercept, &slope, &cov00, &cov01, &cov11, &sumsq);
    if(rms!=NULL)
        *rms = sqrt(sumsq/y.size());
}

// ----- multi-parameter linear least-square fit ----- //
void linearMultiFit(const Matrix<double>& coefs, const std::vector<double>& rhs, 
    const std::vector<double>* w, std::vector<double>& result, double* rms)
{
    if(coefs.numRows() != rhs.size())
        throw std::invalid_argument(
            "LinearMultiFit: number of rows in matrix is different from the length of RHS vector");
    result.assign(coefs.numCols(), 0);
    gsl_matrix* covarMatrix =
        gsl_matrix_alloc(coefs.numCols(), coefs.numCols());
    gsl_multifit_linear_workspace* fitWorkspace =
        gsl_multifit_linear_alloc(coefs.numRows(),coefs.numCols());
    if(covarMatrix==NULL || fitWorkspace==NULL) {
        if(fitWorkspace)
            gsl_multifit_linear_free(fitWorkspace);
        if(covarMatrix)
            gsl_matrix_free(covarMatrix);
        throw std::bad_alloc();
    }
    double sumsq;
    if(w==NULL)
        gsl_multifit_linear(MatC(coefs), VecC(rhs), Vec(result), covarMatrix, &sumsq, fitWorkspace);
    else
        gsl_multifit_wlinear(MatC(coefs), VecC(*w), VecC(rhs), Vec(result),
            covarMatrix, &sumsq, fitWorkspace);
    gsl_multifit_linear_free(fitWorkspace);
    gsl_matrix_free(covarMatrix);
    if(rms!=NULL)
        *rms = sqrt(sumsq/rhs.size());
}

// ----- nonlinear least-square fit ----- //
int nonlinearMultiFit(const IFunctionNdimDeriv& F, const double xinit[],
    const double relToler, const int maxNumIter, double result[])
{
    const unsigned int Nparam = F.numVars();   // number of parameters to vary
    const unsigned int Ndata  = F.numValues(); // number of data points to fit
    if(Ndata < Nparam)
        throw std::invalid_argument(
            "nonlinearMultiFit: number of data points is less than the number of parameters to fit");
    gsl_multifit_function_fdf fnc;
    fnc.p = Nparam;
    fnc.n = Ndata;
    fnc.f = functionWrapperNdimMval;
    fnc.df = functionWrapperNdimMvalDer;
    fnc.fdf = functionWrapperNdimMvalFncDer;
    fnc.params = const_cast<IFunctionNdimDeriv*>(&F);
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Nparam);
    gsl_multifit_fdfsolver* solver = gsl_multifit_fdfsolver_alloc(
        gsl_multifit_fdfsolver_lmsder, Ndata, Nparam);
    int numIter = 0;
    if(gsl_multifit_fdfsolver_set(solver, &fnc, &v_xinit.vector) == GSL_SUCCESS)
    {   // iterate
        do {
            numIter++;
            if(gsl_multifit_fdfsolver_iterate(solver) != GSL_SUCCESS)
                break;
        } while(numIter<maxNumIter && 
            gsl_multifit_test_delta(solver->dx, solver->x, 0, relToler) == GSL_CONTINUE);
    }
    // store the found location of minimum
    for(unsigned int i=0; i<Nparam; i++)
        result[i] = solver->x->data[i];
    gsl_multifit_fdfsolver_free(solver);
    return numIter;
}

// ----- multidimensional minimization ------ //

int findMinNdim(const IFunctionNdim& F, const double xinit[], const double xstep[],
    const double absToler, const int maxNumIter, double result[])
{
    if(F.numValues() != 1)
        throw std::invalid_argument("findMinNdim: function must provide a single output value");
    const unsigned int Ndim = F.numVars();
    // instance of minimizer algorithm
    gsl_multimin_fminimizer* mizer = gsl_multimin_fminimizer_alloc(
        gsl_multimin_fminimizer_nmsimplex2, Ndim);
    gsl_multimin_function fnc;
    fnc.n = Ndim;
    fnc.f = functionWrapperNdim;
    fnc.params = const_cast<IFunctionNdim*>(&F);
    int numIter = 0;
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Ndim);
    gsl_vector_const_view v_xstep = gsl_vector_const_view_array(xstep, Ndim);
    if(gsl_multimin_fminimizer_set(mizer, &fnc, &v_xinit.vector, &v_xstep.vector ) == GSL_SUCCESS)
    {   // iterate
        double sizePrev = gsl_multimin_fminimizer_size(mizer);
        int numIterStall = 0;
        do {
            if(gsl_multimin_fminimizer_iterate(mizer) != GSL_SUCCESS)
                break;
            double sizeCurr = gsl_multimin_fminimizer_size(mizer);
            if(sizeCurr <= absToler)
                break;
            // check if the simplex is stuck
            if(fabs(sizeCurr-sizePrev)/sizePrev <= 1e-4)
                numIterStall++;
            else
                numIterStall = 0;  // reset counter
            if(numIterStall >= 10*(int)Ndim)  // no progress
                break;  // may need to restart it instead?
            sizePrev = sizeCurr;
            numIter++;
        } while(numIter<maxNumIter);
    }
    // store the found location of minimum
    for(unsigned int i=0; i<Ndim; i++)
        result[i] = mizer->x->data[i];
    gsl_multimin_fminimizer_free(mizer);
    return numIter;
}

int findMinNdimDeriv(const IFunctionNdimDeriv& F, const double xinit[], const double xstep,
    const double absToler, const int maxNumIter, double result[])
{
    if(F.numValues() != 1)
        throw std::invalid_argument("findMinNdimDeriv: function must provide a single output value");
    const unsigned int Ndim = F.numVars();
    // instance of minimizer algorithm
    gsl_multimin_fdfminimizer* mizer = gsl_multimin_fdfminimizer_alloc(
        gsl_multimin_fdfminimizer_vector_bfgs2, Ndim);
    gsl_multimin_function_fdf fnc;
    fnc.n = Ndim;
    fnc.f = functionWrapperNdim;
    fnc.df = functionWrapperNdimDer;
    fnc.fdf = functionWrapperNdimFncDer;
    fnc.params = const_cast<IFunctionNdimDeriv*>(&F);
    int numIter = 0;
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Ndim);
    if(gsl_multimin_fdfminimizer_set(mizer, &fnc, &v_xinit.vector, xstep, 0.1) == GSL_SUCCESS)
    {   // iterate
        do {
            numIter++;
            if(gsl_multimin_fdfminimizer_iterate(mizer) != GSL_SUCCESS)
                break;
        } while(numIter<maxNumIter && 
            gsl_multimin_test_gradient(mizer->gradient, absToler) == GSL_CONTINUE);
    }
    // store the found location of minimum
    for(unsigned int i=0; i<Ndim; i++)
        result[i] = mizer->x->data[i];
    gsl_multimin_fdfminimizer_free(mizer);
    return numIter;
}

}  // namespace
