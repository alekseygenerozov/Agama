#include "math_spline.h"
#include "math_core.h"
#include "math_fit.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_version.h>

namespace math {

//-------------- PENALIZED SPLINE APPROXIMATION ---------------//

/// Implementation of penalized spline approximation
class SplineApproxImpl {
private:
    const size_t numDataPoints;        ///< number of x[i],y[i] pairs (original data)
    const size_t numKnots;             ///< number of X[k] knots in the fitting spline; the number of basis functions is numKnots+2
    std::vector<double> knots;         ///< b-spline knots  X[k], k=0..numKnots-1
    std::vector<double> xvalues;       ///< x[i], i=0..numDataPoints-1
    std::vector<double> yvalues;       ///< y[i], overwritten each time loadYvalues is called
    std::vector<double> weightCoefs;   ///< w_p, weight coefficients for basis functions to be found in the process of fitting 
    std::vector<double> zRHS;          ///< z_p = C^T y, right hand side of linear system
    Matrix<double> bsplineMatrix;      ///< matrix "C_ip" used in fitting process; i=0..numDataPoints-1, p=0..numBasisFnc-1
    Matrix<double> LMatrix;            ///< lower triangular matrix L is Cholesky decomposition of matrix A = C^T C, of size numBasisFnc*numBasisFnc
    Matrix<double> MMatrix;            ///< matrix "M" which is the transformed version of roughness matrix "R_pq" of integrals of product of second derivatives of basis functions; p,q=0..numBasisFnc-1
    std::vector<double> singValues;    ///< part of the decomposition of the roughness matrix
    std::vector<double> MTz;           ///< pre-computed M^T z
    gsl_bspline_workspace*
        bsplineWorkspace;              ///< workspace for b-spline evaluation
#if GSL_MAJOR_VERSION < 2
    gsl_bspline_deriv_workspace*
        bsplineDerivWorkspace;         ///< workspace for derivative computation
#endif
    double ynorm2;                     ///< |y|^2 - used to compute residual sum of squares (RSS)

public:
    SplineApproxImpl(const std::vector<double> &_xvalues, const std::vector<double> &_knots);
    ~SplineApproxImpl();

    /** load the y-values of data points and precompute zRHS */
    void loadyvalues(const std::vector<double> &_yvalues);

    /** compute integrals over products of second derivatives of basis functions, 
        and transform R to M+singValues  */
    void initRoughnessMatrix();

    /** compute the weights of basis function for the given value of smoothing parameter */
    void computeWeights(double lambda=0);

    /** compute the RMS scatter of data points about the approximating spline,
        and the number of effective degrees of freedom (EDF) */
    void computeRMSandEDF(double lambda, double* rmserror=NULL, double* edf=NULL) const;

    /** compute the value of Akaike information criterion (AIC)
        for the given value of smoothing parameter 'lambda'  */
    double computeAIC(double lambda);

    /** compute Y-values at spline knots X[k], and also two endpoint derivatives, 
        after the weights w have been determined  */
    void computeYvalues(std::vector<double>& splineValues, double& der_left, double& der_right) const;

    /** compute values of spline at an arbitrary set of points  */
    //void computeRegressionAtPoints(const std::vector<double> &xpoints, std::vector<double> &ypoints) const;

    /** check if the basis matrix L is singular */
    bool isSingular() const { return LMatrix.rows()==0; }

private:
    /** In the unfortunate case that the fit matrix appears to be singular, another algorithm
        is used which is based on the GSL multifit routine, which performs SVD of bsplineMatrix.
        It is much slower and cannot accomodate nonzero smoothing. */
    void computeWeightsSingular();

    SplineApproxImpl& operator= (const SplineApproxImpl&);  ///< assignment operator forbidden
    SplineApproxImpl(const SplineApproxImpl&);              ///< copy constructor forbidden
};

SplineApproxImpl::SplineApproxImpl(const std::vector<double> &_xvalues, const std::vector<double> &_knots) :
    numDataPoints(_xvalues.size()), numKnots(_knots.size()),
    knots(_knots),
    xvalues(_xvalues),
    weightCoefs  (numKnots+2),
    zRHS         (numKnots+2),
    bsplineMatrix(numDataPoints, numKnots+2),
    LMatrix      (numKnots+2, numKnots+2),
    MTz          (numKnots+2)
{
    // first check for validity of input range
    bool range_ok = (numDataPoints>2 && numKnots>2);
    for(size_t k=1; k<numKnots; k++)
        range_ok &= (_knots[k-1]<_knots[k]);  // knots must be in ascending order
    if(!range_ok)
        throw std::invalid_argument("Error in SplineApprox initialization: knots must be in ascending order");
    for(size_t v=0; v<xvalues.size(); v++) {
        if(xvalues[v] < knots.front() || xvalues[v] > knots.back()) 
            throw std::invalid_argument("Error in SplineApprox initialization: "
                "source data points must lie within spline definition region");
    }
    ynorm2 = gsl_nan();  // to indicate that no y-values have been loaded yet

    bsplineWorkspace      = gsl_bspline_alloc(4, numKnots);
#if GSL_MAJOR_VERSION < 2
    bsplineDerivWorkspace = gsl_bspline_deriv_alloc(4);
#endif
    gsl_vector* bsplineValues = gsl_vector_alloc(numKnots+2);
    if(bsplineWorkspace==NULL ||
#if GSL_MAJOR_VERSION < 2
        bsplineDerivWorkspace==NULL ||
#endif
        bsplineValues==NULL)
    {
        gsl_bspline_free(bsplineWorkspace);
#if GSL_MAJOR_VERSION < 2
        gsl_bspline_deriv_free(bsplineDerivWorkspace);
#endif
        gsl_vector_free(bsplineValues);
        throw std::bad_alloc();
    }

    // initialize b-spline matrix C
    gsl_vector_const_view v_knots = gsl_vector_const_view_array(&knots.front(), knots.size());
    gsl_bspline_knots(&v_knots.vector, bsplineWorkspace);
    for(size_t i=0; i<numDataPoints; i++) {
        gsl_bspline_eval(_xvalues[i], bsplineValues, bsplineWorkspace);
        //double wei[4];
        //bsplineWeights<3>(_xvalues[i], &knots.front(), knots.size(), wei);
        for(size_t p=0; p<numKnots+2; p++)
            bsplineMatrix(i, p) = gsl_vector_get(bsplineValues, p);
    }
    gsl_vector_free(bsplineValues);

    // pre-compute matrix L
    blas_dgemm(CblasTrans, CblasNoTrans, 1, bsplineMatrix, bsplineMatrix, 0, LMatrix);
    try {
        choleskyDecomp(LMatrix);
    }
    catch(std::domain_error&) {   // means that the matrix is not positive definite, i.e. fit is singular
        LMatrix = Matrix<double>();
    }
}

SplineApproxImpl::~SplineApproxImpl()
{
    gsl_bspline_free(bsplineWorkspace);
#if GSL_MAJOR_VERSION < 2
    gsl_bspline_deriv_free(bsplineDerivWorkspace);
#endif
}

void SplineApproxImpl::loadyvalues(const std::vector<double> &_yvalues)
{
    if(_yvalues.size() != numDataPoints) 
        throw std::invalid_argument("SplineApprox: input array sizes do not match");
    yvalues = _yvalues;
    ynorm2  = pow_2(blas_dnrm2(yvalues));
    if(!isSingular()) {   // precompute z = C^T y
        blas_dgemv(CblasTrans, 1, bsplineMatrix, yvalues, 0, zRHS);
    }
}

/// convenience function returning values from band matrix or zero if indexes are outside the band
inline double getVal(const Matrix<double>& deriv, size_t row, size_t col)
{
    if(row<col || row>=col+3) return 0;
    else return deriv(row-col, col);
}

void SplineApproxImpl::initRoughnessMatrix()
{
    if(MMatrix.rows()>0) {  // already computed
        blas_dgemv(CblasTrans, 1, MMatrix, zRHS, 0, MTz);  // precompute M^T z
        return;
    }
    // init matrix with roughness penalty (integrals of product of second derivatives of basis functions)
    MMatrix = Matrix<double>(numKnots+2, numKnots+2);   // matrix R_pq
    Matrix<double> derivs(3, numKnots);
    gsl_matrix* bsplineDerivValues = gsl_matrix_alloc(4, 3);
    for(size_t k=0; k<numKnots; k++)
    {
        size_t istart, iend;
#if GSL_MAJOR_VERSION >= 2
        gsl_bspline_deriv_eval_nonzero(knots[k], 2, bsplineDerivValues, 
            &istart, &iend, bsplineWorkspace);
#else
        gsl_bspline_deriv_eval_nonzero(knots[k], 2, bsplineDerivValues, 
            &istart, &iend, bsplineWorkspace, bsplineDerivWorkspace);
#endif
        for(size_t b=0; b<3; b++)
            derivs(b, k) = gsl_matrix_get(bsplineDerivValues, b+k-istart, 2);
    }
    gsl_matrix_free(bsplineDerivValues);
    for(size_t p=0; p<numKnots+2; p++)
    {
        size_t kmin = p>3 ? p-3 : 0;
        size_t kmax = std::min<size_t>(p+3, knots.size()-1);
        for(size_t q=p; q<std::min<size_t>(p+4,numKnots+2); q++)
        {
            double result=0;
            for(size_t k=kmin; k<kmax; k++)
            {
                double x0 = knots[k];
                double x1 = knots[k+1];
                double Gp = getVal(derivs,p,k)*x1 - getVal(derivs,p,k+1)*x0;
                double Hp = getVal(derivs,p,k+1)  - getVal(derivs,p,k);
                double Gq = getVal(derivs,q,k)*x1 - getVal(derivs,q,k+1)*x0;
                double Hq = getVal(derivs,q,k+1)  - getVal(derivs,q,k);
                result += (Hp*Hq*(pow(x1,3.0)-pow(x0,3.0))/3.0 +
                    (Gp*Hq+Gq*Hp)*(pow_2(x1)-pow_2(x0))/2.0 + Gp*Gq*(x1-x0)) / pow_2(x1-x0);
            }
            MMatrix(p, q) = result;
            MMatrix(q, p) = result;  // it is symmetric
        }
    }

    // now transform the roughness matrix R into more suitable form (so far MMatrix contains R)
    // obtain Q = L^{-1} R L^{-T}, where R is the roughness penalty matrix (store Q instead of R)
    blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, LMatrix, MMatrix);
    blas_dtrsm(CblasRight, CblasLower, CblasTrans, CblasNonUnit, 1, LMatrix, MMatrix);   // now MMatrix contains Q = L^{-1} R L^(-T}

    // next decompose this Q via singular value decomposition: Q = U * diag(SV) * V^T
    singValues = std::vector<double>(numKnots+2);   // vector SV
    Matrix<double> tempm(numKnots+2, numKnots+2);
    singularValueDecomp(MMatrix, tempm, singValues);   // now MMatrix contains U, and tempm contains V^T

    // Because Q was symmetric and positive definite, we expect that U=V, but don't actually check it.
    singValues[numKnots] = 0;   // the smallest two singular values must be zero; set explicitly to avoid roundoff error
    singValues[numKnots+1] = 0;

    // precompute M = L^{-T} U  which is used in computing basis weight coefs.
    blas_dtrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, 1, LMatrix, MMatrix);   // now M is finally in place
    // now the weight coefs for any lambda are given by  w = M (I + lambda*diag(singValues))^{-1} M^T  z

    blas_dgemv(CblasTrans, 1, MMatrix, zRHS, 0, MTz);  // precompute M^T z
}

// obtain solution of linear system for the given smoothing parameter, store the weights of basis functions in weightCoefs
void SplineApproxImpl::computeWeights(double lambda)
{
    if(isSingular()) {
        computeWeightsSingular();
        return;
    }
    if(lambda==0)  // simple case, no need to use roughness penalty matrix
        linearSystemSolveCholesky(LMatrix, zRHS, weightCoefs);
    else {
        std::vector<double>tempv(numKnots+2);
        for(size_t p=0; p<numKnots+2; p++) {
            double sv = singValues[p];
            tempv[p] = MTz[p] / (1 + (sv>0 ? sv*lambda : 0));
        }
        blas_dgemv(CblasNoTrans, 1, MMatrix, tempv, 0, weightCoefs);
    }
}

// compute weights of basis functions in the case that the matrix is singular
void SplineApproxImpl::computeWeightsSingular()
{
    double rms;
    linearMultiFit(bsplineMatrix, yvalues, NULL, weightCoefs, &rms);
    ynorm2 = pow_2(rms);
}

void SplineApproxImpl::computeRMSandEDF(double lambda, double* rmserror, double* edf) const
{
    if(rmserror == NULL && edf == NULL)
        return;
    if(isSingular()) {
        if(rmserror)
            *rmserror = ynorm2;
        if(*edf)
            *edf = static_cast<double>(numKnots+2);
        return;
    }
    std::vector<double>tempv(weightCoefs);
    blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, LMatrix, tempv);
    double wTz = blas_ddot(weightCoefs, zRHS);
    if(rmserror)
        *rmserror = (ynorm2 - 2*wTz + pow_2(blas_dnrm2(tempv))) / numDataPoints;
    if(edf == NULL)
        return;
    // equivalent degrees of freedom
    *edf = 0;
    if(!gsl_finite(lambda))
        *edf = 2;
    else if(lambda>0) 
        for(size_t c=0; c<numKnots+2; c++)
            *edf += 1 / (1 + lambda * singValues[c]);
    else
        *edf = static_cast<double>(numKnots+2);
}

double SplineApproxImpl::computeAIC(double lambda) {
    double rmserror, edf;
    computeWeights(lambda);
    computeRMSandEDF(lambda, &rmserror, &edf);
    return log(rmserror) + 2*edf/(numDataPoints-edf-1);
}

/// after the weights of basis functions have been determined, evaluate the values of approximating spline 
/// at its nodes, and additionally its derivatives at endpoints
void SplineApproxImpl::computeYvalues(std::vector<double>& splineValues, double& der_left, double& der_right) const
{
    splineValues.assign(numKnots, 0);
    gsl_vector* bsplineValues = gsl_vector_alloc(numKnots+2);
    gsl_matrix* bsplineDerivValues = gsl_matrix_alloc(numKnots+2, 2);
    for(size_t k=1; k<numKnots-1; k++) {  // loop over interior nodes
        gsl_bspline_eval(knots[k], bsplineValues, bsplineWorkspace);
        double val=0;
        for(size_t p=0; p<numKnots+2; p++)
            val += gsl_vector_get(bsplineValues, p) * weightCoefs[p];
        splineValues[k] = val;
    }
    for(size_t k=0; k<numKnots; k+=numKnots-1) {  // two endpoints: values and derivatives
#if GSL_MAJOR_VERSION >= 2
        gsl_bspline_deriv_eval(knots[k], 1, bsplineDerivValues, bsplineWorkspace);
#else
        gsl_bspline_deriv_eval(knots[k], 1, bsplineDerivValues, bsplineWorkspace, bsplineDerivWorkspace);
#endif
        double val=0, der=0;
        for(size_t p=0; p<numKnots+2; p++) {
            val += gsl_matrix_get(bsplineDerivValues, p, 0) * weightCoefs[p];
            der += gsl_matrix_get(bsplineDerivValues, p, 1) * weightCoefs[p];
        }
        splineValues[k] = val;
        if(k==0)
            der_left = der;
        else
            der_right = der;
    }
    gsl_matrix_free(bsplineDerivValues);
    gsl_vector_free(bsplineValues);
}

//-------- helper class for root-finder -------//
class SplineAICRootFinder: public IFunctionNoDeriv {
public:
    SplineAICRootFinder(SplineApproxImpl& _impl, double _targetAIC) :
        impl(_impl), targetAIC(_targetAIC) {};
    virtual double value(double lambda) const {
        return impl.computeAIC(lambda) - targetAIC;
    }
private:
    SplineApproxImpl& impl;
    double targetAIC;       ///< target value of AIC for root-finder
};

//----------- DRIVER CLASS FOR PENALIZED SPLINE APPROXIMATION ------------//

SplineApprox::SplineApprox(const std::vector<double> &xvalues, const std::vector<double> &knots)
{
    impl = new SplineApproxImpl(xvalues, knots);
}

SplineApprox::~SplineApprox()
{
    delete impl;
}

bool SplineApprox::isSingular() const {
    return impl->isSingular();
}

void SplineApprox::fitData(const std::vector<double> &yvalues, const double lambda,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, double *rmserror, double* edf)
{
    impl->loadyvalues(yvalues);
    if(impl->isSingular() || lambda==0)
        impl->computeWeights();
    else {
        impl->initRoughnessMatrix();
        impl->computeWeights(lambda);
    }
    impl->computeYvalues(splineValues, deriv_left, deriv_right);
    impl->computeRMSandEDF(lambda, rmserror, edf);
}

void SplineApprox::fitDataOversmooth(const std::vector<double> &yvalues, const double deltaAIC,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, 
    double *rmserror, double* edf, double *lambda)
{
    impl->loadyvalues(yvalues);
    double lambdaFit = 0;
    if(impl->isSingular()) {
        impl->computeWeights();
    } else {
        impl->initRoughnessMatrix();
        if(deltaAIC <= 0) {  // find optimal fit
            SplineAICRootFinder fnc(*impl, 0);
            lambdaFit = findMin(fnc, 0, INFINITY, NAN, 1e-6);  // no initial guess
        } else {  // allow for somewhat higher AIC value, to smooth more than minimum necessary amount
            SplineAICRootFinder fnc(*impl, impl->computeAIC(0) + deltaAIC);
            lambdaFit = findRoot(fnc, 0, INFINITY, 1e-6);
            if(!isFinite(lambdaFit))   // root does not exist, i.e. function is everywhere lower than target value
                lambdaFit = INFINITY;  // basically means fitting with a linear regression
        }
    }
    impl->computeYvalues(splineValues, deriv_left, deriv_right);
    impl->computeRMSandEDF(lambdaFit, rmserror, edf);
    if(lambda!=NULL)
        *lambda=lambdaFit;
}

void SplineApprox::fitDataOptimal(const std::vector<double> &yvalues,
    std::vector<double>& splineValues, double& deriv_left, double& deriv_right, 
    double *rmserror, double* edf, double *lambda) 
{
    fitDataOversmooth(yvalues, 0.0, splineValues, deriv_left, deriv_right, rmserror, edf, lambda);
}


//-------------- CUBIC SPLINE --------------//

CubicSpline::CubicSpline(const std::vector<double>& xa,
    const std::vector<double>& ya, double der1, double der2) :
    xval(xa), yval(ya)
{
    unsigned int num_points = xa.size();
    if(ya.size() != num_points)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
    if(num_points < 2)
        throw std::invalid_argument("Error in spline initialization: number of nodes should be >=2");
    unsigned int max_index = num_points - 1;  /* Engeln-Mullges + Uhlig "n" */
    unsigned int sys_size = max_index - 1;    /* linear system is sys_size x sys_size */
    std::vector<double> rhs(sys_size), diag(sys_size), offdiag(sys_size);  // temporary arrays

    for(unsigned int i = 0; i < sys_size; i++) {
        const double h_i   = xa[i + 1] - xa[i];
        const double h_ip1 = xa[i + 2] - xa[i + 1];
        if(h_i<=0 || h_ip1<=0)
            throw std::invalid_argument("Error in spline initialization: x values are not monotonic");
        const double ydiff_i   = ya[i + 1] - ya[i];
        const double ydiff_ip1 = ya[i + 2] - ya[i + 1];
        const double g_i   = 1.0 / h_i;
        const double g_ip1 = 1.0 / h_ip1;
        offdiag[i] = h_ip1;
        diag[i] = 2.0 * (h_ip1 + h_i);
        rhs[i]  = 6.0 * (ydiff_ip1 * g_ip1 -  ydiff_i * g_i);
        if(i == 0 && isFinite(der1)) {
            diag[i] = 1.5 * h_i + 2.0 * h_ip1;
            rhs[i]  = 6.0 * ydiff_ip1 * g_ip1 - 9.0 * ydiff_i * g_i + 3.0 * der1;
        }
        if(i == sys_size-1 && isFinite(der2)) {
            diag[i] = 1.5 * h_ip1 + 2.0 * h_i;
            rhs[i]  = 9.0 * ydiff_ip1 * g_ip1 - 3.0 * der2 - 6.0 * ydiff_i * g_i;
        }
    }

    if(sys_size == 0) {
        cval.assign(2, 0.);
    } else
    if(sys_size == 1) {
        cval.assign(3, 0.);
        cval[1] = rhs[0] / diag[0];
    } else {
        offdiag.resize(sys_size-1);
        linearSystemSolveTridiagSymm(diag, offdiag, rhs, cval);
        cval.insert(cval.begin(), 0.);  // for natural cubic spline,
        cval.push_back(0.);             // 2nd derivatives are zero at endpoints;
    }
    if(isFinite(der1))              // but for a clamped spline they are not.
        cval[0] = ( 3. * (ya[1]-ya[0]) / (xa[1]-xa[0]) 
            - 3. * der1 - 0.5 * cval[1] * (xa[1]-xa[0]) ) / (xa[1]-xa[0]);
    if(isFinite(der2))
        cval[max_index] = ( -3. * (ya[max_index]-ya[max_index-1]) / (xa[max_index]-xa[max_index-1]) 
            + 3. * der2 - 0.5 * cval[max_index-1] * (xa[max_index]-xa[max_index-1]) )
            / (xa[max_index]-xa[max_index-1]);
}

// evaluate spline value, derivative and 2nd derivative at once (faster than doing it separately);
// possibly for several splines (K>=1), k=0,...,K-1);
// input arguments contain the value(s) and 2nd derivative(s) of these splines
// at the boundaries of interval [xl..xh] that contain the point x.
template<unsigned int K>
inline void evalCubicSplines(
    const double xl,   // input:   xl <= x
    const double xh,   // input:   x  <= xh 
    const double x,    // input:   x-value where y is wanted
    const double* yl,  // input:   Y_k(xl)
    const double* yh,  // input:   Y_k(xh)
    const double* cl,  // input:   d2Y_k(xl)
    const double* ch,  // input:   d2Y_k(xh)
    double* y,         // output:  y_k(x)      if y   != NULL
    double* dy,        // output:  dy_k/dx     if dy  != NULL
    double* d2y)       // output:  d^2y_k/d^2x if d2y != NULL
{
    const double h = xh - xl, dx = x - xl;
    for(unsigned int k=0; k<K; k++) {
        double c = cl[k];
        double b = (yh[k] - yl[k]) / h - h * (1./6 * ch[k] + 1./3 * c);
        double d = (ch[k] - c)   / h;
        if(y)
            y[k]   = yl[k] + dx * (b + dx * (1./2 * c + dx * 1./6 * d));
        if(dy)
            dy[k]  = b + dx * (c + dx * 1./2 * d);
        if(d2y)
            d2y[k] = c + dx * d;
    }
}

void CubicSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        double dx  =  xval[1]-xval[0];
        double der = (yval[1]-yval[0]) / dx - dx * (1./6 * cval[1] + 1./3 * cval[0]);
        if(val)
            *val   = yval[0] +
            (der==0 ? 0 : der*(x-xval[0]));  // if der==0, correct result even for infinite x
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        const unsigned int size = xval.size();
        double dx  =  xval[size-1]-xval[size-2];
        double der = (yval[size-1]-yval[size-2]) / dx + dx * (1./6 * cval[size-2] + 1./3 * cval[size-1]);
        if(val)
            *val   = yval[size-1] + (der==0 ? 0 : der*(x-xval[size-1]));
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }

    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalCubicSplines<1> (xval[index], xval[index+1], x, 
        &yval[index], &yval[index+1], &cval[index], &cval[index+1], val, deriv, deriv2);
}

bool CubicSpline::isMonotonic() const
{
    if(xval.size()==0)
        throw std::range_error("Empty spline");
    bool ismonotonic=true;
    for(unsigned int index=0; ismonotonic && index < xval.size()-1; index++) {
        const double
        dx = xval[index + 1] - xval[index],
        dy = yval[index + 1] - yval[index],
        cl = cval[index],
        ch = cval[index+1],
        a  = dx * (ch - cl) * 0.5,
        b  = dx * cl,
        c  = (dy / dx) - dx * (1./6 * ch + 1./3 * cl),
        // derivative is  a * t^2 + b * t + c,  with 0<=t<=1 on the given interval.
        D  = b*b-4*a*c;  // discriminant of the above quadratic equation
        if(D>=0) {       // need to check roots
            double chi1 = (-b-sqrt(D)) / (2*a);
            double chi2 = (-b+sqrt(D)) / (2*a);
            if( (chi1>0 && chi1<1) || (chi2>0 && chi2<1) )
                ismonotonic=false;    // there is a root ( y'=0 ) somewhere on the given interval
        }  // otherwise there are no roots
    }
    return ismonotonic;
}

namespace {
// definite integral of x^(m+n)
class MonomialIntegral: public IFunctionIntegral {
public:
    MonomialIntegral(int _n) : n(_n) {};
    virtual double integrate(double x1, double x2, int m=0) const {
        return m+n+1==0 ? log(x2/x1) : (powInt(x2, m+n+1) - powInt(x1, m+n+1)) / (m+n+1);
    }
private:
    const int n;
};
}

double CubicSpline::integrate(double x1, double x2, int n) const {
    return integrate(x1, x2, MonomialIntegral(n));
}

double CubicSpline::integrate(double x1, double x2, const IFunctionIntegral& f) const
{
    if(x1==x2)
        return 0;
    if(x1>x2)
        return integrate(x2, x1, f);
    double result = 0;
    if(x1<xval.front()) {    // spline is linearly extrapolated at x<xval[0]
        double dx  =  xval[1]-xval[0];
        double der = (yval[1]-yval[0]) / dx - dx * (1./6 * cval[1] + 1./3 * cval[0]);
        double X2  = fmin(x2, xval.front());
        result +=
            f.integrate(x1, X2, 0) * (yval.front() - der * xval.front()) +
            f.integrate(x1, X2, 1) * der;
        if(x2<=xval.front())
            return result;
        x1 = xval.front();
    }
    if(x2>xval.back()) {    // same for x>xval[end]
        unsigned int size = xval.size();
        double dx  =  xval[size-1]-xval[size-2];
        double der = (yval[size-1]-yval[size-2]) / dx + dx * (1./6 * cval[size-2] + 1./3 * cval[size-1]);
        double X1  = fmax(x1, xval.back());
        result +=
            f.integrate(X1, x2, 0) * (yval.back() - der * xval.back()) +
            f.integrate(X1, x2, 1) * der;
        if(x1>=xval.back())
            return result;
        x2 = xval.back();
    }
    unsigned int i1 = binSearch(x1, &xval.front(), xval.size());
    unsigned int i2 = binSearch(x2, &xval.front(), xval.size());
    for(unsigned int i=i1; i<=i2; i++) {
        double x  = xval[i];
        double h  = xval[i+1] - x;
        double a  = yval[i];
        double c  = cval[i];
        double b  = (yval[i+1] - a) / h - h * (1./6 * cval[i+1] + 1./3 * c);
        double d  = (cval[i+1] - c) / h;
        // spline(x) = yval[i] + dx * (b + dx * (c/2 + dx*d/6)), where dx = x-xval[i]
        double X1 = i==i1 ? x1 : x;
        double X2 = i==i2 ? x2 : xval[i+1];
        result   +=
            f.integrate(X1, X2, 0) * (a - x * (b - 1./2 * x * (c - 1./3 * x * d))) +
            f.integrate(X1, X2, 1) * (b - x * (c - 1./2 * x * d)) +
            f.integrate(X1, X2, 2) * (c - x * d) / 2 +
            f.integrate(X1, X2, 3) * d / 6;
    }
    return result;
}

// ------ Hermite cubic spline ------ //

HermiteSpline::HermiteSpline(const std::vector<double>& xv,
    const std::vector<double>& yv, const std::vector<double>& yd) :
    xval(xv), yval(yv), yder(yd)
{
    unsigned int num_points = xv.size();
    if(yv.size() != num_points || yd.size() != num_points)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
    if(num_points < 2)
        throw std::invalid_argument("Error in spline initialization: number of nodes should be >=2");
}

void HermiteSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        if(val)
            *val   = yval.front() +
            (yder.front()==0 ? 0 : yder.front() * (x-xval.front()));  // if der==0, correct result even for infinite x
        if(deriv)
            *deriv = yder.front();
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        if(val)
            *val   = yval.back() + (yder.back()==0 ? 0 : yder.back() * (x-xval.back()));
        if(deriv)
            *deriv = yder.back();
        if(deriv2)
            *deriv2= 0;
        return;
    }
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    const double dx = xval[index+1]-xval[index];
    const double t = (x-xval[index]) / dx;
    if(val)
        *val = pow_2(1-t) * ( (1+2*t) * yval[index]   + t     * yder[index]   * dx )
             + pow_2(t)   * ( (3-2*t) * yval[index+1] + (t-1) * yder[index+1] * dx );
    if(deriv)
        *deriv = 6*t*(1-t) * (yval[index+1]-yval[index]) / dx
               + (1-t)*(1-3*t) * yder[index] + t*(3*t-2) * yder[index+1];
    if(deriv2)
        *deriv2 = ( (6-12*t) * (yval[index+1]-yval[index]) / dx
                + (6*t-4) * yder[index] + (6*t-2) * yder[index+1] ) / dx;
}

// ------ Quintic spline ------- //
QuinticSpline::QuinticSpline(const std::vector<double>& xgrid, 
    const std::vector<double>& ygrid, const std::vector<double>& yderivs): 
    xval(xgrid), yval(ygrid), yder(yderivs)
{
    unsigned int numPoints = xval.size();
    if(yval.size() != numPoints || yder.size() != numPoints)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
    if(numPoints < 2)
        throw std::invalid_argument("Error in spline initialization: number of nodes should be >=2");
    yder3.assign(numPoints, 0);
    std::vector<double> v(numPoints-1);
    double dx = xval[1]-xval[0];
    double dy = yval[1]-yval[0];
    yder3[0]  = v[0] = 0.;
    for(unsigned int i=1; i<numPoints-1; i++) {
        double dx1 = xval[i+1] - xval[i];
        double dx2 = xval[i+1] - xval[i-1];
        double dy1 = yval[i+1] - yval[i];
        double sig = dx/dx2;
        double p   = sig*v[i-1] - 3;
        double der3= 12 * ( 7*yder[i]*dx2 / (dx*dx1) 
            + 3 * (yder[i-1]/dx + yder[i+1]/dx1)
            - 10* (dy / (dx*dx) + dy1 / (dx1*dx1)) ) / dx2;
        yder3[i]   = (der3 - sig*yder3[i-1] ) / p;
        v[i]       = (sig-1)/p;
        dx = dx1;
        dy = dy1;
    }
    yder3[numPoints-1] = 0.;
    for(unsigned int i=numPoints-1; i>0; i--)
        yder3[i-1] += v[i-1]*yder3[i];
}

template<unsigned int K>   // K>=1 - number of splines to compute; k=0,...,K-1
inline void evalQuinticSplines(
    const double xl,   // input:   xl <= x
    const double xh,   // input:   x  <= xh 
    const double x,    // input:   x-value where y is wanted
    const double* yl,  // input:   Y_k(xl)
    const double* yh,  // input:   Y_k(xh)
    const double* y1l, // input:   dY_k(xl)
    const double* y1h, // input:   dY_k(xh)
    const double* y3l, // input:   d3Y_k(xl)
    const double* y3h, // input:   d3Y_k(xh)
    double* y,         // output:  y_k(x)      if y   != NULL
    double* dy,        // output:  dy_k/dx     if dy  != NULL
    double* d2y,       // output:  d^2y_k/d^2x if d2y != NULL
    double* d3y=NULL)  // output:  d^3y_k/d^3x if d3y != NULL
{
    const double h=xh-xl, hi=1./h, hq=h*h, hf=hq/48.;
    const double
        A  = hi*(xh-x),
        Aq = A*A,
        B  = 1-A,
        Bq = B*B,
        C  = h*Aq*B,
        D  =-h*Bq*A,
        E  = hf*C*(2*Aq-A-1),
        F  = hf*D*(2*Bq-B-1);
    double Cp=0, Dp=0, Ep=0, Fp=0, Epp=0, Fpp=0;
    if(dy) {
        Cp = Aq-2*A*B;
        Dp = Bq-2*A*B,
        Ep = 2*A*B*hf * (1+A-5*Aq);
        Fp = 2*A*B*hf * (1+B-5*Bq);
    }
    if(d2y) {
        Epp = hf * (4*Aq*(9*B-A)-2);
        Fpp = hf * (4*Bq*(B-9*A)+2);
    }
    for(unsigned int k=0; k<K; k++) {
        double
            t1 = hi * (yh[k] - yl[k]),
            C2 = y1l[k] - t1,
            C3 = y1h[k] - t1,
            t2 = 6 * (C2 + C3) / hq,
            C4 = y3l[k] - t2,
            C5 = y3h[k] - t2;
        if(y)
            y[k]   = A*yl[k] + B*yh[k] + C*C2 + D*C3+ E*C4 + F*C5;
        if(dy)
            dy[k]  = t1 + Cp*C2 + Dp*C3 + Ep*C4 + Fp*C5;
        if(d2y)
            d2y[k] = ((2*B-4*A)*C2 - (2*A-4*B)*C3 + Epp*C4 + Fpp*C5) * hi;
        if(d3y)
            d3y[k] = A*(2.5*A-1.5)*y3l[k] + B*(2.5*B-1.5)*y3h[k] + 5*A*B*t2;

    }
}

void QuinticSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        if(val)
            *val   = yval[0] + yder[0]*(x-xval[0]);
        if(deriv)
            *deriv = yder[0];
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        const unsigned int size = xval.size();
        if(val)
            *val   = yval[size-1] + yder[size-1]*(x-xval[size-1]);
        if(deriv)
            *deriv = yder[size-1];
        if(deriv2)
            *deriv2= 0;
        return;
    }
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalQuinticSplines<1> (xval[index], xval[index+1], x,
        &yval[index], &yval[index+1], &yder[index], &yder[index+1], &yder3[index], &yder3[index+1],
        val, deriv, deriv2);
}

double QuinticSpline::deriv3(const double x) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front() || x > xval.back())
        return 0;
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    double der3;
    evalQuinticSplines<1> (xval[index], xval[index+1], x,
        &yval[index], &yval[index+1], &yder[index], &yder[index+1], &yder3[index], &yder3[index+1],
        NULL, NULL, NULL, &der3);
    return der3;
}


// ------ INTERPOLATION IN 2D ------ //

BaseInterpolator2d::BaseInterpolator2d(
    const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& zvalues) :
    xval(xgrid), yval(ygrid), zval(zvalues)
{
    const unsigned int xsize = xgrid.size();
    const unsigned int ysize = ygrid.size();
    if(xsize<2 || ysize<2)
        throw std::invalid_argument("Error in 2d interpolator initialization: number of nodes should be >=2 in each direction");
    if(zvalues.rows() != xsize)
        throw std::invalid_argument("Error in 2d interpolator initialization: x and z array lengths differ");
    if(zvalues.cols() != ysize)
        throw std::invalid_argument("Error in 2d interpolator initialization: y and z array lengths differ");
}

// ------- Bilinear interpolation in 2d ------- //

void LinearInterpolator2d::evalDeriv(const double x, const double y, 
     double *z, double *deriv_x, double *deriv_y,
     double* deriv_xx, double* deriv_xy, double* deriv_yy) const
{
    if(isEmpty())
        throw std::range_error("Empty 2d interpolator");
    if(deriv_xx)
        *deriv_xx = 0;
    if(deriv_xy)
        *deriv_xy = 0;
    if(deriv_yy)
        *deriv_yy = 0;
    if(x<xval.front() || x>xval.back() || y<yval.front() || y>yval.back()) {
        if(z)
            *z = NAN;
        if(deriv_x)
            *deriv_x = NAN;
        if(deriv_y)
            *deriv_y = NAN;
        return;
    }
    const unsigned int xi = binSearch(x, &xval.front(), xval.size());
    const unsigned int yi = binSearch(y, &yval.front(), yval.size());
    const double zlowlow = zval(xi, yi);
    const double zlowupp = zval(xi, yi + 1);
    const double zupplow = zval(xi + 1, yi);
    const double zuppupp = zval(xi + 1, yi + 1);
    // Get the width and height of the grid cell
    const double dx = xval[xi+1] - xval[xi];
    const double dy = yval[yi+1] - yval[yi];
    // t and u are the positions within the grid cell at which we are computing
    // the interpolation, in units of grid cell size
    const double t = (x - xval[xi]) / dx;
    const double u = (y - yval[yi]) / dy;
    if(z)
        *z = (1-t)*(1-u)*zlowlow + t*(1-u)*zupplow + (1-t)*u*zlowupp + t*u*zuppupp;
    if(deriv_x)
        *deriv_x = (-(1-u)*zlowlow + (1-u)*zupplow - u*zlowupp + u*zuppupp) / dx;
    if(deriv_y)
        *deriv_y = (-(1-t)*zlowlow - t*zupplow + (1-t)*zlowupp + t*zuppupp) / dy;
}


//------------ 2D CUBIC SPLINE -------------//
// based on interp2d library by David Zaslavsky

CubicSpline2d::CubicSpline2d(const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& zvalues,
    double deriv_xmin, double deriv_xmax, double deriv_ymin, double deriv_ymax) :
    BaseInterpolator2d(xgrid, ygrid, zvalues)
{
    const unsigned int xsize = xval.size();
    const unsigned int ysize = yval.size();
    zx.resize (xsize, ysize);
    zy.resize (xsize, ysize);
    zxy.resize(xsize, ysize);
    std::vector<double> tmpvalues(ysize);
    for(unsigned int i=0; i<xsize; i++) {
        for(unsigned int j=0; j<ysize; j++)
            tmpvalues[j] = zval(i, j);
        CubicSpline spl(yval, tmpvalues, deriv_ymin, deriv_ymax);
        for(unsigned int j=0; j<ysize; j++)
            spl.evalDeriv(yval[j], NULL, &zy(i, j));
    }
    tmpvalues.resize(xsize);
    for(unsigned int j=0; j<ysize; j++) {
        for(unsigned int i=0; i<xsize; i++)
            tmpvalues[i] = zval(i, j);
        CubicSpline spl(xval, tmpvalues, deriv_xmin, deriv_xmax);
        for(unsigned int i=0; i<xsize; i++)
            spl.evalDeriv(xval[i], NULL, &zx(i, j));
    }
    for(unsigned int j=0; j<ysize; j++) {
        // if derivs at the boundary are specified, 2nd deriv must be zero
        if( (j==0 && isFinite(deriv_ymin)) || (j==ysize-1 && isFinite(deriv_ymax)) ) {
            for(unsigned int i=0; i<xsize; i++)
                zxy(i, j) = 0.;
        } else {
            for(unsigned int i=0; i<xsize; i++)
                tmpvalues[i] = zy(i, j);
            CubicSpline spl(xval, tmpvalues,
                isFinite(deriv_xmin) ? 0. : NAN, isFinite(deriv_xmax) ? 0. : NAN);
            for(unsigned int i=0; i<xsize; i++)
                spl.evalDeriv(xval[i], NULL, &zxy(i, j));
        }
    }
}

void CubicSpline2d::evalDeriv(const double x, const double y, 
    double *z, double *z_x, double *z_y, double *z_xx, double *z_xy, double *z_yy) const
{
    if(isEmpty())
        throw std::range_error("Empty 2d spline");
    if(x<xval.front() || x>xval.back() || y<yval.front() || y>yval.back()) {
        if(z)
            *z = NAN;
        if(z_x)
            *z_x = NAN;
        if(z_y)
            *z_y = NAN;
        if(z_xx)
            *z_xx = NAN;
        if(z_xy)
            *z_xy = NAN;
        if(z_yy)
            *z_yy = NAN;
        return;
    }
    // Get the indices of grid cell in both dimensions
    const unsigned int xi = binSearch(x, &xval.front(), xval.size());
    const unsigned int yi = binSearch(y, &yval.front(), yval.size());
    const double
        // Get the values on the corners of the grid cell
        xlow = xval[xi],
        xupp = xval[xi+1],
        ylow = yval[yi],
        yupp = yval[yi+1],
        zlowlow = zval(xi,   yi),
        zlowupp = zval(xi,   yi+1),
        zupplow = zval(xi+1, yi),
        zuppupp = zval(xi+1, yi+1),
        // Get the width and height of the grid cell
        dx = xupp - xlow,
        dy = yupp - ylow,
        // t and u are the positions within the grid cell at which we are computing
        // the interpolation, in units of grid cell size
        t = (x - xlow) / dx,
        u = (y - ylow) / dy,
        dt = 1./dx, // partial t / partial x
        du = 1./dy, // partial u / partial y
        dtdu = dt*du,
        zxlowlow  = zx (xi  , yi  ) / dt,
        zxlowupp  = zx (xi  , yi+1) / dt,
        zxupplow  = zx (xi+1, yi  ) / dt,
        zxuppupp  = zx (xi+1, yi+1) / dt,
        zylowlow  = zy (xi  , yi  ) / du,
        zylowupp  = zy (xi  , yi+1) / du,
        zyupplow  = zy (xi+1, yi  ) / du,
        zyuppupp  = zy (xi+1, yi+1) / du,
        zxylowlow = zxy(xi  , yi  ) / dtdu,
        zxylowupp = zxy(xi  , yi+1) / dtdu,
        zxyupplow = zxy(xi+1, yi  ) / dtdu,
        zxyuppupp = zxy(xi+1, yi+1) / dtdu,
        t0 = 1,
        t1 = t,
        t2 = t*t,
        t3 = t*t2,
        u0 = 1,
        u1 = u,
        u2 = u*u,
        u3 = u*u2,
        t0u0 = t0*u0, t0u1=t0*u1, t0u2=t0*u2, t0u3=t0*u3,
        t1u0 = t1*u0, t1u1=t1*u1, t1u2=t1*u2, t1u3=t1*u3,
        t2u0 = t2*u0, t2u1=t2*u1, t2u2=t2*u2, t2u3=t2*u3,
        t3u0 = t3*u0, t3u1=t3*u1, t3u2=t3*u2, t3u3=t3*u3,
        sum0 = zlowlow - zupplow - zlowupp + zuppupp,
        sum1 = 2*sum0 +   zxlowlow + zxupplow -   zxlowupp - zxuppupp,
        sum2 = 3*sum0 + 2*zxlowlow + zxupplow - 2*zxlowupp - zxuppupp,
        sum3 = zylowlow   -   zyupplow + zylowupp - zyuppupp,
        sum4 = 2*zylowlow - 2*zyupplow + zylowupp - zyuppupp;
    double
        zvalue= 0,
        zderx = 0,
        zdery = 0,
        zd_xx = 0,
        zd_xy = 0,
        zd_yy = 0;
    double v = zlowlow;
    zvalue += v*t0u0;
    v = zylowlow;
    zvalue += v*t0u1;
    zdery  += v*t0u0;
    v = -3*zlowlow + 3*zlowupp - 2*zylowlow - zylowupp;
    zvalue += v*t0u2;
    zdery  += 2*v*t0u1;
    zd_yy  += 2*v*t0u0;
    v = 2*zlowlow - 2*zlowupp + zylowlow + zylowupp;
    zvalue += v*t0u3;
    zdery  += 3*v*t0u2;
    zd_yy  += 6*v*t0u1;
    v = zxlowlow;
    zvalue += v*t1u0;
    zderx  += v*t0u0;
    v = zxylowlow;
    zvalue += v*t1u1;
    zderx  += v*t0u1;
    zdery  += v*t1u0;
    zd_xy  += v*t0u0;
    v = -3*zxlowlow + 3*zxlowupp - 2*zxylowlow - zxylowupp;
    zvalue += v*t1u2;
    zderx  += v*t0u2;
    zdery  += 2*v*t1u1;
    zd_xy  += 2*v*t0u1;
    zd_yy  += 2*v*t1u0;
    v = 2*zxlowlow - 2*zxlowupp + zxylowlow + zxylowupp;
    zvalue += v*t1u3;
    zderx  += v*t0u3;
    zdery  += 3*v*t1u2;
    zd_xy  += 3*v*t0u2;
    zd_yy  += 6*v*t1u1;
    v = -3*zlowlow + 3*zupplow - 2*zxlowlow - zxupplow;
    zvalue += v*t2u0;
    zderx  += 2*v*t1u0;
    zd_xx  += 2*v*t0u0;
    v = -3*zylowlow + 3*zyupplow - 2*zxylowlow - zxyupplow;
    zvalue += v*t2u1;
    zderx  += 2*v*t1u1;
    zdery  += v*t2u0;
    zd_xx  += 2*v*t0u1;
    zd_xy  += 2*v*t1u0;
    v = 4*zxylowlow + 2*zxyupplow + 2*zxylowupp + zxyuppupp + 3*sum2 + 3*sum4;
    zvalue += v*t2u2;
    zderx  += 2*v*t1u2;
    zdery  += 2*v*t2u1;
    zd_xx  += 2*v*t0u2;
    zd_xy  += 4*v*t1u1;
    zd_yy  += 2*v*t2u0;
    v =  -2*zxylowlow - zxyupplow - 2*zxylowupp - zxyuppupp - 2*sum2 - 3*sum3;
    zvalue += v*t2u3;
    zderx  += 2*v*t1u3;
    zdery  += 3*v*t2u2;
    zd_xx  += 2*v*t0u3;
    zd_xy  += 6*v*t1u2;
    zd_yy  += 6*v*t2u1;
    v = 2*zlowlow - 2*zupplow + zxlowlow + zxupplow;
    zvalue += v*t3u0;
    zderx  += 3*v*t2u0;
    zd_xx  += 6*v*t1u0;
    v = 2*zylowlow - 2*zyupplow + zxylowlow + zxyupplow;
    zvalue += v*t3u1;
    zderx  += 3*v*t2u1;
    zdery  += v*t3u0;
    zd_xx  += 6*v*t1u1;
    zd_xy  += 3*v*t2u0;
    v = -2*zxylowlow - 2*zxyupplow - zxylowupp - zxyuppupp - 3*sum1 - 2*sum4;
    zvalue += v*t3u2;
    zderx  += 3*v*t2u2;
    zdery  += 2*v*t3u1;
    zd_xx  += 6*v*t1u2;
    zd_xy  += 6*v*t2u1;
    zd_yy  += 2*v*t3u0;
    v = zxylowlow + zxyupplow + zxylowupp + zxyuppupp + 2*sum1 + 2*sum3;
    zvalue += v*t3u3;
    zderx  += 3*v*t2u3;
    zdery  += 3*v*t3u2;
    zd_xx  += 6*v*t1u3;
    zd_xy  += 9*v*t2u2;
    zd_yy  += 6*v*t3u1;

    if(z   !=NULL) *z    = zvalue;
    if(z_x !=NULL) *z_x  = zderx*dt;
    if(z_y !=NULL) *z_y  = zdery*du;
    if(z_xx!=NULL) *z_xx = zd_xx*dt*dt;
    if(z_xy!=NULL) *z_xy = zd_xy*dt*du;
    if(z_yy!=NULL) *z_yy = zd_yy*du*du;
}


//------------ 2D QUINTIC SPLINE -------------//

QuinticSpline2d::QuinticSpline2d(const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& zvalues, const Matrix<double>& dzdx, const Matrix<double>& dzdy) :
    BaseInterpolator2d(xgrid, ygrid, zvalues), zx(dzdx), zy(dzdy)
{
    const unsigned int xsize = xval.size();
    const unsigned int ysize = yval.size();
    // 1. for each y do 1d quintic spline for z in x, and record d^3z/dx^3
    zxxx.resize(xsize, ysize);
    zyyy.resize(xsize, ysize);
    zxyy.resize(xsize, ysize);
    zxxxyy.resize(xsize, ysize);
    std::vector<double> t(xsize), t1(xsize);
    for(unsigned int j=0; j<ysize; j++) {
        for(unsigned int i=0; i<xsize; i++) {
            t[i]  = zval(i, j);
            t1[i] = dzdx(i, j);
        }
        QuinticSpline s(xval, t, t1);
        for(unsigned int i=0; i<xsize; i++)
            zxxx(i, j) = s.deriv3(xval[i]);
    }
    // 2. for each x do 1d quintic spline for z and splines for dz/dx, d^3z/dx^3 in y
    t.resize(ysize);
    t1.resize(ysize);
    for(unsigned int i=0; i<xsize; i++) {
        for(unsigned int j=0; j<ysize; j++) {
            t[j]  = zval(i, j);
            t1[j] = dzdy(i, j);
        }
        QuinticSpline s(yval, t, t1);
        for(unsigned int j=0; j<ysize; j++)
            t1[j] = dzdx(i, j);
        CubicSpline u(yval, t1, 0, 0);
        for(unsigned int j=0; j<ysize; j++)
            t1[j] = zxxx(i, j);
        CubicSpline v(yval, t1);
        for(unsigned int j=0; j<ysize; j++) {
            zyyy(i, j) = s.deriv3(yval[j]);
            u.evalDeriv(yval[j], NULL, NULL, &zxyy(i, j));
            v.evalDeriv(yval[j], NULL, NULL, &zxxxyy(i, j));
        }
    }
}

void QuinticSpline2d::evalDeriv(const double x, const double y,
    double* z, double* z_x, double* z_y,
    double* z_xx, double* z_xy, double* z_yy) const
{
    if(isEmpty())
        throw std::range_error("Empty 2d spline");
    if(x<xval.front() || x>xval.back() || y<yval.front() || y>yval.back()) {
        if(z)
            *z = NAN;
        if(z_x)
            *z_x = NAN;
        if(z_y)
            *z_y = NAN;
        if(z_xx)
            *z_xx = NAN;
        if(z_xy)
            *z_xy = NAN;
        if(z_yy)
            *z_yy = NAN;
        return;
    }
    // get the indices of grid cell
    const unsigned int xl = binSearch(x, &xval.front(), xval.size()), xu = xl+1;
    const unsigned int yl = binSearch(y, &yval.front(), yval.size()), yu = yl+1;

    // obtain values of various derivatives at the boundaries of intervals for intermediate splines
    const double
        fl[2]  = { zval(xl, yl), zval(xu, yl) },
        fh[2]  = { zval(xl, yu), zval(xu, yu) },
        f1l[2] = { zy  (xl, yl), zy  (xu, yl) },
        f1h[2] = { zy  (xl, yu), zy  (xu, yu) },
        f3l[2] = { zyyy(xl, yl), zyyy(xu, yl) },
        f3h[2] = { zyyy(xl, yu), zyyy(xu, yu) },
        flo[4] = { zx  (xl, yl), zx  (xu, yl), zxxx(xl, yl),   zxxx(xu, yl) },
        fhi[4] = { zx  (xl, yu), zx  (xu, yu), zxxx(xl, yu),   zxxx(xu, yu) },
        f2l[4] = { zxyy(xl, yl), zxyy(xu, yl), zxxxyy(xl, yl), zxxxyy(xu, yl) },
        f2h[4] = { zxyy(xl, yu), zxyy(xu, yu), zxxxyy(xl, yu), zxxxyy(xu, yu) };
    bool der  = z_y!=NULL || z_xy!=NULL;
    bool der2 = z_yy!=NULL;
    // compute intermediate splines
    double F[2], G[4], dF[2], dG[4], d2F[2], d2G[4];
    evalQuinticSplines<2> (yval[yl], yval[yu], y, 
        fl, fh, f1l, f1h, f3l, f3h,  F, der? dF : NULL, der2? d2F : NULL);
    evalCubicSplines<4>   (yval[yl], yval[yu], y, 
        flo, fhi, f2l, f2h,  G, der? dG : NULL, der2? d2G : NULL);
    // compute and output requested values and derivatives
    evalQuinticSplines<1> (xval[xl], xval[xu], x,
        &F[0], &F[1], &G[0], &G[1], &G[2], &G[3],  z, z_x, z_xx);
    if(z_y || z_xy)
        evalQuinticSplines<1> (xval[xl], xval[xu], x,
            &dF[0], &dF[1], &dG[0], &dG[1], &dG[2], &dG[3],  z_y, z_xy, NULL);
    if(z_yy)
        evalQuinticSplines<1> (xval[xl], xval[xu], x,
            &d2F[0], &d2F[1], &d2G[0], &d2G[1], &d2G[2], &d2G[3],  z_yy, NULL, NULL);
}

// ------- Interpolation in 3d ------- //

namespace {

/// linear interpolation on a grid with a special treatment for indices outside the grid
static inline double linInt(const double x, const double grid[], int size, int i1, int i2)
{
    double x1 = grid[i1<0 ? 0 : i1>=size ? size-1 : i1];
    double x2 = grid[i2<0 ? 0 : i2>=size ? size-1 : i2];
    if(x1==x2) {
        return x==x1 ? 1 : 0;
    } else {
        return (x-x1) / (x2-x1);
    }
}

/// find the index of the segment on the grid that the given point x belongs to,
/// and evaluate the values of B-spline functions of order N that are non-zero at this point
template<int N>
int bsplineWeights(const double x, const double grid[], int size, double B[])
{
    if(x<grid[0] || x>grid[size-1]) {
        for(int i=0; i<=N; i++)
            B[i] = NAN;
        return 0;
    }
    const int ind  = binSearch(x, grid, size);
    // de Boor's algorithm:
    // 0th order basis functions are all zero except the one on the grid segment `ind`
    for(int i=0; i<=N; i++)
        B[i] = i==N ? 1 : 0;
    for(int l=1; l<=N; l++) {
        double Bip1=0;
        for(int i=N, j=ind; j>=ind-l; i--, j--) {
            double Bi = B[i] * linInt(x, grid, size, j, j+l)
                      + Bip1 * linInt(x, grid, size, j+l+1, j+1);
            Bip1 = B[i];
            B[i] = Bi;
        }
    }
    return ind;
}

/** Compute the weights of kernel b-spline functions used for 1d interpolation.
    \tparam N   is the order of spline basis functions;
    \param[in]  x  is the input position on the grid;
    \param[in]  grid  is the array of grid nodes;
    \param[out] weights  are the amplitudes of N+1 basis functions at this point,
    to be multiplied by function values at grid nodes that enclose this point;
    if the point is outside the grid then the weights are filled with NaN and return value is 0;
    \return  the index of the leftmost out of N+1 grid nodes used in the interpolation.
*/
template<int N>
int bsplineInterp(const double x, const std::vector<double> &grid, double weights[N+1]);

/// specialization for the case of cubic b-splines
template<>
int bsplineInterp<3>(const double x, const std::vector<double> &grid, double weights[])
{
    const int size = grid.size();
    int ind = bsplineWeights<3>(x, &grid.front(), size, weights);
    // Normally the interpolation uses 4 values of original function at the adjacent nodes, as follows:
    // if  x[ind-1] < x[ind] <= x < x[ind+1] < x[ind+2], these are the values from ind-1 to ind+2.
    // However, if the point belongs to the first or the last grid segment, the left- or right-adjacent
    // segments would fall outside the grid, meaning that we need to identify the -1'th node with the 0th
    // (if ind==0), or similarly for ind==size-2. In these cases we shift the indexed nodes and their
    // weights by one, and ignore the extra 4th node.
    if(ind==0) {
        weights[0]+= weights[1];
        weights[1] = weights[2];
        weights[2] = weights[3];
        weights[3] = 0;
        return 0;
    }
    if(ind==size-2) {
        weights[3]+= weights[2];
        weights[2] = weights[1];
        weights[1] = weights[0];
        weights[0] = 0;
        return size-4;
    }
    return ind-1;
}

/// linear interpolation between two adjacent grid points
template<>
inline int bsplineInterp<1>(const double x, const std::vector<double> &grid, double weights[])
{
    if(x<grid.front() || x>grid.back()) {
        weights[0] = weights[1] = NAN;
        return 0;
    }
    int ind = binSearch(x, &grid.front(), grid.size());
    double dx = grid[ind+1] - grid[ind];
    weights[0] = (grid[ind+1]-x) / dx;
    weights[1] = (x-grid[ind]) / dx;
    return ind;
}

}  // internal namespace

template<int N>
BaseInterpolator3d<N>::BaseInterpolator3d(
    const std::vector<double>& xgrid, const std::vector<double>& ygrid, const std::vector<double>& zgrid,
    const std::vector<double>& fvalues) :
    xval(xgrid), yval(ygrid), zval(zgrid), fncvalues(fvalues)
{
    const unsigned int xsize = xval.size();
    const unsigned int ysize = yval.size();
    const unsigned int zsize = zval.size();
    if(xsize<N+1 || ysize<N+1 || zsize<N+1)
        throw std::invalid_argument("Error in 3d interpolator initialization: number of nodes is too small");
    if(!fncvalues.empty() && fncvalues.size() != xsize * ysize * zsize)
        throw std::invalid_argument("Error in 3d interpolator initialization: "
            "the array of function values is not compatible with sizes of coordinate grids");
}

template<int N>
void BaseInterpolator3d<N>::eval(const double vars[3], double *value) const
{
    if(fncvalues.empty())
        throw std::range_error("Error in 3d interpolator: function values not initialized");
    double weights[(N+1)*(N+1)*(N+1)];
    unsigned int leftInd[3];
    components(vars, leftInd, weights);
    *value=0;
    const unsigned int ysize = yval.size();
    const unsigned int zsize = zval.size();
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                *value += weights[(i * (N+1) + j) * (N+1) + k] *
                fncvalues[ ((i+leftInd[0]) * ysize + j+leftInd[1]) * zsize + k+leftInd[2] ]; 
}

template<int N>
void BaseInterpolator3d<N>::components(const double vars[3],
    unsigned int leftIndices[3], double weights[]) const
{
    if(isEmpty())
        throw std::range_error("Empty 3d interpolator");
    double bsplineWeights[3][N+1];
    for(int d=0; d<3; d++)
        leftIndices[d] = bsplineInterp<N>(vars[d], d==0? xval : d==1? yval : zval, bsplineWeights[d]);
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                weights[(i * (N+1) + j) * (N+1) + k] =
                    bsplineWeights[0][i] * bsplineWeights[1][j] * bsplineWeights[2][k];
}

// force the template instantiations to compile
template class BaseInterpolator3d<1>;
template class BaseInterpolator3d<3>;

//------------ GENERATION OF UNEQUALLY SPACED GRIDS ------------//

std::vector<double> createExpGrid(unsigned int nnodes, double xmin, double xmax)
{
    if(nnodes<2 || xmin<=0 || xmax<=xmin)
        throw std::invalid_argument("Invalid parameters for grid creation");
    double logmin = log(xmin), logmax = log(xmax);
    std::vector<double> grid(nnodes);
    grid.front() = xmin;
    grid.back()  = xmax;
    for(unsigned int k=1; k<nnodes-1; k++)
        grid[k] = exp(logmin + k*(logmax-logmin)/(nnodes-1));
    return grid;
}

// Creation of grid with cells increasing first near-linearly, then near-exponentially
class GridSpacingFinder: public IFunctionNoDeriv {
public:
    GridSpacingFinder(double _dynrange, int _nnodes) : dynrange(_dynrange), nnodes(_nnodes) {};
    virtual double value(const double A) const {
        return (A==0) ? nnodes-dynrange :
            (exp(A*nnodes)-1)/(exp(A)-1) - dynrange;
    }
private:
    double dynrange;
    int nnodes;
};

std::vector<double> createNonuniformGrid(unsigned int nnodes, double xmin, double xmax, bool zeroelem)
{   // create grid so that x_k = B*(exp(A*k)-1)
    if(nnodes<2 || xmin<=0 || xmax<=xmin)
        throw std::invalid_argument("Invalid parameters for grid creation");
    double A, B, dynrange=xmax/xmin;
    std::vector<double> grid(nnodes);
    int indexstart=zeroelem?1:0;
    if(zeroelem) {
        grid[0] = 0;
        nnodes--;
    }
    if(fcmp(static_cast<double>(nnodes), dynrange, 1e-6)==0) { // no need for non-uniform grid
        for(unsigned int i=0; i<nnodes; i++)
            grid[i+indexstart] = xmin+(xmax-xmin)*i/(nnodes-1);
        return grid;
    }
    // solve for A:  dynrange = (exp(A*nnodes)-1)/(exp(A)-1)
    GridSpacingFinder F(dynrange, nnodes);
    // first localize the root coarsely, to avoid overflows in root solver
    double Amin=0, Amax=0;
    double step=1;
    while(step>10./nnodes)
        step/=2;
    if(dynrange>nnodes) {
        while(Amax<10 && F(Amax)<=0)
            Amax+=step;
        Amin = Amax-step;
    } else {
        while(Amin>-10 && F(Amin)>=0)
            Amin-=step;
        Amax = Amin+step;
    }
    A = findRoot(F, Amin, Amax, 1e-4);
    B = xmin / (exp(A)-1);
    for(unsigned int i=0; i<nnodes; i++)
        grid[i+indexstart] = B*(exp(A*(i+1))-1);
    grid[nnodes-1+indexstart] = xmax;
    return grid;
}

/// creation of a grid with minimum guaranteed number of input points per bin
static void makegrid(std::vector<double>::iterator begin, std::vector<double>::iterator end, double startval, double endval)
{
    double step=(endval-startval)/(end-begin-1);
    while(begin!=end){
        *begin=startval;
        startval+=step;
        ++begin;
    }
    *(end-1)=endval;  // exact value
}

std::vector<double> createAlmostUniformGrid(const std::vector<double> &srcpoints, 
    unsigned int minbin, unsigned int& gridsize)
{
    if(srcpoints.size()==0)
        throw std::invalid_argument("Error in creating a grid: input points array is empty");
    gridsize = std::max<size_t>(2, std::min<size_t>(gridsize, static_cast<size_t>(srcpoints.size()/minbin)));
    std::vector<double> grid(gridsize);
    std::vector<double>::iterator gridbegin=grid.begin(), gridend=grid.end();
    std::vector<double>::const_iterator srcbegin=srcpoints.begin(), srcend=srcpoints.end();
    std::vector<double>::const_iterator srciter;
    std::vector<double>::iterator griditer;
    bool ok=true, directionBackward=false;
    int numChangesDirection=0;
    do{
        makegrid(gridbegin, gridend, *srcbegin, *(srcend-1));
        ok=true; 
        // find the index of bin with the largest number of points
        int largestbin=-1;
        unsigned int maxptperbin=0;
        for(srciter=srcbegin, griditer=gridbegin; griditer!=gridend-1; ++griditer) {
            unsigned int ptperbin=0;
            while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                ++ptperbin;
            if(ptperbin>maxptperbin) {
                maxptperbin=ptperbin;
                largestbin=griditer-grid.begin();
            }
            srciter+=ptperbin;
        }
        // check that all bins contain at least minbin srcpoints
        if(!directionBackward) {  // forward scan
            srciter = srcbegin;
            griditer = gridbegin;
            while(ok && griditer!=gridend-1) {
                unsigned int ptperbin=0;
                while(srciter+ptperbin!=srcend && *(srciter+ptperbin) < *(griditer+1)) 
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the next one
                {
                    ++griditer;
                    srciter+=ptperbin;
                } else {  // assign minbin points and decrease the available grid interval from the front
                    if(griditer-grid.begin() < largestbin) { 
                        // bad bin is closer to the grid front; move gridbegin forward
                        while(ptperbin<minbin && srciter+ptperbin!=srcend) 
                            ptperbin++;
                        if(srciter+ptperbin==srcend)
                            directionBackward=true; // oops, hit the end of array..
                        else {
                            srcbegin=srciter+ptperbin;
                            gridbegin=griditer+1;
                        }
                    } else {
                        directionBackward=true;
                    }   // will restart scanning from the end of the grid
                    ok=false;
                }
            }
        } else {  // backward scan
            srciter = srcend-1;
            griditer = gridend-1;
            while(ok && griditer!=gridbegin) {
                unsigned int ptperbin=0;
                while(srciter+1-ptperbin!=srcbegin && *(srciter-ptperbin) >= *(griditer-1))
                    ptperbin++;
                if(ptperbin>=minbin)  // ok, move to the previous one
                {
                    --griditer;
                    if(srciter+1-ptperbin==srcbegin)
                        srciter=srcbegin;
                    else
                        srciter-=ptperbin;
                } else {  // assign minbin points and decrease the available grid interval from the back
                    if(griditer-grid.begin() <= largestbin) { 
                        // bad bin is closer to the grid front; reset direction to forward
                        directionBackward=false;
                        numChangesDirection++;
                        if(numChangesDirection>10) {
//                            my_message(FUNCNAME, "grid creation seems not to converge?");
                            return grid;  // don't run forever but would not fulfill the minbin condition
                        }
                    } else {
                        // move gridend backward
                        while(ptperbin<minbin && srciter-ptperbin!=srcbegin) 
                            ++ptperbin;
                        if(srciter-ptperbin==srcbegin) {
                            directionBackward=false;
                            numChangesDirection++;
                            if(numChangesDirection>10) {
//                                my_message(FUNCNAME, "grid creation seems not to converge?");
                                return grid;  // don't run forever but would not fulfill the minbin condition
                            }
                        } else {
                            srcend=srciter-ptperbin+1;
                            gridend=griditer;
                        }
                    }
                    ok=false;
                }
            }
        }
    } while(!ok);
    return grid;
}

std::vector<double> mirrorGrid(const std::vector<double> &input)
{
    unsigned int size = input.size();
    if(size==0 || input[0]!=0)
        throw std::invalid_argument("incorrect input in mirrorGrid");
    std::vector<double> output(size*2-1);
    output[size-1] = 0;
    for(unsigned int i=1; i<size; i++) {
        if(input[i] <= input[i-1])
            throw std::invalid_argument("incorrect input in mirrorGrid");
        output[size-1-i] = -input[i];
        output[size-1+i] =  input[i];
    }
    return output;
}

}  // namespace
