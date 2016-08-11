#include "math_spline.h"
#include "math_core.h"
#include "math_fit.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#ifdef VERBOSE_REPORT
#include <iostream>
#endif
namespace math {

// ------- Some machinery for B-splines ------- //
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

/** Compute the weights of B-spline functions used for 1d interpolation.
    For any point inside the grid, at most N+1 basis functions are non-zero out of the entire set
    of (N_grid+N-1) basis functions; this routine reports only the nontrivial ones.
    \tparam N   is the degree of spline basis functions;
    \param[in]  x  is the input position on the grid;
    \param[in]  grid  is the array of grid nodes;
    \param[in]  size  is the length of this array;
    \param[out] B  are the values of N+1 possibly nonzero basis functions at this point,
    if the point is outside the grid then all values are zeros;
    \return  the index of the leftmost out of N+1 nontrivial basis functions.
*/
template<int N>
static inline int bsplineWeights(const double x, const double grid[], int size, double B[])
{
    if(x<grid[0] || x>grid[size-1]) {
        for(int i=0; i<=N; i++)
            B[i] = 0;
        return 0;
    }
    const int ind = binSearch(x, grid, size);
    // de Boor's algorithm:
    // 0th degree basis functions are all zero except the one on the grid segment `ind`
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

/// subexpression in b-spline derivatives (inverse distance between grid nodes or 0 if they coincide)
static inline double denom(const double grid[], int size, int i1, int i2)
{
    double x1 = grid[i1<0 ? 0 : i1>=size ? size-1 : i1];
    double x2 = grid[i2<0 ? 0 : i2>=size ? size-1 : i2];
    return x1==x2 ? 0 : 1 / (x2-x1);
}

/// recursive template definition for B-spline derivatives through B-spline derivatives
/// of lower degree and order; this is probably not very efficient but good enough for our purposes;
/// the arguments are the same as for `bsplineWeights`, and `order` is the order of derivative.
template<int N, int Order>
static inline int bsplineDerivs(const double x, const double grid[], int size, double B[])
{
    int ind = bsplineDerivs<N-1, Order-1>(x, grid, size, B+1);
    B[0] = 0;
    for(int i=0, j=ind-N; i<=N; i++, j++) {
        B[i] = N * (B[i]   * denom(grid, size, j, j+N)
            + (i<N? B[i+1] * denom(grid, size, j+N+1, j+1) : 0) );
    }
    return ind;
}

/// the above recursion terminates when the order of derivative is zero, returning B-spline values;
/// however, C++ rules do not permit to declare a partial function template specialization
/// (for arbitrary N and order 0), therefore we use full specializations for several values of N
template<>
static inline int bsplineDerivs<0,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineWeights<0>(x, grid, size, B);
}
template<>
static inline int bsplineDerivs<1,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineWeights<1>(x, grid, size, B);
}
template<>
static inline int bsplineDerivs<2,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineWeights<2>(x, grid, size, B);
}
template<>
static inline int bsplineDerivs<3,0>(const double x, const double grid[], int size, double B[]) {
    return bsplineWeights<3>(x, grid, size, B);
}
template<>
static inline int bsplineDerivs<0,1>(const double, const double[], int, double[]) {
    assert(!"Should not be called");
}
template<>
static inline int bsplineDerivs<0,2>(const double, const double[], int, double[]) {
    assert(!"Should not be called");
}

/** Similar to bsplineWeights, but uses linear extrapolation outside the grid domain */
template<int N>
static inline int bsplineWeightsExtrapolated(const double x, const double grid[], int size, double B[])
{
    if(x<grid[0] || x>grid[size-1]) {
        // extrapolate using the derivatives
        double x0 = x<grid[0] ? grid[0] : grid[size-1];
        int ind = bsplineWeights<N>(x0, grid, size, B);
        double D[N+1];
        bsplineDerivs<N,1>(x0, grid, size, D);
        for(int i=0; i<=N; i++)
            B[i] += D[i] * (x-x0);
        return ind;
    }
    else return bsplineWeights<N>(x, grid, size, B);
}

/** Compute the matrix of overlap integrals for the array of 1d B-spline functions or their derivs.
    Let N>=1 be the degree of B-splines, and Order - the order of derivative in question.
    There are numBasisFnc = numKnots+N-1 basis functions B_p(x) on the entire interval spanned by knots,
    and each of them is nonzero on at most N+1 consecutive sub-intervals between knots.
    Define the matrix M_{pq}, 0<=p<=q<numBasisFnc, to be the symmetric matrix of overlap intervals:
    \f$  M_{pq} = \int dx B_p(x) B_q(x)  \f$, where the integrand is nonzero on at most q-p+N+1 
    consecutive sub-intervals.
    This routine fills the values of this matrix passed as the output argument `mat`.
*/
template<int N, int Order>
static Matrix<double> computeOverlapMatrix(const std::vector<double> &knots)
{
    int numKnots = knots.size(), numBasisFnc = numKnots+N-1;
    // B-spline of degree N is a polynomial of degree N, so its Order'th derivative is a polynomial
    // of degree N-Order. To compute the integral of a product of two such functions over a sub-interval,
    // it is sufficient to employ a Gauss-Legendre quadrature rule with the number of nodes = N-Order+1.
    const int Nnodes = std::max<int>(N-Order+1, 0);
    double glnodes[Nnodes], glweights[Nnodes];
    prepareIntegrationTableGL(0, 1, Nnodes, glnodes, glweights);

    // Collect the values of all possibly non-zero basis functions (or their Order'th derivatives)
    // at Nnodes points of each sub-interval between knots. There are at most N+1 such non-zero functions,
    // so these values are stored in a 2d array [N+1] x [number of subintervals * number of GL nodes].
    Matrix<double> values(N+1, (numKnots-1)*Nnodes);
    for(int k=0; k<numKnots-1; k++) {
        double der[N+1];
        for(int n=0; n<Nnodes; n++) {
            // evaluate the possibly non-zero functions and keep track of the index of the leftmost one
            int ind = bsplineDerivs<N, Order> ( knots[k] + (knots[k+1] - knots[k]) * glnodes[n],
                &knots.front(), numKnots, der);
            for(int b=0; b<=N; b++)
                values(b, k*Nnodes+n) = der[b+k-ind];
        }
    }

    // evaluate overlap integrals and store them in the symmetric matrix M_pq, which is a banded matrix
    // with nonzero values only within N+1 cells from the diagonal
    Matrix<double> mat(numBasisFnc, numBasisFnc);
    mat.fill(0);
    for(int p=0; p<numBasisFnc; p++) {
        int kmin = std::max<int>(p-N, 0);   // index of leftmost knot of the integration sub-intervals
        int kmax = std::min<int>(p+1, numKnots-1);     // same for the rightmost
        int qmax = std::min<int>(p+N+1, numBasisFnc);  // max index of the column of the banded matrix
        for(int q=p; q<qmax; q++) {
            double result = 0;
            // loop over sub-intervals where the integrand might be nonzero
            for(int k=kmin; k<kmax; k++) {
                double dx = knots[k+1]-knots[k];
                // loop over nodes of GL quadrature rule over the sub-interval
                for(int n=0; n<Nnodes; n++) {
                    double P = p>=k && p<=k+N ? values(p-k, k*Nnodes+n) : 0;
                    double Q = q>=k && q<=k+N ? values(q-k, k*Nnodes+n) : 0;
                    result  += P * Q * glweights[n] * dx;
                }
            }
            mat(p, q) = result;
            mat(q, p) = result;  // it is symmetric
        }
    }
    return mat;
}

#if 0
/** Convert the weighted combination of 1d B-spline functions of degree N=3
    \f$  f(x) = \sum_{i=0}^{K+1}  A_i  B_i(x)  \f$,  defined by grid nodes X[k] (0<=k<K),
    into the input data for an ordinary clamped cubic spline:
    the values of f(x) at grid knots plus two endpoint derivatives.
    \param[in]  ampl  is the array of K+2 amplitudes of each B-spline basis function;
    \param[in]  grid  is the array of K grid nodes;
    \param[out] fvalues  will contain the values of f at the grid nodes;
    \param[out] derivLeft, derifRight  will contain the two derivatives at X[0] and X[K-1].
    \throws  std::invalid_argument exception if the input grid is not in ascending order
    or does not match the size of amplitudes array.
*/
static void convertToCubicSpline(const std::vector<double> &ampl, const std::vector<double> &grid,
    std::vector<double> &fvalues, double &derivLeft, double &derivRight)
{
    unsigned int numKnots = grid.size();
    if(ampl.size() != numKnots+2)
        throw std::invalid_argument("convertToCubicSpline: ampl.size() != grid.size()+2");
    fvalues.assign(numKnots, 0);
    for(unsigned int k=0; k<numKnots; k++) {
        if(k>0 && grid[k]<=grid[k-1])
            throw std::invalid_argument("convertToCubicSpline: grid nodes must be in ascending order");
        // for any x, at most 4 basis functions are non-zero, starting from ind
        double Bspl[4];
        int ind = bsplineWeights<3>(grid[k], &grid[0], numKnots, Bspl);
        double val=0;
        for(int p=0; p<=3; p++)
            val += Bspl[p] * ampl[p+ind];
        fvalues[k] = val;
        if(k==0 || k==numKnots-1) {  // at endpoints also compute derivatives
            bsplineDerivs<3,1>(grid[k], &grid[0], numKnots, Bspl);
            double der=0;
            for(int p=0; p<=3; p++)
                der += Bspl[p] * ampl[p+ind];
            if(k==0)
                derivLeft = der;
            else
                derivRight = der;
        }
    }
}
#endif
}  // internal namespace


BaseInterpolator1d::BaseInterpolator1d(const std::vector<double>& xv, const std::vector<double>& fv) :
    xval(xv), fval(fv)
{
    if(xv.size() < 2)
        throw std::invalid_argument("Error in 1d interpolator: number of nodes should be >=2");
    for(unsigned int i=1; i<xv.size(); i++)
        if(xv[i] <= xv[i-1])
            throw std::invalid_argument("Error in 1d interpolator: "
                "x values must be monotonically increasing");
}

LinearInterpolator::LinearInterpolator(const std::vector<double>& xv, const std::vector<double>& yv) :
    BaseInterpolator1d(xv, yv)
{
    if(fval.size() != xval.size())
        throw std::invalid_argument("LinearInterpolator: input arrays are not equal in length");
}

void LinearInterpolator::evalDeriv(const double x, double* value, double* deriv, double* deriv2) const
{
    if(value) {
        int i = x<=xval.front() ? 0 : x>=xval.back() ? xval.size()-2 : binSearch(x, &xval[0], xval.size());
        *value = linearInterp(x, xval[i], xval[i+1], fval[i], fval[i+1]);
    }
    if(deriv)
        *deriv = 0;
    if(deriv2)
        *deriv2 = 0;
}


//-------------- CUBIC SPLINE --------------//

CubicSpline::CubicSpline(const std::vector<double>& _xval,
    const std::vector<double>& _fval, double der1, double der2) :
    BaseInterpolator1d(_xval, _fval)
{
    unsigned int numPoints = xval.size();
    if(fval.size() == numPoints+2 && der1!=der1 && der2!=der2) {
        // initialize from the amplitudes of B-splines defined at these nodes
        std::vector<double> ampl(_fval);  // temporarily store the amplitudes of B-splines
        fval.assign(numPoints, 0);
        fder2.resize(numPoints);
        for(unsigned int i=0; i<numPoints; i++) {
            // compute values and second derivatives of B-splines at grid nodes
            double val[4], der2[4];
            int ind = bsplineWeights<3>(xval[i], &xval[0], numPoints, val);
            bsplineDerivs<3,2>(xval[i], &xval[0], numPoints, der2);
            for(int p=0; p<=3; p++) {
                fval [i] += val [p] * ampl[p+ind];
                fder2[i] += der2[p] * ampl[p+ind];
            }
        }
        return;
    }
    if(fval.size() != numPoints)
        throw std::invalid_argument("CubicSpline: input arrays are not equal in length");
    std::vector<double> rhs(numPoints-2), diag(numPoints-2), offdiag(numPoints-3);
    for(unsigned int i = 1; i < numPoints-1; i++) {
        const double
        dxm = xval[i] - xval[i-1],
        dxp = xval[i+1] - xval[i],
        dym = (fval[i] - fval[i-1]) / dxm,
        dyp = (fval[i+1] - fval[i]) / dxp;
        if(i < numPoints-2)
            offdiag[i-1] = dxp;
        diag[i-1] = 2.0 * (dxp + dxm);
        rhs [i-1] = 6.0 * (dyp - dym);
        if(i == 1 && isFinite(der1)) {
            diag[i-1] = 1.5 * dxm + 2.0 * dxp;
            rhs [i-1] = 6.0 * dyp - 9.0 * dym + 3.0 * der1;
        }
        if(i == numPoints-2 && isFinite(der2)) {
            diag[i-1] = 1.5 * dxp + 2.0 * dxm;
            rhs [i-1] = 9.0 * dyp - 6.0 * dym - 3.0 * der2;
        }
    }

    if(numPoints == 2) {
        fder2.assign(2, 0.);
    } else
    if(numPoints == 3) {
        fder2.assign(3, 0.);
        fder2[1] = rhs[0] / diag[0];
    } else {
        linearSystemSolveTridiagSymm(diag, offdiag, rhs, fder2);
        fder2.insert(fder2.begin(), 0.);  // for natural cubic spline,
        fder2.push_back(0.);             // 2nd derivatives are zero at endpoints;
    }
    if(isFinite(der1))              // but for a clamped spline they are not.
        fder2[0] = ( 3 * (fval[1]-fval[0]) / (xval[1]-xval[0]) 
            -3 * der1 - 0.5 * fder2[1] * (xval[1]-xval[0]) ) / (xval[1]-xval[0]);
    if(isFinite(der2))
        fder2[numPoints-1] = (
            -3 * (fval[numPoints-1]-fval[numPoints-2]) / (xval[numPoints-1]-xval[numPoints-2]) 
            +3 * der2 - 0.5 * fder2[numPoints-2] * (xval[numPoints-1]-xval[numPoints-2]) ) /
            (xval[numPoints-1]-xval[numPoints-2]);
}

namespace {
// definite integral of x^(m+n)
class MonomialIntegral: public IFunctionIntegral {
    const int n;
public:
    MonomialIntegral(int _n) : n(_n) {};
    virtual double integrate(double x1, double x2, int m=0) const {
        return m+n+1==0 ? log(x2/x1) : (powInt(x2, m+n+1) - powInt(x1, m+n+1)) / (m+n+1);
    }
};

// evaluate spline value, derivative and 2nd derivative at once (faster than doing it separately);
// possibly for several splines (K>=1), k=0,...,K-1);
// input arguments contain the value(s) and 2nd derivative(s) of these splines
// at the boundaries of interval [xl..xh] that contain the point x.
template<unsigned int K>
static inline void evalCubicSplines(
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
}  // internal namespace

void CubicSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        double dx  =  xval[1]-xval[0];
        double der = (fval[1]-fval[0]) / dx - dx * (1./6 * fder2[1] + 1./3 * fder2[0]);
        if(val)
            *val   = fval[0] +
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
        double der = (fval[size-1]-fval[size-2]) / dx + dx * (1./6 * fder2[size-2] + 1./3 * fder2[size-1]);
        if(val)
            *val   = fval[size-1] + (der==0 ? 0 : der*(x-xval[size-1]));
        if(deriv)
            *deriv = der;
        if(deriv2)
            *deriv2= 0;
        return;
    }

    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalCubicSplines<1> (xval[index], xval[index+1], x, 
        &fval[index], &fval[index+1], &fder2[index], &fder2[index+1], val, deriv, deriv2);
}

bool CubicSpline::isMonotonic() const
{
    if(xval.size()==0)
        throw std::range_error("Empty spline");
    bool ismonotonic=true;
    for(unsigned int index=0; ismonotonic && index < xval.size()-1; index++) {
        const double
        dx = xval[index + 1] - xval[index],
        dy = fval[index + 1] - fval[index],
        cl = fder2[index],
        ch = fder2[index+1],
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
        double der = (fval[1]-fval[0]) / dx - dx * (1./6 * fder2[1] + 1./3 * fder2[0]);
        double X2  = fmin(x2, xval.front());
        result +=
            f.integrate(x1, X2, 0) * (fval.front() - der * xval.front()) +
            f.integrate(x1, X2, 1) * der;
        if(x2<=xval.front())
            return result;
        x1 = xval.front();
    }
    if(x2>xval.back()) {    // same for x>xval[end]
        unsigned int size = xval.size();
        double dx  =  xval[size-1]-xval[size-2];
        double der = (fval[size-1]-fval[size-2]) / dx + dx * (1./6 * fder2[size-2] + 1./3 * fder2[size-1]);
        double X1  = fmax(x1, xval.back());
        result +=
            f.integrate(X1, x2, 0) * (fval.back() - der * xval.back()) +
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
        double a  = fval[i];
        double c  = fder2[i];
        double b  = (fval[i+1] - a) / h - h * (1./6 * fder2[i+1] + 1./3 * c);
        double d  = (fder2[i+1] - c) / h;
        // spline(x) = fval[i] + dx * (b + dx * (c/2 + dx*d/6)), where dx = x-xval[i]
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

HermiteSpline::HermiteSpline(const std::vector<double>& _xval,
    const std::vector<double>& _fval, const std::vector<double>& _fder) :
    BaseInterpolator1d(_xval, _fval), fder(_fder)
{
    unsigned int numPoints = xval.size();
    if(fval.size() != numPoints || fder.size() != numPoints)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
}

void HermiteSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        if(val)
            *val   = fval.front() +
            (fder.front()==0 ? 0 : fder.front() * (x-xval.front()));
            // if der==0, will give correct result even for infinite x
        if(deriv)
            *deriv = fder.front();
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        if(val)
            *val   = fval.back() + (fder.back()==0 ? 0 : fder.back() * (x-xval.back()));
        if(deriv)
            *deriv = fder.back();
        if(deriv2)
            *deriv2= 0;
        return;
    }
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    const double dx = xval[index+1]-xval[index];
    const double t = (x-xval[index]) / dx;
    if(val)
        *val = pow_2(1-t) * ( (1+2*t) * fval[index]   + t     * fder[index]   * dx )
             + pow_2(t)   * ( (3-2*t) * fval[index+1] + (t-1) * fder[index+1] * dx );
    if(deriv)
        *deriv = 6*t*(1-t) * (fval[index+1]-fval[index]) / dx
               + (1-t)*(1-3*t) * fder[index] + t*(3*t-2) * fder[index+1];
    if(deriv2)
        *deriv2 = ( (6-12*t) * (fval[index+1]-fval[index]) / dx
                + (6*t-4) * fder[index] + (6*t-2) * fder[index+1] ) / dx;
}

// ------ Quintic spline ------- //
QuinticSpline::QuinticSpline(const std::vector<double>& _xval, 
    const std::vector<double>& _fval, const std::vector<double>& _fder): 
    BaseInterpolator1d(_xval, _fval), fder(_fder), fder3(xval.size())
{
    unsigned int numPoints = xval.size();
    if(fval.size() != numPoints || fder.size() != numPoints)
        throw std::invalid_argument("Error in spline initialization: input arrays are not equal in length");
    std::vector<double> v(numPoints-1);
    double dx = xval[1]-xval[0];
    double dy = fval[1]-fval[0];
    fder3[0]  = v[0] = 0.;
    for(unsigned int i=1; i<numPoints-1; i++) {
        double dx1 = xval[i+1] - xval[i];
        double dx2 = xval[i+1] - xval[i-1];
        double dy1 = fval[i+1] - fval[i];
        double sig = dx/dx2;
        double p   = sig*v[i-1] - 3;
        double der3= 12 * ( 7*fder[i]*dx2 / (dx*dx1) 
            + 3 * (fder[i-1]/dx + fder[i+1]/dx1)
            - 10* (dy / (dx*dx) + dy1 / (dx1*dx1)) ) / dx2;
        fder3[i]   = (der3 - sig*fder3[i-1] ) / p;
        v[i]       = (sig-1)/p;
        dx = dx1;
        dy = dy1;
    }
    fder3[numPoints-1] = 0.;
    for(unsigned int i=numPoints-1; i>0; i--)
        fder3[i-1] += v[i-1]*fder3[i];
}

namespace{

template<unsigned int K>   // K>=1 - number of splines to compute; k=0,...,K-1
static inline void evalQuinticSplines(
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
}

void QuinticSpline::evalDeriv(const double x, double* val, double* deriv, double* deriv2) const
{
    if(xval.size() == 0)
        throw std::range_error("Empty spline");
    if(x < xval.front()) {
        if(val)
            *val   = fval[0] + fder[0]*(x-xval[0]);
        if(deriv)
            *deriv = fder[0];
        if(deriv2)
            *deriv2= 0;
        return;
    }
    if(x > xval.back()) {
        const unsigned int size = xval.size();
        if(val)
            *val   = fval[size-1] + fder[size-1]*(x-xval[size-1]);
        if(deriv)
            *deriv = fder[size-1];
        if(deriv2)
            *deriv2= 0;
        return;
    }
    unsigned int index = binSearch(x, &xval.front(), xval.size());
    evalQuinticSplines<1> (xval[index], xval[index+1], x,
        &fval[index], &fval[index+1], &fder[index], &fder[index+1], &fder3[index], &fder3[index+1],
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
        &fval[index], &fval[index+1], &fder[index], &fder[index+1], &fder3[index], &fder3[index+1],
        NULL, NULL, NULL, &der3);
    return der3;
}


// ------ INTERPOLATION IN 2D ------ //

BaseInterpolator2d::BaseInterpolator2d(
    const std::vector<double>& xgrid, const std::vector<double>& ygrid,
    const Matrix<double>& fvalues) :
    xval(xgrid), yval(ygrid), fval(fvalues)
{
    const unsigned int xsize = xgrid.size();
    const unsigned int ysize = ygrid.size();
    if(xsize<2 || ysize<2)
        throw std::invalid_argument(
            "Error in 2d interpolator initialization: number of nodes should be >=2 in each direction");
    if(fvalues.rows() != xsize)
        throw std::invalid_argument(
            "Error in 2d interpolator initialization: x and f array lengths differ");
    if(fvalues.cols() != ysize)
        throw std::invalid_argument(
            "Error in 2d interpolator initialization: y and f array lengths differ");
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
    const double zlowlow = fval(xi, yi);
    const double zlowupp = fval(xi, yi + 1);
    const double zupplow = fval(xi + 1, yi);
    const double zuppupp = fval(xi + 1, yi + 1);
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
    const Matrix<double>& fvalues,
    double deriv_xmin, double deriv_xmax, double deriv_ymin, double deriv_ymax) :
    BaseInterpolator2d(xgrid, ygrid, fvalues)
{
    const unsigned int xsize = xval.size();
    const unsigned int ysize = yval.size();
    fx.resize (xsize, ysize);
    fy.resize (xsize, ysize);
    fxy.resize(xsize, ysize);
    std::vector<double> tmpvalues(ysize);
    for(unsigned int i=0; i<xsize; i++) {
        for(unsigned int j=0; j<ysize; j++)
            tmpvalues[j] = fval(i, j);
        CubicSpline spl(yval, tmpvalues, deriv_ymin, deriv_ymax);
        for(unsigned int j=0; j<ysize; j++)
            spl.evalDeriv(yval[j], NULL, &fy(i, j));
    }
    tmpvalues.resize(xsize);
    for(unsigned int j=0; j<ysize; j++) {
        for(unsigned int i=0; i<xsize; i++)
            tmpvalues[i] = fval(i, j);
        CubicSpline spl(xval, tmpvalues, deriv_xmin, deriv_xmax);
        for(unsigned int i=0; i<xsize; i++)
            spl.evalDeriv(xval[i], NULL, &fx(i, j));
    }
    for(unsigned int j=0; j<ysize; j++) {
        // if derivs at the boundary are specified, 2nd deriv must be zero
        if( (j==0 && isFinite(deriv_ymin)) || (j==ysize-1 && isFinite(deriv_ymax)) ) {
            for(unsigned int i=0; i<xsize; i++)
                fxy(i, j) = 0.;
        } else {
            for(unsigned int i=0; i<xsize; i++)
                tmpvalues[i] = fy(i, j);
            CubicSpline spl(xval, tmpvalues,
                isFinite(deriv_xmin) ? 0. : NAN, isFinite(deriv_xmax) ? 0. : NAN);
            for(unsigned int i=0; i<xsize; i++)
                spl.evalDeriv(xval[i], NULL, &fxy(i, j));
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
        zlowlow = fval(xi,   yi),
        zlowupp = fval(xi,   yi+1),
        zupplow = fval(xi+1, yi),
        zuppupp = fval(xi+1, yi+1),
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
        zxlowlow  = fx (xi  , yi  ) / dt,
        zxlowupp  = fx (xi  , yi+1) / dt,
        zxupplow  = fx (xi+1, yi  ) / dt,
        zxuppupp  = fx (xi+1, yi+1) / dt,
        zylowlow  = fy (xi  , yi  ) / du,
        zylowupp  = fy (xi  , yi+1) / du,
        zyupplow  = fy (xi+1, yi  ) / du,
        zyuppupp  = fy (xi+1, yi+1) / du,
        zxylowlow = fxy(xi  , yi  ) / dtdu,
        zxylowupp = fxy(xi  , yi+1) / dtdu,
        zxyupplow = fxy(xi+1, yi  ) / dtdu,
        zxyuppupp = fxy(xi+1, yi+1) / dtdu,
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
    const Matrix<double>& fvalues, const Matrix<double>& dfdx, const Matrix<double>& dfdy) :
    BaseInterpolator2d(xgrid, ygrid, fvalues), fx(dfdx), fy(dfdy)
{
    const unsigned int xsize = xval.size();
    const unsigned int ysize = yval.size();
    // 1. for each y do 1d quintic spline for z in x, and record d^3z/dx^3
    fxxx.resize(xsize, ysize);
    fyyy.resize(xsize, ysize);
    fxyy.resize(xsize, ysize);
    fxxxyy.resize(xsize, ysize);
    std::vector<double> t(xsize), t1(xsize);
    for(unsigned int j=0; j<ysize; j++) {
        for(unsigned int i=0; i<xsize; i++) {
            t[i]  = fval(i, j);
            t1[i] = dfdx(i, j);
        }
        QuinticSpline s(xval, t, t1);
        for(unsigned int i=0; i<xsize; i++)
            fxxx(i, j) = s.deriv3(xval[i]);
    }
    // 2. for each x do 1d quintic spline for z and splines for dz/dx, d^3z/dx^3 in y
    t.resize(ysize);
    t1.resize(ysize);
    for(unsigned int i=0; i<xsize; i++) {
        for(unsigned int j=0; j<ysize; j++) {
            t[j]  = fval(i, j);
            t1[j] = dfdy(i, j);
        }
        QuinticSpline s(yval, t, t1);
        for(unsigned int j=0; j<ysize; j++)
            t1[j] = dfdx(i, j);
        CubicSpline u(yval, t1, 0, 0);
        for(unsigned int j=0; j<ysize; j++)
            t1[j] = fxxx(i, j);
        CubicSpline v(yval, t1);
        for(unsigned int j=0; j<ysize; j++) {
            fyyy(i, j) = s.deriv3(yval[j]);
            u.evalDeriv(yval[j], NULL, NULL, &fxyy(i, j));
            v.evalDeriv(yval[j], NULL, NULL, &fxxxyy(i, j));
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
        fl[2]  = { fval(xl, yl), fval(xu, yl) },
        fh[2]  = { fval(xl, yu), fval(xu, yu) },
        f1l[2] = { fy  (xl, yl), fy  (xu, yl) },
        f1h[2] = { fy  (xl, yu), fy  (xu, yu) },
        f3l[2] = { fyyy(xl, yl), fyyy(xu, yl) },
        f3h[2] = { fyyy(xl, yu), fyyy(xu, yu) },
        flo[4] = { fx  (xl, yl), fx  (xu, yl), fxxx(xl, yl),   fxxx(xu, yl) },
        fhi[4] = { fx  (xl, yu), fx  (xu, yu), fxxx(xl, yu),   fxxx(xu, yu) },
        f2l[4] = { fxyy(xl, yl), fxyy(xu, yl), fxxxyy(xl, yl), fxxxyy(xu, yl) },
        f2h[4] = { fxyy(xl, yu), fxyy(xu, yu), fxxxyy(xl, yu), fxxxyy(xu, yu) };
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

template<int N>
BsplineInterpolator3d<N>::BsplineInterpolator3d(
    const std::vector<double>& xgrid, const std::vector<double>& ygrid, const std::vector<double>& zgrid) :
    xnodes(xgrid), ynodes(ygrid), znodes(zgrid),
    numComp(indComp(xnodes.size()+N-2, ynodes.size()+N-2, znodes.size()+N-2)+1)
{
    if(xnodes.size()<2 || ynodes.size()<2 || znodes.size()<2)
        throw std::invalid_argument("BsplineInterpolator3d: number of nodes is too small");
    bool monotonic = true;
    for(unsigned int i=1; i<xnodes.size(); i++)
        monotonic &= xnodes[i-1] < xnodes[i];
    for(unsigned int i=1; i<ynodes.size(); i++)
        monotonic &= ynodes[i-1] < ynodes[i];
    for(unsigned int i=1; i<znodes.size(); i++)
        monotonic &= znodes[i-1] < znodes[i];
    if(!monotonic)
        throw std::invalid_argument("BsplineInterpolator3d: grid nodes must be sorted in ascending order");
}

template<int N>
void BsplineInterpolator3d<N>::nonzeroComponents(const double point[3],
    unsigned int leftIndices[3], double values[]) const
{
    double weights[3][N+1];
    for(int d=0; d<3; d++) {
        const std::vector<double>& nodes = d==0? xnodes : d==1? ynodes : znodes;
        leftIndices[d] = bsplineWeights<N>(point[d], &nodes[0], nodes.size(), weights[d]);
    }
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                values[(i * (N+1) + j) * (N+1) + k] = weights[0][i] * weights[1][j] * weights[2][k];
}

template<int N>
double BsplineInterpolator3d<N>::interpolate(
    const double point[3], const std::vector<double> &amplitudes) const
{
    if(amplitudes.size() != numComp)
        throw std::range_error("BsplineInterpolator3d: invalid size of amplitudes array");
    double weights[(N+1)*(N+1)*(N+1)];
    unsigned int leftInd[3];
    nonzeroComponents(point, leftInd, weights);
    double val=0;
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                val += weights[ (i * (N+1) + j) * (N+1) + k ] *
                    amplitudes[ indComp(i+leftInd[0], j+leftInd[1], k+leftInd[2]) ];
    return val;
}

template<int N>
void BsplineInterpolator3d<N>::eval(const double point[3], double values[]) const
{
    unsigned int leftInd[3];
    double weights[(N+1)*(N+1)*(N+1)];
    nonzeroComponents(point, leftInd, weights);
    for(unsigned int i=0; i<numComp; i++)
        values[i] = 0;
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                values[ indComp(i+leftInd[0], j+leftInd[1], k+leftInd[2]) ] =
                    weights[(i * (N+1) + j) * (N+1) + k];
}

template<int N>
double BsplineInterpolator3d<N>::valueOfComponent(const double point[3], unsigned int indComp) const
{
    if(indComp>=numComp)
        throw std::range_error("BsplineInterpolator3d: component index out of range");
    unsigned int leftInd[3], indices[3];
    double weights[(N+1)*(N+1)*(N+1)];
    nonzeroComponents(point, leftInd, weights);
    decomposeIndComp(indComp, indices);
    if( indices[0]>=leftInd[0] && indices[0]<=leftInd[0]+N &&
        indices[1]>=leftInd[1] && indices[1]<=leftInd[1]+N &&
        indices[2]>=leftInd[2] && indices[2]<=leftInd[2]+N )
        return weights[ ((indices[0]-leftInd[0]) * (N+1)
                         +indices[1]-leftInd[1]) * (N+1) + indices[2]-leftInd[2] ];
    else
        return 0;
}

template<int N>
void BsplineInterpolator3d<N>::nonzeroDomain(unsigned int indComp,
    double xlower[3], double xupper[3]) const
{
    if(indComp>=numComp)
        throw std::range_error("BsplineInterpolator3d: component index out of range");
    unsigned int indices[3];
    decomposeIndComp(indComp, indices);
    for(int d=0; d<3; d++) {
        const std::vector<double>& nodes = d==0? xnodes : d==1? ynodes : znodes;
        xlower[d] = nodes[ indices[d]<N ? 0 : indices[d]-N ];
        xupper[d] = nodes[ std::min<unsigned int>(indices[d]+1, nodes.size()-1) ];
    }
}

template<int N>
SpMatrix<double> BsplineInterpolator3d<N>::computeRoughnessPenaltyMatrix() const
{
    std::vector<Triplet> values;      // elements of sparse matrix will be accumulated here
    Matrix<double>
    X0(computeOverlapMatrix<N,0>(xnodes)),  // matrices of products of 1d basis functions or derivs
    X1(computeOverlapMatrix<N,1>(xnodes)),
    X2(computeOverlapMatrix<N,2>(xnodes)),
    Y0(computeOverlapMatrix<N,0>(ynodes)),
    Y1(computeOverlapMatrix<N,1>(ynodes)),
    Y2(computeOverlapMatrix<N,2>(ynodes)),
    Z0(computeOverlapMatrix<N,0>(znodes)),
    Z1(computeOverlapMatrix<N,1>(znodes)),
    Z2(computeOverlapMatrix<N,2>(znodes));
    for(unsigned int index1=0; index1<numComp; index1++) {
        unsigned int ind[3];
        decomposeIndComp(index1, ind);
        // use the fact that in each dimension, the overlap matrix elements are zero if
        // |rowIndex-colIndex| > N (i.e. it is a band matrix with width 2N+1).
        unsigned int
        imin = ind[0]<N ? 0 : ind[0]-N,
        jmin = ind[1]<N ? 0 : ind[1]-N,
        kmin = ind[2]<N ? 0 : ind[2]-N,
        imax = std::min<unsigned int>(ind[0]+N+1, xnodes.size()+N-1),
        jmax = std::min<unsigned int>(ind[1]+N+1, ynodes.size()+N-1),
        kmax = std::min<unsigned int>(ind[2]+N+1, znodes.size()+N-1);
        for(unsigned int i=imin; i<imax; i++) {
            for(unsigned int j=jmin; j<jmax; j++) {
                for(unsigned int k=kmin; k<kmax; k++) {
                    unsigned int index2 = indComp(i, j, k);
                    if(index2>index1)
                        continue;  // will initialize from a symmetric element
                    double val =
                        X2(ind[0], i) * Y0(ind[1], j) * Z0(ind[2], k) +
                        X0(ind[0], i) * Y2(ind[1], j) * Z0(ind[2], k) + 
                        X0(ind[0], i) * Y0(ind[1], j) * Z2(ind[2], k) + 
                        X1(ind[0], i) * Y1(ind[1], j) * Z0(ind[2], k) * 2 +
                        X0(ind[0], i) * Y1(ind[1], j) * Z1(ind[2], k) * 2 +
                        X1(ind[0], i) * Y0(ind[1], j) * Z1(ind[2], k) * 2;
                    values.push_back(Triplet(index1, index2, val));
                    if(index1!=index2)
                        values.push_back(Triplet(index2, index1, val));
                }
            }
        }
    }
    return SpMatrix<double>(numComp, numComp, values);
}

template<int N>
std::vector<double> createInterpolator3dArray(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes)
{
    if(F.numVars() != 3 || F.numValues() != 1)
        throw std::invalid_argument(
            "createInterpolator3dArray: input function must have numVars=3, numValues=1");
    BsplineInterpolator3d<N> interp(xnodes, ynodes, znodes);

    // collect the function values at all nodes of 3d grid
    std::vector<double> fncvalues(interp.numValues());
    double point[3];
    for(unsigned int i=0; i<xnodes.size(); i++) {
        point[0] = xnodes[i];
        for(unsigned int j=0; j<ynodes.size(); j++) {
            point[1] = ynodes[j];
            for(unsigned int k=0; k<znodes.size(); k++) {
                point[2] = znodes[k];
                unsigned int index = interp.indComp(i, j, k);
                F.eval(point, &fncvalues[index]);
            }
        }
    }
    if(N==1)
        // in this case no further action is necessary: the values of function at grid nodes
        // are identical to the amplitudes used in the interpolation
        return fncvalues;

    // the matrix of values of basis functions at grid nodes (could be *BIG*, although it is sparse)
    std::vector<Triplet> values;  // elements of sparse matrix will be accumulated here
    const std::vector<double>* nodes[3] = {&xnodes, &ynodes, &znodes};
    // values of 1d B-splines at each grid node in each of the three dimensions, or -
    // for the last two rows in each matrix - 2nd derivatives of B-splines at the first/last grid nodes
    Matrix<double> weights[3];
    // indices of first non-trivial B-spline functions at each grid node in each dimension
    std::vector<int> leftInd[3];

    // collect the values of all basis functions at each grid node in each dimension
    for(int d=0; d<3; d++) {
        unsigned int Ngrid = nodes[d]->size();
        weights[d].resize(Ngrid+N-1, N+1);
        leftInd[d].resize(Ngrid+N-1);
        const double* arr = &(nodes[d]->front());
        for(unsigned int n=0; n<Ngrid; n++)
            leftInd[d][n] = bsplineWeights<N>(arr[n], arr, Ngrid, &weights[d](n, 0));
        // collect 2nd derivatives at the endpoints
        leftInd[d][Ngrid]   = bsplineDerivs<N,2>(arr[0],       arr, Ngrid, &weights[d](Ngrid,   0));
        leftInd[d][Ngrid+1] = bsplineDerivs<N,2>(arr[Ngrid-1], arr, Ngrid, &weights[d](Ngrid+1, 0));
    }
    // each row of the matrix corresponds to the value of source function at a given grid point,
    // or to its the second derivative at the endpoints of grid which is assumed to be zero
    // (i.e. natural cubic spline boundary condition);
    // each column corresponds to the weights of each element of amplitudes array,
    // which is formed as a product of non-zero 1d basis functions in three dimensions,
    // or their 2nd derivs at extra endpoint nodes
    for(unsigned int i=0; i<xnodes.size()+N-1; i++) {
        for(unsigned int j=0; j<ynodes.size()+N-1; j++) {
            for(unsigned int k=0; k<znodes.size()+N-1; k++) {
                unsigned int indRow = interp.indComp(i, j, k);
                for(int ti=0; ti<=N; ti++) {
                    for(int tj=0; tj<=N; tj++) {
                        for(int tk=0; tk<=N; tk++) {
                            unsigned int indCol = interp.indComp(
                                ti + leftInd[0][i], tj + leftInd[1][j], tk + leftInd[2][k]);
                            values.push_back(Triplet(indRow, indCol, 
                                weights[0](i, ti) * weights[1](j, tj) * weights[2](k, tk)));
                        }
                    }
                }
            }
        }
    }

    // solve the linear system (could take *LONG* )
    return LUDecomp(SpMatrix<double>(interp.numValues(), interp.numValues(), values)).solve(fncvalues);
}

template<int N>
std::vector<double> createInterpolator3dArrayFromSamples(
    const Matrix<double>& points, const std::vector<double>& pointWeights,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes)
{
    if(points.rows() != pointWeights.size() || points.cols() != 3)
        throw std::invalid_argument("createInterpolator3dArrayFromSamples: invalid size of input arrays");
    BsplineInterpolator3d<N> interp(xnodes, ynodes, znodes);

    std::vector<double> amplitudes(interp.numValues());
    // NOT IMPLEMENTED
    return amplitudes;
}

// force the template instantiations to compile
template class BsplineInterpolator3d<1>;
template class BsplineInterpolator3d<3>;

template std::vector<double> createInterpolator3dArray<1>(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);
template std::vector<double> createInterpolator3dArray<3>(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);
template std::vector<double> createInterpolator3dArrayFromSamples<1>(
    const Matrix<double>& points, const std::vector<double>& pointWeights,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);
template std::vector<double> createInterpolator3dArrayFromSamples<3>(
    const Matrix<double>& points, const std::vector<double>& pointWeights,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);


//-------------- PENALIZED SPLINE APPROXIMATION ---------------//

/// Implementation of penalized spline approximation
class SplineApproxImpl {
    const std::vector<double> knots;   ///< b-spline knots  X[k], k=0..numKnots-1
    const std::vector<double> xvalues; ///< x[i], i=0..numDataPoints-1
    const unsigned int numKnots;       ///< number of X[k] knots in the fitting spline;
    ///< the number of basis functions is  numBasisFnc = numKnots+2
    const unsigned int numDataPoints;  ///< number of x[i],y[i] pairs (original data)

    /// sparse matrix  C  containing the values of each basis function at each data point:
    /// (size: numDataPoints rows, numBasisFnc columns, with only 4 nonzero values in each row)
    SpMatrix<double> CMatrix;

    /// in the non-singular case, the matrix A = C^T C  of the system of normal equations is formed,
    /// and the lower triangular matrix L contains its Cholesky decomposition (size: numBasisFnc^2)
    Matrix<double> LMatrix;

    /// matrix "M" is the transformed version of roughness matrix R, which contains
    /// integrals of product of second derivatives of basis functions (size: numBasisFnc^2)
    Matrix<double> MMatrix;

    /// part of the decomposition of the matrix M (size: numBasisFnc)
    std::vector<double> singValues;

public:
    /// Auxiliary data used in the fitting process, pre-initialized for each set of data points `y`
    /// (these data cannot be members of the class, since they are not constant)
    struct FitData {
        std::vector<double> zRHS;  ///< C^T y, right hand side of the system of normal equations
        std::vector<double> MTz;   ///< the product  M^T z
        double ynorm2;             ///< the squared norm of vector y
    };

    /** Prepare internal tables for fitting the data points at the given set of x-coordinates
        and the given array of knots which determine the basis functions */
    SplineApproxImpl(const std::vector<double> &knots, const std::vector<double> &xvalues);

    /** find the weights of basis functions that provide the best fit to the data points `y`
        for the given value of smoothing parameter `lambda`, determined indirectly by EDF.
        \param[in]  yvalues  are the data values corresponding to x-coordinates
        that were provided to the constructor;
        \param[in]  EDF>=0  is the equivalent number of degrees of freedom (2<=EDF<=numBasisFnc);
        \param[out] weights  will contain the computed weights of basis functions;
        \param[out] RSS  will contain the residual sum of squared differences between data and appxox;
    */
    void solveForWeightsWithEDF(const std::vector<double> &yvalues, double EDF,
        std::vector<double> &weights, double &RSS) const;

    /** find the weights of basis functions that provide the best fit to the data points `y`
        with the Akaike information criterion (AIC) being offset by deltaAIC from its minimum value
        (the latter corresponding to the case of optimal smoothing).
        \param[in]  yvalues  are the data values;
        \param[in]  deltaAIC is the offset of AIC (0 means the optimally smoothed spline);
        \param[out] weights  will contain the computed weights of basis functions;
        \param[out] RSS,EDF  same as in the previous function;
    */
    void solveForWeightsWithAIC(const std::vector<double> &yvalues, double deltaAIC,
        std::vector<double> &weights, double &RSS, double &EDF) const;

    /** Obtain the best-fit solution for the given value of smoothing parameter lambda
        (this method is called repeatedly in the process of finding the optimal value of lambda).
        \param[in]  fitData contains the pre-initialized auxiliary arrays constructed by `initFit()`;
        \param[in]  lambda is the smoothing parameter;
        \param[out] weights  will contain the computed weights of basis functions;
        \param[out] RSS,EDF  same as in the previous function;
        \return  the value of AIC (Akaike information criterion) corresponding to these RSS and EDF
    */
    double computeWeights(const FitData &fitData, double lambda,
        std::vector<double> &weights, double &RSS, double &EDF) const;

private:
    /** Initialize temporary arrays used in the fitting process for the provided data vector y,
        in the case that the normal equations are not singular.
        \param[in]  yvalues is the vector of data values `y` at each data point;
        \returns    the data structure used by other methods later in the fitting process
    */
    FitData initFit(const std::vector<double> &yvalues) const;
};

namespace{
// compute the number of equivalent degrees of freedom
static double computeEDF(const std::vector<double>& singValues, double lambda)
{
    if(!isFinite(lambda))  // infinite smoothing leads to a straight line (2 d.o.f)
        return 2;
    else if(lambda==0)     // no smoothing means the number of d.o.f. equal to the number of basis fncs
        return singValues.size();
    else {
        double EDF = 0;
        for(unsigned int c=0; c<singValues.size(); c++)
            EDF += 1 / (1 + lambda * singValues[c]);
        return EDF;
    }
}
//-------- helper classes for root-finders -------//
class SplineEDFRootFinder: public IFunctionNoDeriv {
    const std::vector<double>& singValues;
    double targetEDF;
public:
    SplineEDFRootFinder(const std::vector<double>& _singValues, double _targetEDF) :
        singValues(_singValues), targetEDF(_targetEDF) {}
    virtual double value(double lambda) const {
        return computeEDF(singValues, lambda) - targetEDF;
    }
};

class SplineAICRootFinder: public IFunctionNoDeriv {
    const SplineApproxImpl& impl; ///< the fitting interface
    const SplineApproxImpl::FitData& fitData; ///< data for the fitting procedure
    const double targetAIC;       ///< target value of AIC for root-finder
public:
    SplineAICRootFinder(const SplineApproxImpl& _impl,
        const SplineApproxImpl::FitData& _fitData, double _targetAIC) :
        impl(_impl), fitData(_fitData), targetAIC(_targetAIC) {};
    virtual double value(double lambda) const {
        std::vector<double> weights;
        double RSS, EDF;
        double AIC = impl.computeWeights(fitData, lambda, weights, RSS, EDF);
        return AIC - targetAIC;
    }
};
}  // internal namespace

SplineApproxImpl::SplineApproxImpl(const std::vector<double> &_knots, const std::vector<double> &_xvalues) :
    knots(_knots), xvalues(_xvalues),
    numKnots(_knots.size()), numDataPoints(_xvalues.size())
{
    for(unsigned int k=1; k<numKnots; k++)
        if(knots[k]<=knots[k-1])
            throw std::invalid_argument("SplineApprox: knots must be in ascending order");

    // compute the roughness matrix R (integrals over products of second derivatives of basis functions)
    Matrix<double> RMatrix(computeOverlapMatrix<3,2>(knots));
    
    // initialize b-spline matrix C
    std::vector<Triplet> Cvalues;
    Cvalues.reserve(numDataPoints * 4);
    for(unsigned int i=0; i<numDataPoints; i++) {
        // for each input point, at most 4 basis functions are non-zero, starting from index 'ind'
        double B[4];
        unsigned int ind = bsplineWeightsExtrapolated<3>(xvalues[i], &knots.front(), numKnots, B);
        assert(ind<=numKnots-2);
        // store non-zero elements of the matrix
        for(int k=0; k<4; k++)
            Cvalues.push_back(Triplet(i, k+ind, B[k]));
    }
    CMatrix = SpMatrix<double>(numDataPoints, numKnots+2, Cvalues);
    Cvalues.clear();

    SpMatrix<double> SpA(numKnots+2, numKnots+2);  // temporary sparse matrix containing A = C^T C
    blas_dgemm(CblasTrans, CblasNoTrans, 1, CMatrix, CMatrix, 0, SpA);
    Matrix<double> AMatrix(SpA);

    // to prevent a failure of Cholesky decomposition in the case if A is not positive definite,
    // we add a small multiple of R to A (following the recommendation in Ruppert,Wand&Carroll)
    blas_daxpy(1e-10, RMatrix, AMatrix);
        
    // pre-compute matrix L which is the Cholesky decomposition of matrix of normal equations A
    CholeskyDecomp CholA(AMatrix);
    LMatrix = CholA.L();

    // transform the roughness matrix R into a more suitable form M+singValues:
    // obtain Q = L^{-1} R L^{-T}, where R is the roughness penalty matrix (replace R by Q)
    blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, LMatrix, RMatrix);
    blas_dtrsm(CblasRight, CblasLower,  CblasTrans, CblasNonUnit, 1, LMatrix, RMatrix);

    // decompose this Q via singular value decomposition: Q = U * diag(S) * V^T
    SVDecomp SVD(RMatrix);
    singValues = SVD.S();       // vector of singular values of matrix Q:
    singValues[numKnots] = 0;   // the smallest two singular values must be zero;
    singValues[numKnots+1] = 0; // set it explicitly to avoid roundoff error

    // precompute M = L^{-T} U  which is used in computing basis weight coefs.
    MMatrix = SVD.U();
    blas_dtrsm(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, 1, LMatrix, MMatrix);
    // now M is finally in place, and the weight coefs for any lambda are given by
    // w = M (I + lambda * diag(singValues))^{-1} M^T  z
}

// initialize the temporary arrays used in the fitting process
SplineApproxImpl::FitData SplineApproxImpl::initFit(const std::vector<double> &yvalues) const
{
    if(yvalues.size() != numDataPoints) 
        throw std::invalid_argument("SplineApprox: input array sizes do not match");
    FitData fitData;
    fitData.ynorm2  = blas_ddot(yvalues, yvalues);
    fitData.zRHS.resize(numKnots+2);
    fitData.MTz. resize(numKnots+2);
    blas_dgemv(CblasTrans, 1, CMatrix, yvalues, 0, fitData.zRHS);     // precompute z = C^T y
    blas_dgemv(CblasTrans, 1, MMatrix, fitData.zRHS, 0, fitData.MTz); // precompute M^T z
    return fitData;
}

// obtain solution of linear system for the given smoothing parameter,
// using the pre-computed matrix M^T z, where z = C^T y is the rhs of the system of normal equations;
// output the weights of basis functions and other relevant quantities (RSS, EDF); return AIC
double SplineApproxImpl::computeWeights(const FitData &fitData, double lambda,
    std::vector<double> &weights, double &RSS, double &EDF) const
{
    std::vector<double> tempv(numKnots+2);
    for(unsigned int p=0; p<numKnots+2; p++) {
        double sv = singValues[p];
        tempv[p]  = fitData.MTz[p] / (1 + (sv>0 ? sv*lambda : 0));
    }
    weights.resize(numKnots+2);
    blas_dgemv(CblasNoTrans, 1, MMatrix, tempv, 0, weights);
    // compute the residual sum of squares (note: may be prone to cancellation errors?)
    tempv = weights;
    blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, LMatrix, tempv); // tempv = L^T w
    double wTz = blas_ddot(weights, fitData.zRHS);
    RSS = (fitData.ynorm2 - 2*wTz + blas_ddot(tempv, tempv));
    EDF = computeEDF(singValues, lambda);  // equivalent degrees of freedom
    return log(RSS) + 2*EDF / (numDataPoints-EDF-1);  // AIC
}

void SplineApproxImpl::solveForWeightsWithEDF(const std::vector<double> &yvalues, double EDF,
    std::vector<double> &weights, double &RSS) const
{
    if(EDF==0)
        EDF = numKnots+2;
    else if(EDF<2 || EDF>numKnots+2)
        throw std::invalid_argument("SplineApprox: incorrect number of equivalent degrees of freedom");
    double lambda = findRoot(SplineEDFRootFinder(singValues, EDF), 0, INFINITY, 1e-6);
    computeWeights(initFit(yvalues), lambda, weights, RSS, EDF);
}

void SplineApproxImpl::solveForWeightsWithAIC(const std::vector<double> &yvalues, double deltaAIC,
    std::vector<double> &weights, double &RSS, double &EDF) const
{
    double lambda=0;
    FitData fitData = initFit(yvalues);
    if(deltaAIC < 0)
        throw std::invalid_argument("SplineApprox: deltaAIC must be non-negative");
    if(deltaAIC == 0) {  // find the value of lambda corresponding to the optimal fit
        lambda = findMin(SplineAICRootFinder(*this, fitData, 0),
            0, INFINITY, NAN /*no initial guess*/, 1e-6);
        if(lambda!=lambda)
            lambda = 0;  // no smoothing in case of weird problems
    } else {  // find an oversmoothed solution
        // the reference value of AIC at lambda=0 (NOT the value that minimizes AIC, but very close to it)
        double AIC0 = computeWeights(fitData, 0, weights, RSS, EDF);
        // find the value of lambda so that AIC is larger than the reference value by the required amount
        lambda = findRoot(SplineAICRootFinder(*this, fitData, AIC0 + deltaAIC),
            0, INFINITY, 1e-6);
        if(!isFinite(lambda))   // root does not exist, i.e. AIC is everywhere lower than target value
            lambda = INFINITY;  // basically means fitting with a linear regression
    }
    // compute the weights for the final value of lambda
    computeWeights(fitData, lambda, weights, RSS, EDF);
}

//----------- DRIVER CLASS FOR PENALIZED SPLINE APPROXIMATION ------------//

SplineApprox::SplineApprox(const std::vector<double> &grid, const std::vector<double> &xvalues)
{
    impl = new SplineApproxImpl(grid, xvalues);
}

SplineApprox::~SplineApprox()
{
    delete impl;
}

std::vector<double> SplineApprox::fit(
    const std::vector<double> &yvalues, const double edf,
    double *rms) const
{
    std::vector<double> weights;
    double RSS;
    impl->solveForWeightsWithEDF(yvalues, edf, weights, RSS);
    if(rms)
        *rms = sqrt(RSS / yvalues.size());
    return weights;
}

std::vector<double> SplineApprox::fitOversmooth(
    const std::vector<double> &yvalues, const double deltaAIC,
    double *rms, double* edf) const
{
    std::vector<double> weights;
    double RSS, EDF;
    impl->solveForWeightsWithAIC(yvalues, deltaAIC, weights, RSS, EDF);
    if(rms)
        *rms = sqrt(RSS / yvalues.size());
    if(edf)
        *edf = EDF;
    return weights;
}


//------------ LOG-SPLINE DENSITY ESTIMATOR ------------//
namespace {

/** Data for LogSplineDensity fitting procedure that is changing during the fit */
struct LogSplineFitParams {
    std::vector<double> init;   ///< array of amplitudes used to start the multidimensional minimizer
    std::vector<double> result; ///< array of amplitudes that correspond to the found minimum
    double lambda;              ///< smoothing parameter
    double targetLogL;          ///< target value of likelihood for the case with smoothing
    LogSplineFitParams() : lambda(0), targetLogL(0) {}
};

/** The engine of log-spline density estimator relies on the maximization of log-likelihood
    of input samples by varying the parameters of the estimator.

    Let  x_i, w_i; i=0..N_{data}-1  be the coordinates and weights of samples drawn from
    an unknown density distribution that we wish to estimate by constructing a function P(x).
    The total weight of all samples is M = \sum_{i=0}^{N_{data}-1} w_i  (does not need to be unity),
    and we stipulate that  \int P(x) dx = M.

    The logarithm of estimated density P(x) is represented as
    \ln P(x) = \sum_{k=0}^{N_{basis}-1}  A_k B_k(x) - \ln G_0 + \ln M = Q(x) - \ln G_0 + \ln M,
    where  A_k  are the amplitudes -- free parameters that are adjusted during the fit,
    B_k(x)  are basis functions (B-splines of degree N defined by grid nodes),
    Q(x) = \sum_k  A_k B_k(x)  is the weighted sum of basis function, and
    G_0  = \int \exp[Q(x)] dx  is the normalization constant determined from the condition
    that the integral of P(x) over the entire domain equals to M.
    If we shift the entire weighted sum of basis functions Q(x) up or down by a constant,
    this will have no effect on P(x), because this shift will be counterbalanced by G_0.
    Therefore, there is an extra gauge freedom of choosing {A_k};
    we elimitate it by fixing the amplitude of the last B-spline to zero: A_{N_{basis}-1} = 0.
    In the end, there are N_{ampl} = N_{basis}-1 free parameters that are adjusted during the fit.

    The total likelihood of the model given the amplitudes {A_k} is
    \ln L = \sum_{i=0}^{N_{data}-1}  w_i  \ln P(x_i)
          = \sum_{i=0}^{N_{data}-1}  w_i (\sum_{k=0}^{N_{ampl}-1} A_k B_k(x_i) - \ln G_0 + \ln M)
          = \sum_{k=0}^{N_{ampl}-1}  A_k  L_k  - M \ln G_0({A_k})  + M \ln M,
    where  L_k = \sum_{i=0}^{N_{data}-1} w_i B_k(x_i)  is an array of 'basis likelihoods'
    that is computed from input samples only once at the beginning of the fitting procedure.

    Additionally, we may impose a penalty for unsmoothness of the estimated P(x), by adding a term
    -\lambda \int (\ln P(x)'')^2 dx  into the above expression.
    Here lambda>=0 is the smoothing parameter, and the integral of squared second derivative
    of the sum of B-splines is expressed as a quadratic form in amplitudes:
    \sum_k \sum_l  A_k A_l R_{kl} ,  where R_{kl} = \int B_k''(x) B_l''(x) dx
    is the 'roughhness matrix', again pre-computed at the beginning of the fitting procedure.
    This addition decreases the overall likelihood, but makes the estimated ln P(x) more smooth.
    To find the suitable value of lambda, we use the following consideration:
    if P(x) were the true density distribution, then the likelihood of the finite number of samples
    is a random quantity with mean E and rms scatter D.
    We can tolerate the decrease in the likelihood by an amount comparable to D,
    if this makes the estimate more smooth.
    Therefore, we first find the values of amplitudes that maximize the likelihood without smoothing,
    and then determine the value of lambda that decreases log L by the prescribed amount.
    The mean and rms scatter of ln L are given (for equal-weight samples) by
    E   = \int P(x) \ln P(x) dx
        = M (G_1 + \ln M - \ln G_0),
    D   = \sqrt{ M \int P(x) [\ln P(x)]^2 dx  -  E^2 } / \sqrt{N_{data}}
        = M \sqrt{ G_2/G_0 - (G_1/G_0)^2 } / \sqrt{N_{data}} ,  where we defined
    G_d = \int \exp[Q(x)] [Q(x)]^d dx  for d=0,1,2.

    We minimize  -log(L)  by varying the amplitudes A_k (for a fixed value of lambda),
    using a nonlinear multidimensional root-finder with derivatives to locate the point
    where d log(L) / d A_k = 0 for all A_k.
    This class implements the interface needed for `findRootNdimDeriv()`: computation of
    gradient and hessian of log(L) w.r.t. each of the free parameters A_k, k=0..N_{ampl}-1.
*/
template<int N>
class LogSplineDensityFitter: public IFunctionNdimDeriv {
public:
    LogSplineDensityFitter(
        const std::vector<double>& xvalues, const std::vector<double>& weights,
        const std::vector<double>& grid, bool leftInfinite, bool rightInfinite,
        LogSplineFitParams& params);

    /** Return the array of properly normalized amplitudes, such that the integral of
        P(x) over the entire domain is equal to the sum of sample weights M.
    */
    std::vector<double> getNormalizedAmplitudes(const std::vector<double>& ampl) const;

    /** Compute the expected rms scatter in log-likelihood for a density function defined
        by the given amplitudes, for the given number of samples */
    double logLrms(const std::vector<double>& ampl) const;

    /** Compute the log-likelihood of the data given the amplitudes.
        \param[in]  ampl  is the array of amplitudes A_k, k=0..numAmpl-1.
        \return     \ln L = \sum_k A_k V_k - M \ln G_0({A_k}) + M \ln M ,  where
        G_0 is the normalization constant that depends on all A_k,
        V_k is the pre-computed array Vbasis.
    */
    double logL(const std::vector<double>& ampl) const;

    /** Compute the cross-validation likelihood of the data given the amplitudes.
        \param[in]  ampl  is the array of amplitudes.
        \return     \ln L_CV = \ln L - tr(H^{-1} B^T B) + (d \ln G_0 / d A) H^{-1} W.
    */
    double logLcv(const std::vector<double>& ampl) const;

private:
    /** Compute the gradient and hessian of the full log-likelihood function
        (including the roughness penalty) multiplied by a constant:
        \ln L_full = \ln L - \lambda \sum_k \sum_l A_k A_l R_{kl},
        where \ln L is defined in logL(),  R_{kl} is the pre-computed roughnessMatrix,
        and lambda is taken from an external LogSplineFitParams variable.
        This routine is used in the nonlinear root-finder to determine the values of A_k
        that correspond to grad=0.
        \param[in]  ampl  is the array of amplitudes A_k that are varied during the fit;
        \param[out] grad  = (-1/M) d \ln L / d A_k;
        \param[out] hess  = (-1/M) d^2 \ln L / d A_k d A_l.
    */
    virtual void evalDeriv(const double ampl[], double grad[], double hess[]) const;
    virtual unsigned int numVars() const { return numAmpl; }
    virtual unsigned int numValues() const { return numAmpl; }

    /** Compute the logarithm of G_d:
        \return  \ln(G_d) = \ln( \int \exp(Q(x)) [Q(x)]^d  dx ),  where
        Q(x) = \sum_{k=0}^{N_{ampl}-1}  A_k B_k(x)  is the weighted sum of basis functions,
        B_k(x) are basis functions (B-splines of degree N defined by the grid nodes),
        and the integral is taken over the finite or (semi-) infinite interval,
        depending on the boolean constants leftInfinite, rightInfinite
        (if any of them is false, the corresponding boundary is the left/right-most grid point,
        otherwise it is +-infinity).
        \param[in]  d     is the integer power index in the above expression.
        \param[in]  ampl  is the array of A_k.
        \param[out] deriv if not NULL, will contain the derivatives of  \ln(G_d) w.r.t. A_k;
        only implemented for d=0 which is used to compute the normalization constant.
        \param[out] deriv2 if not NULL, will contain the second derivatives:
        d^2 \ln G / d A_k d A_l, only implemented for d=0.
    */
    double logG(const int d, const double ampl[], double deriv[]=NULL, double deriv2[]=NULL) const;

    const std::vector<double> grid;   ///< grid nodes that define the B-splines
    const unsigned int numNodes;      ///< shortcut for grid.size()
    const unsigned int numBasisFnc;   ///< shortcut for the number of B-splines (numNodes+N-1)
    const unsigned int numAmpl;       ///< the number of amplitudes that may be varied (numBasisFnc-1)
    const unsigned int numData;       ///< number of sample points
    const bool leftInfinite, rightInfinite;  ///< whether the definition interval extends to +-inf
    static const int GL_ORDER = 10;          ///< order of GL quadrature for computing the normalization
    double GLnodes[GL_ORDER], GLweights[GL_ORDER];  ///< nodes and weights of GL quadrature
    std::vector<double> Vbasis;       ///< basis likelihoods: V_k = \sum_i w_i B_k(x_i)
    std::vector<double> Wbasis;       ///< W_k = \sum_i w_i^2 B_k(x_i) 
    Matrix<double> BTBmatrix;         ///< matrix B^T B, where B_{ik} = w_i B_k(x_i)
    double sumWeights;                ///< sum of weights of input points (M)
    Matrix<double> roughnessMatrix;   ///< roughness penalty matrix - integrals of B_k''(x) B_l''(x)
    LogSplineFitParams& params;       ///< external parameters that may be changed during the fit
};

template<int N>
LogSplineDensityFitter<N>::LogSplineDensityFitter(const std::vector<double>& _grid,
    const std::vector<double>& xvalues, const std::vector<double>& weights,
    bool _leftInfinite, bool _rightInfinite, LogSplineFitParams& _params) :
    grid(_grid), numNodes(grid.size()), numBasisFnc(numNodes + N - 1), numAmpl(numBasisFnc - 1),
    numData(xvalues.size()), leftInfinite(_leftInfinite), rightInfinite(_rightInfinite),
    params(_params)
{
    if(numData <= 0)
        throw std::invalid_argument("logSplineDensity: no data");
    if(numData != weights.size())
        throw std::invalid_argument("logSplineDensity: sizes of input arrays are not equal");
    if(numNodes<2)
        throw std::invalid_argument("logSplineDensity: grid size should be at least 2");
    for(unsigned int k=1; k<numNodes; k++)
        if(grid[k-1] >= grid[k])
            throw std::invalid_argument("logSplineDensity: grid nodes are not monotonic");
    prepareIntegrationTableGL(0, 1, GL_ORDER, GLnodes, GLweights);

    // prepare the roughness penalty matrix
    // (integrals over products of second or third derivatives of basis functions)
    roughnessMatrix = computeOverlapMatrix<N,3>(grid);

    // prepare the log-likelihoods of each basis fnc and the initial guess for amplitudes
    params.init.assign(numBasisFnc, 0);
    Vbasis.assign(numBasisFnc, 0);
    Wbasis.assign(numBasisFnc, 0);
    std::vector<Triplet> Bvalues;
    Bvalues.reserve(numData * (N+1));
    sumWeights = 0;
    double minWeight = INFINITY;
    for(unsigned int p=0; p<numData; p++) {
        if(weights[p] < 0)
            throw std::invalid_argument("LogSplineDensity: sample weights may not be negative");
        if(weights[p] == 0)
            continue;
        // if the interval is (semi-)finite, no samples should appear beyond its boundaries
        if( (xvalues[p] < grid[0] && !leftInfinite) ||
            (xvalues[p] > grid[numNodes-1] && !rightInfinite) )
            throw std::invalid_argument("LogSplineDensity: samples found outside the grid");
        double Bspl[N+1];
        int ind = bsplineWeightsExtrapolated<N>(xvalues[p], &grid[0], numNodes, Bspl);
        for(unsigned int b=0; b<=N; b++) {
            Vbasis[b+ind] += weights[p] * Bspl[b];
            Wbasis[b+ind] += pow_2(weights[p]) * Bspl[b];
            if(b+ind<numAmpl)
                Bvalues.push_back(Triplet(p, b+ind, weights[p] * Bspl[b]));
        }
        // add to the initial guess for amplitudes, if the point is inside the grid
        if(xvalues[p] >= grid[0] && xvalues[p] <= grid[numNodes-1])
            for(unsigned int b=0; b<=N; b++)
                params.init[b+ind] += weights[p] * Bspl[b];
        sumWeights += weights[p];
        minWeight = fmin(minWeight, weights[p]);
    }
    // sanity check    
    if(sumWeights==0)
        throw std::invalid_argument("LogSplineDensity: sum of sample weights should be positive");
    
    // sanity check: all of basis functions must have a contribution from sample points,
    // otherwise the max-likelihood solution is unattainable
    minWeight *= 0.1 / numNodes;
    for(int k=0; k<(int)numBasisFnc; k++) {
        if(Vbasis[k] == 0) {
            // make up a fake sample to maintain positive weights of this basis fnc
            int ind = std::min<int>(numNodes-2, std::max<int>(0, k-N/2));
            double Bspl[N+1];
            ind = bsplineWeights<N>((grid[ind] + grid[ind+1]) / 2, &grid[0], numNodes, Bspl);
            for(unsigned int b=0; b<=N; b++)
                Vbasis[b+ind] += minWeight * Bspl[b];
            sumWeights += minWeight;
        }
    }
    Vbasis.resize(numAmpl);
    Wbasis.resize(numAmpl);

    SpMatrix<double> Bmatrix(numData, numAmpl, Bvalues);
    SpMatrix<double> SpBTB(numAmpl, numAmpl);  // temporary sparse matrix containing B^T B
    blas_dgemm(CblasTrans, CblasNoTrans, 1, Bmatrix, Bmatrix, 0, SpBTB);
    BTBmatrix = Matrix<double>(SpBTB);
#if 0
    // compute the initial guess for amplitudes from histogrammed weights associated with each
    // basis function (plus a small addition to keep the argument of logarithm positive)
    for(unsigned int k=0; k<numBasisFnc; k++)
        params.init[k] = log(fmax(params.init[k], sumWeights/numData));

    // make sure that we start with a density that is declining when extrapolated
    if(leftInfinite)
        params.init[0] = fmin(params.init[0], params.init[1] - 0.1 * (grid[1]-grid[0]));
    if(rightInfinite)
        params.init[numBasisFnc-1] = fmin(params.init[numBasisFnc-1],
            params.init[numBasisFnc-2] - 0.1 * (grid[numBasisFnc-1]-grid[numBasisFnc-2]));

    // now shift all amplitudes by the value of the rightmost one, which is always kept equal to zero
    for(unsigned int k=0; k<numAmpl; k++)
        params.init[k] -= params.init.back();
    // and eliminate the last one, since it does not take part in the fitting process
    params.init.pop_back();
#else
    // somehow the initial guess above leads to instability in the root-finder...
    params.init.assign(numAmpl, 1.0);
    params.init[0] = 0.;
#endif
    params.result = params.init;  // allocate space for the results
}

template<int N>
std::vector<double> LogSplineDensityFitter<N>::getNormalizedAmplitudes(
    const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    std::vector<double> result(numBasisFnc);
    double C = log(sumWeights) - logG(0, &ampl[0]);
    for(unsigned int n=0; n<numBasisFnc; n++)
        result[n] = (n<numAmpl ? ampl[n] : 0) + C;
    return result;
}

template<int N>
double LogSplineDensityFitter<N>::logLrms(const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    double
    logG0 = logG(0, &ampl[0]),
    G1G0  = exp(logG(1, &ampl[0]) - logG0),
    G2G0  = exp(logG(2, &ampl[0]) - logG0),
    rms   = sumWeights * sqrt((G2G0 - pow_2(G1G0)) / numData);
#ifdef VERBOSE_REPORT
    double avg = sumWeights * (G1G0 + log(sumWeights) - logG0);
    std::cout << "Expected log L = " << avg << " +- " << rms << "\n";
#endif
    return rms;
}

template<int N>
double LogSplineDensityFitter<N>::logL(const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    double val = sumWeights * (log(sumWeights) - logG(0, &ampl[0]));
    for(unsigned int k=0; k<numAmpl; k++)
        val += Vbasis[k] * ampl[k];
    return val;
}

template<int N>
double LogSplineDensityFitter<N>::logLcv(const std::vector<double>& ampl) const
{
    assert(ampl.size() == numAmpl);
    std::vector<double> grad(numAmpl);
    Matrix<double> hess(numAmpl, numAmpl);
    double val = sumWeights * (log(sumWeights) - logG(0, &ampl[0], &grad[0], hess.data()));
    for(unsigned int k=0; k<numAmpl; k++) {
        val += Vbasis[k] * ampl[k];
        for(unsigned int l=0; l<numAmpl; l++) {
            hess(k, l) = sumWeights * hess(k, l) + 2 * params.lambda * roughnessMatrix(k, l);
        }
    }
    try{
        CholeskyDecomp hessdec(hess);
        Matrix<double> hessL(hessdec.L()), tmpmat(BTBmatrix);
        blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 1, hessL, tmpmat);
        blas_dtrsm(CblasLeft, CblasLower, CblasTrans,   CblasNonUnit, 1, hessL, tmpmat);
        for(unsigned int k=0; k<numAmpl; k++) {
            val -= tmpmat(k, k);  // trace of H^{-1} B^T B
        }
        val += blas_ddot(grad, hessdec.solve(Wbasis));  // dG/dA H^{-1} W
    }
    catch(std::exception&) {
#ifdef VERBOSE_REPORT
        std::cout << "Hessian is not positive-definite!\n";
#endif
    }
    return val;
}

template<int N>
void LogSplineDensityFitter<N>::evalDeriv(const double ampl[], double deriv[], double deriv2[]) const
{
    logG(0, ampl, deriv, deriv2);
    if(deriv!=NULL) {  // (-1/M)  d (log L) / d A_k
        for(unsigned int k=0; k<numAmpl; k++) {
            deriv[k] -= Vbasis[k] / sumWeights;
        }
    }
    // roughness penalty (taking into account symmetry and sparseness of the matrix)
    if(params.lambda!=0) {
        for(unsigned int k=0; k<numAmpl; k++) {
            for(unsigned int l=k; l<std::min(numAmpl, k+N+1); l++) {
                double v = 2 * roughnessMatrix(k, l) * params.lambda / sumWeights;
                if(deriv2!=NULL) {
                    deriv2[k * numAmpl + l] += v;
                    if(k!=l)
                        deriv2[l * numAmpl + k] += v;
                }
                if(deriv!=NULL) {
                    deriv[k] += v * ampl[l];
                    if(k!=l)
                        deriv[l] += v * ampl[k];
                }
            }
        }
    }
}

template<int N>
double LogSplineDensityFitter<N>::logG(
    const int d, const double ampl[], double deriv_arg[], double deriv2[]) const
{
    std::vector<double> deriv_tmp;
    double* deriv = deriv_arg;
    if(deriv_arg==NULL && deriv2!=NULL) {  // need a temporary workspace for the gradient vector
        deriv_tmp.resize(numAmpl);
        deriv = &deriv_tmp.front();
    }
    assert(d==0 || deriv==NULL);   // can only compute derivs for the d==0 case
    assert(d==0 || d==1 || d==2);
    // accumulator for the integral  G = \int \exp( Q(x) ) [Q(x)]^d  dx,
    // where  Q = \sum_k  A_k B_k(x).
    double integral = 0;
    // accumulator for d G / d A_k
    if(deriv) {
        for(unsigned int k=0; k<numAmpl; k++)
            deriv[k] = 0;
    }
    // accumulator for d^2 G / d A_k d A_l
    if(deriv2) {
        for(unsigned int kl=0; kl<pow_2(numAmpl); kl++)
            deriv2[kl] = 0;
    }
    // determine the constant offset needed to keep the magnitude in a reasonable range
    double offset = 0;
    for(unsigned int k=0; k<numAmpl; k++)
        offset = fmax(offset, ampl[k]);

    // loop over grid segments...
    for(unsigned int k=0; k<numNodes-1; k++) {
        double segwidth = grid[k+1] - grid[k];
        // ...and over sub-nodes of Gauss-Legendre quadrature rule within each grid segment
        for(int s=0; s<GL_ORDER; s++) {
            double x = grid[k] + GLnodes[s] * segwidth;
            double Bspl[N+1];
            // obtain the values of all nontrivial basis function at this point,
            // and the index of the first of these functions.
            int ind = bsplineWeights<N>(x, &grid[0], numNodes, Bspl);
            // sum the contributions to Q(x) from each basis function,
            // weighted with the provided amplitudes; 
            // here we substitute zero in place of the last (numBasisFnc-1)'th amplitude.
            double Q = 0;
            for(unsigned int b=0; b<=N && b+ind<numAmpl; b++) {
                Q += Bspl[b] * ampl[b+ind];
            }
            // the contribution of this point to the integral is weighted according to the GL quadrature;
            // the value of integrand is exp(Q) * Q^d,
            // but to avoid possible overflows, we instead compute  exp(Q-offset) Q^d.
            double val = GLweights[s] * segwidth * exp(Q-offset) * powInt(Q, d);
            integral += val;
            // contribution of this point to the integral of derivatives is further multiplied
            // by the value of each basis function at this point.
            if(deriv) {
                for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                    deriv[b+ind] += val * Bspl[b];
            }
            // and contribution to the integral of second derivatives is multiplied
            // by the product of two basis functions
            if(deriv2) {
                for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                    for(unsigned int c=0; c<=N && c+ind<numAmpl; c++)
                        deriv2[(b+ind) * numAmpl + c+ind] += val * Bspl[b] * Bspl[c];
            }
        }
    }

    // if the interval is (semi-)infinite, need to add contributions from the tails beyond the grid
    bool   infinite[2] = {leftInfinite, rightInfinite};
    double endpoint[2] = {grid[0], grid[numNodes-1]};
    double signder [2] = {+1, -1};
    for(int p=0; p<2; p++) {
        if(!infinite[p])
            continue;
        double Bspl[N+1], Bder[N+1];
        int ind = bsplineWeights<N>(endpoint[p], &grid[0], numNodes, Bspl);
        bsplineDerivs<N,1>(endpoint[p], &grid[0], numNodes, Bder);
        double Q = 0, Qder = 0;
        for(unsigned int b=0; b<=N && b+ind<numAmpl; b++) {
            Q    += Bspl[b] * ampl[b+ind];
            Qder += Bder[b] * ampl[b+ind];
        }
        if(signder[p] * Qder <= 0) {  
            // the extrapolated function rises as x-> -inf, so the integral does not exist
            if(deriv)
                deriv[0] = INFINITY;
            if(deriv2)
                deriv2[0] = INFINITY;
            return INFINITY;
        }
        double val = signder[p] * exp(Q-offset) / Qder;
        if(d==1) val *= Q-1;
        if(d==2) val *= pow_2(Q-1)+1;
        integral += val;
        if(deriv) {
            for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                deriv[b+ind] += val * (Bspl[b] - Bder[b] / Qder);
        }
        if(deriv2) {
            for(unsigned int b=0; b<=N && b+ind<numAmpl; b++)
                for(unsigned int c=0; c<=N && c+ind<numAmpl; c++)
                    deriv2[(b+ind) * numAmpl + c+ind] +=
                        val * ( Bspl[b] * Bspl[c] -
                        (Bspl[b] * Bder[c] + Bspl[c] * Bder[b]) / Qder +
                        2 * Bder[b] * Bder[c] / pow_2(Qder) );
        }
    }

    // output the log-derivative: d (ln G) / d A_k = (d G / d A_k) / G
    if(deriv) {
        for(unsigned int k=0; k<numAmpl; k++)
            deriv[k] /= integral;
    }
    // d^2 (ln G) / d A_k d A_l = d^2 G / d A_k d A_l - (d ln G / d A_k) (d ln G / d A_l)
    if(deriv2) {
        for(unsigned int kl=0; kl<pow_2(numAmpl); kl++)
            deriv2[kl] = deriv2[kl] / integral - deriv[kl / numAmpl] * deriv[kl % numAmpl];
    }
    // put back the offset in the logarithm of the computed value of G
    return log(integral) + offset;
}


/** Class for performing the search of the smoothing parameter lambda that
    yields the required value of log-likelihood, or maximizes the cross-validation score
*/
template<int N>
class LogSplineDensityLambdaFinder: public IFunctionNoDeriv {
public:
    LogSplineDensityLambdaFinder(const LogSplineDensityFitter<N>& _fitter, LogSplineFitParams& _params) :
        fitter(_fitter), params(_params) {}
private:
    virtual double value(const double scaledLambda) const
    {
        bool useCV = params.targetLogL==0;   // whether we are in the minimizer or root-finder mode
        params.lambda = exp( 1 / (1-scaledLambda) - 1 / scaledLambda );
        /*int numIter =*/ findRootNdimDeriv(fitter, &params.init[0], 1e-4, 100, &params.result[0]);
#ifdef VERBOSE_REPORT_BLAH
        double logLcv = fitter.logLcv(params.result);
        double logL = fitter.logL(params.result);
        std::cout << "lambda= " << params.lambda << "  #iter= " << numIter <<
        "  logL= " << logL << "  CV= " << logLcv << '\n';
        return useCV ? -logLcv : params.targetLogL - logL;
#else
        return useCV ? -fitter.logLcv(params.result) : params.targetLogL - fitter.logL(params.result);
#endif
    }
    const LogSplineDensityFitter<N>& fitter;
    LogSplineFitParams& params;
};
}  // internal namespace

template<int N>
std::vector<double> logSplineDensity(const std::vector<double> &grid,
    const std::vector<double> &xvalues, const std::vector<double> &weights,
    bool leftInfinite, bool rightInfinite, double smoothing)
{
    LogSplineFitParams params;
    const LogSplineDensityFitter<N> fitter(grid, xvalues, weights, leftInfinite, rightInfinite, params);
    findRootNdimDeriv(fitter, &params.init[0], 1e-6, 100, &params.result[0]);
    if(N>1 && smoothing>=0) {
        // start the search from the best-fit amplitudes for the case of no smoothing
        params.init = params.result;
        LogSplineDensityLambdaFinder<N> rootfinder(fitter, params);
        if(smoothing>0) {
            // target value of log-likelihood is allowed to be worse than
            // the best value for the case of no smoothing by an amount
            // that is proportional to the expected rms variation of logL
            params.targetLogL = fitter.logL(params.result) - smoothing * fitter.logLrms(params.result);
            params.lambda = findRoot(rootfinder, 0.0, 0.5, 1e-4);
        } else {
            // find the value of lambda that maximizes the cross-validation score
            params.lambda = findMin(rootfinder, 0.0, 0.5, NAN, 1e-4);
        }    
        // if something goes wrong, restore the initial amplitudes (corresponding to no smoothing)
        if(!isFinite(params.lambda))
            params.result = params.init;
    }
    return fitter.getNormalizedAmplitudes(params.result);
}

// force the template instantiations to compile
template std::vector<double> logSplineDensity<1>(
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, bool, bool, double);
template std::vector<double> logSplineDensity<3>(
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, bool, bool, double);

//------------ GENERATION OF UNEQUALLY SPACED GRIDS ------------//

std::vector<double> createUniformGrid(unsigned int nnodes, double xmin, double xmax)
{
    if(nnodes<2 || xmax<=xmin)
        throw std::invalid_argument("Invalid parameters for grid creation");
    std::vector<double> grid(nnodes);
    for(unsigned int k=1; k<nnodes-1; k++)
        grid[k] = (xmin * (nnodes-1-k) + xmax * k) / (nnodes-1);
    grid.front() = xmin;
    grid.back()  = xmax;
    return grid;
}

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
static void makegrid(std::vector<double>::iterator begin, std::vector<double>::iterator end,
    double startval, double endval)
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
    unsigned int minbin, unsigned int gridsize)
{
    if(srcpoints.size()==0)
        throw std::invalid_argument("Error in creating a grid: input points array is empty");
    gridsize = std::max<unsigned int>(2, std::min<unsigned int>(gridsize,
        static_cast<unsigned int>(srcpoints.size()/minbin)));
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
