/** \file    math_spline.h
    \brief   spline interpolation and penalized spline approximation
    \author  Eugene Vasiliev
    \date    2011-2016

1d cubic spline is based on the GSL implementation by G.Jungman;
2d cubic spline is based on interp2d library by D.Zaslavsky;
1d and 2d quintic splines are based on the code by W.Dehnen.
*/
#pragma once
#include "math_base.h"
#include "math_linalg.h"

namespace math{

///@{
/// \name One-dimensional interpolation

/** Class that defines a cubic spline with natural or clamped boundary conditions */
class CubicSpline: public IFunction, public IFunctionIntegral {
public:
    /** empty constructor is required for the class to be used in std::vector and alike places */
    CubicSpline() {};

    /** Initialize a cubic spline from the provided values of x and y
        (which should be arrays of equal length, and x values must be monotonically increasing).
        If deriv_left or deriv_right are provided, they set the slope at the lower or upper boundary
        (so-called clamped spline); if either of them is NaN, it means a natural boundary condition.
    */
    CubicSpline(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        double deriv_left=NAN, double deriv_right=NAN);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const;

    /** return the number of derivatives that the spline provides */
    virtual unsigned int numDerivs() const { return 2; }

    /** return the integral of spline function times x^n on the interval [x1..x2] */
    virtual double integrate(double x1, double x2, int n=0) const;

    /** return the integral of spline function times another function f(x) on the interval [x1..x2]; 
        the other function is specified by the interface that provides the integral
        of f(x) * x^n for 0<=n<=3 */
    double integrate(double x1, double x2, const IFunctionIntegral& f) const;

    /** return the lower end of definition interval */
    double xmin() const { return xval.size()? xval.front() : NAN; }

    /** return the upper end of definition interval */
    double xmax() const { return xval.size()? xval.back() : NAN; }

    /** check if the spline is initialized */
    bool isEmpty() const { return xval.size()==0; }

    /** check if the spline is everywhere monotonic on the given interval */
    bool isMonotonic() const;

    /** return the array of spline nodes */
    const std::vector<double>& xvalues() const { return xval; }

private:
    std::vector<double> xval;  ///< grid nodes
    std::vector<double> yval;  ///< values of function at grid nodes
    std::vector<double> cval;  ///< second derivatives of function at grid nodes
};


/** Class that defines a piecewise cubic Hermite spline.
    Input consists of values of y(x) and dy/dx on a grid of x-nodes;
    result is a cubic function in each segment, with continuous first derivative at nodes
    (however second or third derivative is not continuous, unlike the case of quintic spline).
*/
class HermiteSpline: public IFunction {
public:
    /** empty constructor is required for the class to be used in std::vector and alike places */
    HermiteSpline() {};

    /** Initialize the spline from the provided values of x, y(x) and y'(x)
        (which should be arrays of equal length, and x values must be monotonically increasing).
    */
    HermiteSpline(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
                  const std::vector<double>& yderivs);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const;

    virtual unsigned int numDerivs() const { return 2; }

    /** return the lower end of definition interval */
    double xmin() const { return xval.size()? xval.front() : NAN; }

    /** return the upper end of definition interval */
    double xmax() const { return xval.size()? xval.back() : NAN; }

    /** check if the spline is initialized */
    bool isEmpty() const { return xval.size()==0; }

    /** return the array of spline nodes */
    const std::vector<double>& xvalues() const { return xval; }

private:
    std::vector<double> xval;  ///< grid nodes
    std::vector<double> yval;  ///< values of function at grid nodes
    std::vector<double> yder;  ///< first derivatives of function at grid nodes
};


/** Class that defines a quintic spline.
    Given y and dy/dx on a grid, d^3y/dx^3 is computed such that the (unique) 
    polynomials of 5th order between two adjacent grid points that give y,dy/dx,
    and d^3y/dx^3 on the grid are continuous in d^2y/dx^2, i.e. give the same
    value at the grid points. At the grid boundaries  d^3y/dx^3=0  is adopted.
*/
class QuinticSpline: public IFunction {
public:
    /** empty constructor is required for the class to be used in std::vector and alike places */
    QuinticSpline() {};

    /** Initialize a quintic spline from the provided values of x, y(x) and y'(x)
        (which should be arrays of equal length, and x values must be monotonically increasing).
    */
    QuinticSpline(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const std::vector<double>& yderivs);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const;

    /** return the value of 3rd derivative at a given point */
    double deriv3(const double x) const;

    /** two derivatives are returned by evalDeriv() method, and third derivative - by deriv3() */
    virtual unsigned int numDerivs() const { return 3; }

    /** return the lower end of definition interval */
    double xmin() const { return xval.size()? xval.front() : NAN; }

    /** return the upper end of definition interval */
    double xmax() const { return xval.size()? xval.back() : NAN; }

    /** check if the spline is initialized */
    bool isEmpty() const { return xval.size()==0; }

    /** return the array of spline nodes */
    const std::vector<double>& xvalues() const { return xval; }

private:
    std::vector<double> xval;  ///< grid nodes
    std::vector<double> yval;  ///< values of function at grid nodes
    std::vector<double> yder;  ///< first derivatives of function at grid nodes
    std::vector<double> yder3; ///< third derivatives of function at grid nodes
};


///@}
/// \name Two-dimensional interpolation
///@{

/** Generic two-dimensional interpolator class */
class BaseInterpolator2d: public IFunctionNdim {
public:
    BaseInterpolator2d() {};
    /** Initialize a 2d interpolator from the provided values of x, y and z.
        The latter is 2d array with the following indexing convention:  z[i][j] = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
    */
    BaseInterpolator2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const Matrix<double>& zvalues);

    /** compute the value of the interpolating function and optionally its derivatives at point x,y;
        if the input location is outside the definition region, the result is NaN.
        Any combination of value, first and second derivatives is possible:
        if any of them is not needed, the corresponding pointer should be set to NULL.
    */
    virtual void evalDeriv(const double x, const double y,
        double* value=0, double* deriv_x=0, double* deriv_y=0,
        double* deriv_xx=0, double* deriv_xy=0, double* deriv_yy=0) const = 0;

    /** shortcut for computing the value of spline */
    double value(const double x, const double y) const {
        double v;
        evalDeriv(x, y, &v);
        return v;
    }

    /** IFunctionNdim interface */
    virtual void eval(const double vars[], double values[]) const {
        evalDeriv(vars[0], vars[1], values);
    }
    virtual unsigned int numVars() const { return 2; }    
    virtual unsigned int numValues() const { return 1; }

    /** return the boundaries of definition region */
    double xmin() const { return xval.size()? xval.front(): NAN; }
    double xmax() const { return xval.size()? xval.back() : NAN; }
    double ymin() const { return yval.size()? yval.front(): NAN; }
    double ymax() const { return yval.size()? yval.back() : NAN; }

    /** check if the interpolator is initialized */
    bool isEmpty() const { return xval.size()==0 || yval.size()==0; }

    /** return the array of grid nodes in x-coordinate */
    const std::vector<double>& xvalues() const { return xval; }

    /** return the array of grid nodes in y-coordinate */
    const std::vector<double>& yvalues() const { return yval; }

protected:
    std::vector<double> xval, yval;  ///< grid nodes in x and y directions
    Matrix<double> zval;             ///< flattened 2d array of z values
};


/** Two-dimensional bilinear interpolator */
class LinearInterpolator2d: public BaseInterpolator2d {
public:
    LinearInterpolator2d() : BaseInterpolator2d() {};

    /** Initialize a 2d interpolator from the provided values of x, y and z.
        The latter is 2d array with the following indexing convention:  z[i][j] = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
    */
    LinearInterpolator2d(const std::vector<double>& xgrid, const std::vector<double>& ygrid,
        const Matrix<double>& zvalues) : 
        BaseInterpolator2d(xgrid, ygrid, zvalues) {};

    /** Compute the value and/or derivatives of the interpolator;
        note that for the linear interpolator the 2nd derivatives are always zero. */
    virtual void evalDeriv(const double x, const double y,
        double* value=0, double* deriv_x=0, double* deriv_y=0,
        double* deriv_xx=0, double* deriv_xy=0, double* deriv_yy=0) const;
};


/** Two-dimensional cubic spline */
class CubicSpline2d: public BaseInterpolator2d {
public:
    CubicSpline2d() : BaseInterpolator2d() {};

    /** Initialize a 2d cubic spline from the provided values of x, y and z.
        The latter is 2d array (Matrix) with the following indexing convention:  z(i,j) = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
        Derivatives at the boundaries of definition region may be provided as optional arguments
        (currently a single value per entire side of the rectangle is supported);
        if any of them is NaN this means a natural boundary condition.
    */
    CubicSpline2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const Matrix<double>& zvalues,
        double deriv_xmin=NAN, double deriv_xmax=NAN, double deriv_ymin=NAN, double deriv_ymax=NAN);

    /** compute the value of spline and optionally its derivatives at point x,y */
    virtual void evalDeriv(const double x, const double y,
        double* value=0, double* deriv_x=0, double* deriv_y=0,
        double* deriv_xx=0, double* deriv_xy=0, double* deriv_yy=0) const;

private:
    /// flattened 2d arrays of derivatives in x and y directions, and mixed 2nd derivatives
    Matrix<double> zx, zy, zxy;
};


/** Two-dimensional quintic spline */
class QuinticSpline2d: public BaseInterpolator2d {
public:
    QuinticSpline2d() : BaseInterpolator2d() {};

    /** Initialize a 2d quintic spline from the provided values of x, y, z(x,y), dz/dx and dz/dy.
        The latter three are 2d arrays (variables of Matrix type) with the following indexing
        convention:  z(i,j) = f(x[i],y[j]), etc.
        Values of x and y arrays should monotonically increase.
    */
    QuinticSpline2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const Matrix<double>& zvalues, const Matrix<double>& dzdx, const Matrix<double>& dzdy);

    /** compute the value of spline and optionally its derivatives at point x,y */
    virtual void evalDeriv(const double x, const double y,
        double* value=0, double* deriv_x=0, double* deriv_y=0,
        double* deriv_xx=0, double* deriv_xy=0, double* deriv_yy=0) const;

private:
    Matrix<double> zx, zy, zxxx, zyyy, zxyy, zxxxyy;
};

///@}
/// \name Three-dimensional interpolation
///@{

/** Three-dimensional interpolator class: `f(x,y,z)` obtained by a tensor product
    of three 1d interpolaing kernels of order N>=1 that use N+1 grid points in each dimension.
    The interpolation is local: to compute the value at a given point, it uses only the values of
    the original function `v(x,y,z)` at (N+1)^3 nearby grid points, unlike 1d and 2d splines
    that are constructed globally.
    For N=1, the values of interpolated function `f` at grid nodes coincide with the original values `v`,
    but for N>1 this is not the case (like a Bezier curve does not pass through its control points).
*/
template<int N>
class BaseInterpolator3d: public math::IFunctionNdim {
public:
    BaseInterpolator3d() {};
    /** Initialize a 3d interpolator from the provided 1d arrays of grid nodes in x, y and z dimensions,
        and optionally the function values v at the nodes of this 3d grid.
        \param[in] xnodes, ynodes, znodes are the nodes of grid in each dimension,
        sorted in increasing order, must have at least N+1 elements;
        \param[in] fncvalues (optional) are the original function values `v(x,y,z)`
        with the following indexing convention:
        fncvalues[ (i*Ny + j) * Nz + k ] = v(x[i], y[j], z[k]), where Ny=ynodes.size(), Nz=znodes.size().
        If this array is not provided (or is empty), the function values remain uninitialized,
        but it is still possible to compute the weights of linear combination of these values
        for any point (x,y,z) within the grid definition regions, using the method `components`.
    */
    BaseInterpolator3d(const std::vector<double>& xnodes, const std::vector<double>& ynodes,
        const std::vector<double>& znodes, const std::vector<double>& fncvalues=std::vector<double>());

    /** Compute the weights of the linear combination of function values at grid points
        needed to evaluate the interpolated value `f` at the given location.
        \param[in]  vars is the 3d vector of coordinates on the grid;
        \param[out] leftIndices is the 3d array of indices of leftmost grid nodes in each of
        the three coordinates that are used for kernel interpolation (N+1 nodes in each dimension);
        \param[out] weights  is the array of (N+1)^3 weights that must be multiplied by the priginal
        function values `v` at grid nodes to compute the interpolated value, namely:
        \f$  f(x,y,z) = \sum_{i=0}^N \sum_{j=0}^N \sum_{k=0}^N  v(xn_{i+l[0]}, yn_{j+l[1]}, zn_{k+l[2]})
        \times  weights[(i*(N+1)+j)*(N+1)+k]  \f$,  where `l` is the shortcut for `leftIndices`,
        `xn`, `yn` and `zn` are the 1d arrays of grid nodes in each dimension,
        and `v` are the function values at these nodes.
        The sum of weights of all components is always 1, and weights are non-negative.
        The special case when one of these weigths is 1 and the rest are 0 occurs at the corners of
        the cube (the definition region), or, for a linear intepolator (N=1) also at all grid nodes,
        and means that the value of interpolated function `f` is equal to the original function value `v`
        at this node only; in all other points (even grid nodes for N>1) these values need not coincide.
        This method may be used even if the function values `v` at grid nodes were not provided in
        the constructor, in which case it is the responsibility of the user to carry out this summation.
        Otherwise one may use the `value()` method that performs this operation itself.
        If any of the coordinates falls outside grid boundaries in the respective dimension,
        the weights are NaN.
        \throw std::range_error if the grid nodes are empty
        (but uninitialized function values do not trigger an exception).
    */
    void components(const double vars[3], unsigned int leftIndices[3], double weights[]) const;

    /** Compute the value of the interpolating function `f` at point (x,y,z).
        \param[in] vars is the 3d vector of coordinates on the grid;
        \param[out] value will contain the interpolated function value at the given point.
        If the input location is outside the definition region, the result is NaN.
        Keep in mind that if the order of interpolator N>1, the weighted sum of (N+1)^3 components
        multiplied by the original function values at the nearby nodes may not equal the original value
        even if the point coincides with one of grid nodes (in other words, the interpolated curve is
        smoothing the original function); only in the case of trilinear interpolator (N=1) they are equal.
        \throw std::range_error if either grid nodes or the array of function values are not initialized.
    */
    virtual void eval(const double vars[3], double *value) const;

    /** shortcut for computing the value of interpolating function, with the same usage as `eval` */
    double value(const double x, const double y, const double z) const {
        double v, t[3]={x,y,z};
        eval(t, &v);
        return v;
    }

    // IFunctionNdim interface
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
    
    /** return the boundaries of definition region */
    double xmin() const { return xval.size()? xval.front(): NAN; }
    double xmax() const { return xval.size()? xval.back() : NAN; }
    double ymin() const { return yval.size()? yval.front(): NAN; }
    double ymax() const { return yval.size()? yval.back() : NAN; }
    double zmin() const { return zval.size()? zval.front(): NAN; }
    double zmax() const { return zval.size()? zval.back() : NAN; }

    /** check if the interpolator grid is initialized (the function values may remain uninitialized) */
    bool isEmpty() const { return xval.size()==0 || yval.size()==0 || zval.size()==0; }

    /** return the array of grid nodes in x-coordinate */
    const std::vector<double>& xvalues() const { return xval; }

    /** return the array of grid nodes in y-coordinate */
    const std::vector<double>& yvalues() const { return yval; }

    /** return the array of grid nodes in z-coordinate */
    const std::vector<double>& zvalues() const { return zval; }

    /** return the array of function values (may be empty if not provided at the constructor) */
    const std::vector<double>& fncvalues() const { return fncval; }
    
private:
    std::vector<double> xval, yval, zval;  ///< grid nodes in x, y and z directions
    std::vector<double> fncval;            ///< the values of function at the nodes of 3d grid, if provided
};

/// trilinear interpolator
typedef BaseInterpolator3d<1> LinearInterpolator3d;
/// tricubic interpolator
typedef BaseInterpolator3d<3> CubicInterpolator3d;

///@}
/// \name Penalized spline approximation (1d)
///@{

/// opaque internal data for SplineApprox
class SplineApproxImpl;

/** Penalized linear least-square fitting problem.
    Approximate the data series  {x[i], y[i], i=0..numDataPoints-1}
    with spline defined at  {X[k], Y[k], k=0..numKnots-1} in the least-square sense,
    possibly with additional penalty term for 'roughness' (curvature).

    Initialized once for a given set of x, X, and may be used to fit multiple sets of y
    with arbitrary degree of smoothing \f$ \lambda \f$.

    Internally, the approximation is performed by multi-parameter linear least-square fitting:
    minimize
      \f[
        \sum_{i=0}^{numDataPoints-1}  (y_i - \hat y(x_i))^2 + \lambda \int [\hat y''(x)]^2 dx,
      \f]
    where
      \f[
        \hat y(x) = \sum_{p=0}^{numBasisFnc-1} w_p B_p(x)
      \f]
    is the approximated regression for input data,
    \f$ B_p(x) \f$ are its basis functions and \f$ w_p \f$ are weights to be found.

    Basis functions are b-splines with knots at X[k], k=0..numKnots-1;
    the number of basis functions is numKnots+2. Equivalently, the regression
    can be represented by clamped cubic spline with numKnots control points;
    b-splines are only used internally.

    LLS fitting is done by solving the following linear system:
      \f$ (\mathsf{A} + \lambda \mathsf{R}) \mathbf{w} = \mathbf{z} \f$,
    where  A  and  R  are square matrices of size numBasisFnc,
    w and z are vectors of the same size, and \f$ \lambda \f$ is a scalar (smoothing parameter).

    A = C^T C, where C_ip is a matrix of size numDataPoints*numBasisFnc,
    containing value of p-th basis function at x[i], p=0..numBasisFnc-1, i=0..numDataPoints-1.
    z = C^T y, where y[i] is vector of original data points
    R is the roughness penalty matrix:  \f$ R_pq = \int B''_p(x) B''_q(x) dx \f$.

*/
class SplineApprox {
public: 
    /** initialize workspace for xvalues=x, knots=X in the above formulation.
        knots must be sorted in ascending order, and all xvalues must lie 
        between knots.front() and knots.back()   */
    SplineApprox(const std::vector<double> &xvalues, const std::vector<double> &knots);
    ~SplineApprox();

    /** check if the basis-function matrix A is singular: if this is the case, 
        fitting procedure is much slower and cannot accomodate any smoothing */
    bool isSingular() const;

    /** perform actual fitting for the array of y values with the given smoothing parameter.
        \param[in]  yvalues is the array of data points corresponding to x values
        that were passed to the constructor;
        \param[in]  lambda  is the smoothing parameter;
        \param[out] splineValues  are the values of smoothing spline function at grid nodes;
        \param[out] derivLeft, derivRight  are the derivatives of spline at the endpoints,
        which together with splineValues provide a complete description of the clamped cubic spline;
        \param[out] rmserror if not NULL, will contain the root-mean-square deviation of data points
        from the smoothing curve;
        \param[out] edf if not NULL, will contain the number of equivalend degrees of freedom,
        which decreases from numKnots+2 to 2 as the smoothing parameter increases from 0 to infinity.
    */
    void fitData(const std::vector<double> &yvalues, const double lambda, 
        std::vector<double>& splineValues, double& derivLeft, double& derivRight,
        double *rmserror=0, double* edf=0) const;

    /** perform fitting with adaptive choice of smoothing parameter lambda, to minimize
        the value of AIC (Akaike information criterion), defined as 
          log(rmserror^2 * numDataPoints) + 2 * EDF / (numDataPoints-EDF-1) .
        The input and output arguments are similar to `fitData()`, with the difference that
        the smoothing parameter lambda is not provided as input, but may be reported as output
        parameter `lambda` if the latter is not NULL.
    */
    void fitDataOptimal(const std::vector<double> &yvalues, 
        std::vector<double>& splineValues, double& derivLeft, double& derivRight,
        double *rmserror=0, double* edf=0, double *lambda=0) const;

    /** perform an 'oversmooth' fitting with adaptive choice of smoothing parameter lambda.
        deltaAIC>=0 determines the difference in AIC (Akaike information criterion) between
        the solution with no smoothing and the returned solution which is smoothed more than
        the optimal amount defined above.
        The other arguments have the same meaning as in `fitDataOptimal()`.
    */
    void fitDataOversmooth(const std::vector<double> &yvalues, const double deltaAIC, 
        std::vector<double>& splineValues, double& derivLeft, double& derivRight,
        double *rmserror=0, double* edf=0, double *lambda=0) const;

private:
    const SplineApproxImpl* impl;   ///< internal object hiding the implementation details
    SplineApprox& operator= (const SplineApprox&);  ///< assignment operator forbidden
    SplineApprox(const SplineApprox&);              ///< copy constructor forbidden
};

///@}
/// \name Auxiliary routines for grid generation
///@{

/** generate a grid with uniformly spaced nodes.
    \param[in]  nnodes   is the total number of grid points (>=2);
    \param[in]  xmin     is the location of the innermost node (>0);
    \param[in]  xmax     is the location of the outermost node (should be >xmin);
    \return     the array of grid nodes.
*/
std::vector<double> createUniformGrid(unsigned int nnodes, double xmin, double xmax);

/** generate a grid with exponentially spaced nodes, i.e., uniform in log(x):
    log(x[k]) = log(xmin) + log(xmax/xmin) * k/(nnodes-1), k=0..nnodes-1.
    \param[in]  nnodes   is the total number of grid points (>=2);
    \param[in]  xmin     is the location of the innermost node (>0);
    \param[in]  xmax     is the location of the outermost node (should be >xmin);
    \return     the array of grid nodes.
*/
std::vector<double> createExpGrid(unsigned int nnodes, double xmin, double xmax);

/** generate a grid with exponentially growing spacing:
    x[k] = (exp(Z k) - 1)/(exp(Z) - 1), i.e., coordinates of nodes increase nearly linearly
    at the beginning and then nearly exponentially towards the end;
    the value of Z is computed so the the 1st element is at xmin and last at xmax.
    \param[in]  nnodes   is the total number of grid points (>=2);
    \param[in]  xmin     is the location of the innermost nonzero node (>0);
    \param[in]  xmax     is the location of the last node (should be >=nnodes*xmin);
    \param[in]  zeroelem -- if true, 0th node in the output array is placed at zero (otherwise at xmin);
    \return     the array of grid nodes.
*/
std::vector<double> createNonuniformGrid(unsigned int nnodes, double xmin, double xmax, bool zeroelem);

/** create an almost uniform grid so that each bin contains at least minbin points from input array.
    Input points are in srcpoints array and MUST BE SORTED in ascending order (assumed but not cheched).
    \param[in]  srcpoints is the input array of points;
    \param[in]  minbin    is the minimum number of points per bin;
    \param[in]  gridsize  is the required length of the output array;
    \return     the array of grid nodes. 
    NB: in the present implementation, the algorithm is not very robust 
    and works well only for gridsize*minbin << srcpoints.size, assuming that 
    'problematic' bins only are found close to endpoints but not in the middle of the grid.
*/
std::vector<double> createAlmostUniformGrid(const std::vector<double> &srcpoints,
    unsigned int minbin, unsigned int& gridsize);

/** extend the input grid to negative values, by reflecting it about origin.
    \param[in]  input is the vector of N values that should start at zero
                and be monotonically increasing;
    \return     a new vector that has 2*N-1 elements, so that
                input[i] = output[N-1+i] = -output[N-1-i] for 0<=i<N
    \throw      std::invalid_argument if the input does not start with zero or is not increasing.
*/
std::vector<double> mirrorGrid(const std::vector<double> &input);

///@}
}  // namespace
