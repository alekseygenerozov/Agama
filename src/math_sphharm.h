/** \file   math_sphharm.h
    \brief  Legendre polynomials and spherical-harmonic transformations
    \date   2015-2016
    \author Eugene Vasiliev
*/
#pragma once
#include <vector>

namespace math {

// -------- old api -------- //

/** Associated Legendre polynomial (or, rather, function) of the first kind:
    \f$  P_l^m(x)  \f$.
    These functions are used in spherical-harmonic expansion as follows: 
    \f$  Y_l^m = \sqrt{\frac{ (2l+1) (l-m)! }{ 4\pi (l+m)! }}
         P_l^m(\cos(\theta)) * \{\sin,\cos\}(m\phi)  \f$
    (this routine returns un-normalized P).
*/
double legendrePoly(const int l, const int m, const double x);

//---------- new api -----------//

/** Array of normalized associate Legendre polynomials W and their derivatives for l=m..lmax
    (theta-dependent factors in spherical-harmonic expansion):
    \f$  Y_l^m(\theta, \phi) = W_l^m(\theta) \{\sin,\cos\}(m\phi) ,
         W_l^m = \sqrt{\frac{ (2l+1) (l-m)! }{ 4\pi (l+m)! }} P_l^m(\cos(\theta))  \f$,
    where P are un-normalized associated Legendre functions.
    The output arrays contain values of W, dW/dtheta, d^2W/dtheta^2  for l=m,m+1,...,lmax;
    if either deriv_array or deriv2_array = NULL, the corresponding thing is not computed
    (note that if deriv2_array is not NULL, deriv_array must not be NULL too).
    This routine differs from `legendrePolyArray` in the following:
    (1) it takes theta rather than cos(theta) as argument;
    (2) returns normalized functions directly suitable for Y_l^m;
    (3) returns derivatives w.r.t. theta, not cos(theta), and may compute the 2nd derivative;
    (4) accurately handles values of theta close to 0 or pi.
*/
void sphHarmonicArray(const int lmax, const int m, const double theta,
    double* resultArray, double* derivArray=0, double* deriv2Array=0);

/** Indexing scheme for spherical-harmonic transformation.
    It defines the maximum order of expansion in theta (lmax) and phi (mmax),
    that should satisfy 0 <= mmax <= lmax (0 means using only one term),
    and also defines which coefficients to skip (they are assumed to be identically zero
    due to particular symmetries of the function), specified by lstep, mmin and mstep.
    This scheme is used in the forward and inverse SH transformation routines.
*/
class SphHarmIndices {
public:
    SphHarmIndices(int _lmax, int _lstep, int _mmin, int _mmax, int _mstep);
    const int
    lmax,  ///< order of expansion in theta (>=0)
    lstep, ///< 1 if all coefs in theta are used, 2 if only even-l are used
    mmin,  ///< 0 if only cosines are used, -mmax if both sine and cosine coefs are used
    mmax,  ///< order of expansion in phi (0<=mmax<=lmax)
    mstep; ///< 1 if all coefs in phi are used, 2 if only even-m coefs are used

    /// number of elements in the array of spherical-harmonic coefficients
    unsigned int size() const { return (lmax+1)*(lmax+1); }

    /// index of coefficient with the given l and m
    /// (0<=l<=lmax, -l<=m<=l, no range check performed!)
    static unsigned int index(int l, int m) { return l*(l+1)+m; }
    
    /// decode the l-index from the combined index of a coefficient
    static int index_l(unsigned int c);
    
    /// decode the m-index from the combined index of a coefficient
    static int index_m(unsigned int c); 
};

/** Class for performing forward spherical-harmonic transformation.
    For the given coefficient indexing scheme, specified by an instance of `SphHarmIndices`,
    it computes the S-H coefficients C_lm from the values of function f(theta,phi)
    at nodes of a 2d grid in theta and phi.
    The workflow is the following:
      - create the instance of forward transform class for the given indexing scheme;
      - for each function f(theta,phi) the user should collect its values at the nodes of grid
        specified by member functions `costheta(j)` and `phi(k)` and store in a 2d array
        with length `size()`, using the following loop:
        \code
        SphHarmTransformForward trans(ind);
        std::vector<double> input_values(trans.size());
        for(int j=0; j<=ind.lmax; j+=ind.lstep)
            for(int k=ind.mmin; k<=ind.mmax; k++)
                input_values[trans.index(j, k)] = my_function(trans.costheta(j), trans.phi(k));
        std::vector<double> output_coefs(ind.size());
        trans.transform(&values.front(), &output_coefs.front());
        \endcode
    Depending on the symmetry properties specified by the indexing scheme, not all elements
    of input_values need to be filled by the user; the transform routine takes this into account.
    The implementation uses 'naive' summation approach without any FFT or fast Legendre transform
    algorithms, has complexity O(lmax^2*mmax) and is only suitable for lmax <~ few dozen.
    The transformation is 'lossless' (to machine precision) if the original function is
    band-limited, i.e. given by a sum of spherical harmonics with order up to lmax and mmax.
*/
class SphHarmTransformForward {
public:
    /// initialize the grid in theta and the table of transformation coefficients
    SphHarmTransformForward(const SphHarmIndices& ind);

    /// return the required size of input array for the forward transformation
    unsigned int size() const { return (ind.lmax+1) * (2*ind.mmax+1); }

    /// return the index of element in the input array for the given indices
    /// j (for theta, 0<=j<=ind.lmax) and k (for phi, ind.mmin<=k<=ind.mmax)
    unsigned int index(int j, int k) const;

    /// return the coordinate of j-th node for theta on (0:pi), 0 <= j <= ind.lmax
    double theta(int j) const;
    
    /// return the coordinate of k-th node for phi on (-pi:pi), ind.mmin <= k <= ind.mmax
    double phi(int k) const;

    /** perform the transformation of input array (values) into the array of coefficients.
        \param[in]  values is the array of function values at a rectangular grid in (theta,phi):
        cos(theta_j) and phi_k are given by the member functions `costheta(j)` and `phi(k)`;
        the input array of length `size()` must be arranged so that
        values[index(j,k)] = f(theta_j, phi_k).
        \param[out] coefs must point to an existing array of length `ind.size()`,
        which will be filled with spherical-harmonic expansion coefficients as follows:
        coefs[ind.index(l,m)] = C_lm, 0 <= l <= lmax, -l <= m <= l.
    */
    void transform(const double values[] /*in*/, double coefs[] /*out*/) const;

    /// coefficient indexing scheme (including lmax and mmax)
    const SphHarmIndices ind;

private:
    const int 
    nnodth,  ///< number of nodes in Gauss-Legendre grid in cos(theta)
    nlegfn,  ///< number of Legendre functions for each theta-node
    nnodphi, ///< number of nodes in uniform grid in phi
    ntrigfn; ///< number of trig functions for each phi-node
    std::vector<double>
    thnodes, ///< coordinates of the grid nodes in theta on (0:pi)
    legFnc,  ///< values of all associated Legendre functions of order <= lmax,mmax at nodes of theta-grid
    trigFnc; ///< values of sines and cosines at nodes of phi-grid

    /// index of Legendre function P_{lm}(theta_j) in the `legFnc` array
    unsigned int indLeg (int j, int l, int m) const;
    /// index of trig function trig(m phi_k) in the `trigFnc` array
    unsigned int indTrig(int k, int m) const;
    /// index of Fourier coefficient F_m(theta_j) in the intermediate coef array (in transform)
    unsigned int indFour(int j, int m) const;
};

/** Routine for performing inverse spherical-harmonic transformation.
    Given the array of coefficients obtained by the forward transformation,
    it computes the value of function at the given position on unit sphere (theta,phi).
    In doing so, it requires temporary storage that must be provided by the calling code.
    \param[in]  ind   - coefficient indexing scheme, defining lmax, mmax and skipped coefs;
    \param[in]  coefs - the array of coefficients;
    \param[in]  theta - the polar angle;
    \param[in]  phi   - the azimuthal angle;
    \param[in,out] tmp  is a temporary storage with size (at least) ind.lmax+2*ind.mmax+1;
    \returns    the value of function at (theta,phi)
*/
double sphHarmTransformInverse(const SphHarmIndices& ind, const double coefs[],
    const double theta, const double phi, double* tmp);

/** zero out array elements with magnitude smaller than the threshold
    (relative to the L1-norm of the array) */
void eliminateNearZeros(std::vector<double>& vec, double threshold=1e-12);

}  // namespace math