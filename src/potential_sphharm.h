/** \file    potential_sphharm.h
    \brief   potential approximations based on spherical-harmonic expansion
    \author  Eugene Vasiliev
    \date    2010-2015
**/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_spline.h"
#include "math_sphharm.h"
#include "smart.h"

namespace potential {

// ------ old api ------ //

class SphericalHarmonicCoefSet {
public:
    SphericalHarmonicCoefSet(unsigned int _Ncoefs_angular) :
        Ncoefs_angular(_Ncoefs_angular) {}
    unsigned int getNumCoefsAngular() const { return Ncoefs_angular; }

protected:
    coord::SymmetryType mysymmetry;      ///< specifies the type of symmetry
    unsigned int Ncoefs_angular;         ///< l_max, the order of angular expansion (0 means spherically symmetric model)
    int lmax, lstep, mmin, mmax, mstep;  ///< range of angular coefficients used for given symmetry

    /// assigns the range for angular coefficients based on mysymmetry
    void setSymmetry(coord::SymmetryType sym);

};  // class SphericalHarmonicCoefSet


/** parent class for all potential expansions based on spherical harmonics for angular variables **/
class BasePotentialSphericalHarmonic: public BasePotentialSph, public SphericalHarmonicCoefSet {
public:
    BasePotentialSphericalHarmonic(unsigned int _Ncoefs_angular) :
        BasePotentialSph(), SphericalHarmonicCoefSet(_Ncoefs_angular) {}

protected:
    /** The function that computes spherical-harmonic coefficients for potential 
        and its radial (first/second) derivative at given radius.
        Must be implemented in derived classes, and is used in evaluation of potential and forces; 
        unnecessary coefficients are indicated by passing NULL for coefs** and should not be computed.
    */
    virtual void computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const = 0;

private:
    /** Calculate the potential and its derivatives, using the spherical-harmonic
        expansion coefficients and their derivatives returned by computeSHCoefs(). */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;

    virtual coord::SymmetryType symmetry() const { return mysymmetry; }

};  // class BasePotentialSphericalHarmonic


/** basis-set expansion on the Zhao(1996) basis set (alpha models) **/
class BasisSetExp: public BasePotentialSphericalHarmonic {
public:
    /// init potential from an analytic mass model
    BasisSetExp(double _Alpha, unsigned int numCoefsRadial, unsigned int numCoefsAngular, 
        const BaseDensity& density);

    /// init potential from a discrete point mass set
    BasisSetExp(double _Alpha, unsigned int numCoefsRadial, unsigned int numCoefsAngular, 
        const particles::PointMassArray<coord::PosSph> &points, coord::SymmetryType sym=coord::ST_TRIAXIAL);

    /// init potential from stored coefficients
    BasisSetExp(double _Alpha, const std::vector< std::vector<double> > &coefs);

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "BasisSetExp"; };

    //  get functions:
    /// return the array of BSE coefficients
    void getCoefs(std::vector< std::vector<double> > &coefsArray) const
    { coefsArray=SHcoefs; };

    /// return the shape parameter of basis set
    double getAlpha() const { return Alpha; };

    /// return the number of radial basis functions
    unsigned int getNumCoefsRadial() const { return Ncoefs_radial; }

    /// a faster estimate of M(r) from the l=0 harmonic only
    virtual double enclosedMass(const double radius) const;

private:
    unsigned int Ncoefs_radial;                 ///< number of radial basis functions [ =SHcoefs.size() ]
    std::vector<std::vector<double> > SHcoefs;  ///< array of coefficients A_nlm of potential expansion
    double Alpha;                               ///< shape parameter controlling inner and outer slopes of basis functions

    /// compute angular expansion coefs at the given radius
    virtual void computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const;

    /// compute coefficients from a discrete point mass set; 
    /// if Alpha=0 then it is computed automatically from the data
    void prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph>& points);

    /// compute coefficients from a smooth mass profile; 
    /// if Alpha=0 then it is chosen automatically from density->getGamma()
    void prepareCoefsAnalytic(const BaseDensity& density);

    /// assigns symmetry class if some coefficients are (near-)zero
    void checkSymmetry();

};  // class BasisSetExp


/** spherical-harmonic expansion of potential with coefficients being spline functions of radius **/
class SplineExp: public BasePotentialSphericalHarmonic
{
public:
    /** init potential from analytic mass model:
        \param[in] numCoefsRadial  is the number of grid nodes in radius (excluding the one at r=0);
        \param[in] numCoefsAngular  is the maximum order of angular spherical-harmonic expansion;
        \param[in] density  is the input density profile;
        \param[in] Rmin is the radius of the innermost grid node (0 means auto-detect);
        \param[in] Rmax is the radius of the outermost grid node (0 means auto-detect);
    */
    SplineExp(unsigned int numCoefsRadial, unsigned int numCoefsAngular, 
        const BaseDensity& density, double Rmin=0, double Rmax=0);

    /** init potential from an array of N point masses:
        \param[in] numCoefsRadial  is the number of grid nodes in radius (excluding the one at r=0);
        \param[in] numCoefsAngular  is the maximum order of angular spherical-harmonic expansion;
        \param[in] points  is the array of point masses;
        \param[in] sym  is the assumed symmetry of the model;
        \param[in] smoothFactor  is the amount of smoothing applied to the l>0 terms in the expansion;
        \param[in] Rmin is the radius of the innermost grid node (0 means auto-detect);
        \param[in] Rmax is the radius of the outermost grid node (0 means auto-detect);
    */
    SplineExp(unsigned int numCoefsRadial, unsigned int numCoefsAngular, 
        const particles::PointMassArray<coord::PosSph> &points, 
        coord::SymmetryType sym=coord::ST_TRIAXIAL, double smoothFactor=0, 
        double Rmin=0, double Rmax=0);

    /// init potential from stored spherical-harmonic coefficients at given radii
    SplineExp(const std::vector<double> &_gridradii,
        const std::vector< std::vector<double> > &_coefs);

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "SplineExp"; };

    // get functions
    /// return the number of radial points in the spline (excluding r=0)
    unsigned int getNumCoefsRadial() const { return Ncoefs_radial; }

    /** return the array of spherical-harmonic expansion coefficients.
        \param[out] radii  will contain the radii of grid nodes;
        \param[out] coefsArray  will contain the spherical-harmonic 
        expansion coefficients at the given radii.
    */
    void getCoefs(std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray) const;

    /// a faster estimate of M(r) from the l=0 harmonic only
    virtual double enclosedMass(const double radius) const;

private:
    unsigned int Ncoefs_radial;              ///< number of radial coefficients (excluding the one at r=0)
    std::vector<double> gridradii;           ///< defines nodes of radial grid in splines
    double minr, maxr;                       ///< definition range of splines; extrapolation beyond this radius 
    double ascale;                           ///< value of scaling radius for non-spherical expansion coefficients which are tabulated as functions of log(r+ascale)
    double gammain,  coefin;                 ///< slope and coef. for extrapolating potential inside minr (spherically-symmetric part, l=0)
    double gammaout, coefout, der2out;       ///< slope and coef. for extrapolating potential outside maxr (spherically-symmetric part, l=0)
    double potcenter, potmax, potminr;       ///< (abs.value) potential in the center (for transformation of l=0 spline), at the outermost spline node, and at the innermost spline node
    std::vector<math::CubicSpline> splines;  ///< spline coefficients at each harmonic
    std::vector<double> slopein, slopeout;   ///< slope of coefs for l>0 for extrapolating inside rmin/outside rmax

    /// reimplemented function to compute the coefficients of angular expansion of potential at the given radius
    virtual void computeSHCoefs(const double r, double coefsF[], double coefsdFdr[], double coefsd2Fdr2[]) const;

    /// assigns symmetry class if some coefficients are (near-)zero.
    /// called on an intermediate step after computing coefs but before initializing splines
    void checkSymmetry(const std::vector< std::vector<double> > &coefsArray);

    /// create spline objects for all non-zero spherical harmonics from the supplied radii and coefficients.
    /// radii should have "Ncoefs_radial" elements and coefsArray - "Ncoefs_radial * (Ncoefs_angular+1)^2" elements
    void initSpline(const std::vector<double>& radii, const std::vector<std::vector<double> >& coefsArray);

    /** calculate (non-smoothed) spherical harmonic coefs for a discrete point mass set.
        \param[in]  points contains the positions and masses of particles that serve as the input.
        \param[out] outradii will contain radii of particles sorted in ascending order,
        \param[out] outcoefs will contain coefficients of SH expansion for all particles
        in a 'swapped' order, namely: the size of this array is numCoefs^2,
        and elements of this array are `outcoefs[coefIndex][particleIndex]`
        this is done to save memory by not allocating arrays for unused coefficients.
    */
    void computeCoefsFromPoints(const particles::PointMassArray<coord::PosSph>& points, 
        std::vector<double>& outradii, std::vector<std::vector<double> >& outcoefs);

    /** create smoothing splines from the coefficients computed at each particle's radius.
        \param[in] points is the array of particle coordinates and masses
        \param[in] smoothfactor determines how much smoothing is applied to the spline 
                   (good results are obtained for smoothfactor=1-2). 
        \param[in] Rmin,Rmax are the innermost/outermost grid radii;
                   if either is 0 then it is assigned automatically 
                   taking into account the range of radii covered by source points.
    */
    void prepareCoefsDiscrete(const particles::PointMassArray<coord::PosSph>& points, 
        double smoothfactor, double Rmin, double Rmax);

    /** compute expansion coefficients from an analytical mass profile.
        \param[in] density is the density profile to be approximated
        \param[in] Rmin,Rmax are the innermost/outermost grid radii;
                   if either is 0 then it is assigned automatically 
                   taking into account the range of radii covered by source points.
    */
    void prepareCoefsAnalytic(const BaseDensity& density, double Rmin, double Rmax);

    /// evaluate value and optionally up to two derivatives of l=0 coefficient, 
    /// taking into account extrapolation beyond the grid definition range and log-scaling of splines
    void coef0(double r, double *val, double *der, double *der2) const;

    /// evaluate value, and optionally first and second derivative of l>0 coefficients 
    /// (lm is the combined index of angular harmonic >0); corresponding values for 0th coef must be known
    void coeflm(unsigned int lm, double r, double xi, double *val, double *der, double *der2, 
        double c0val, double c0der=0, double c0der2=0) const;

};  // class SplineExp


// --------- new api --------- //


/** Compute spherical-harmonic density expansion coefficients at the given radii.
    First it collects the values of density at a 3d grid in radii and angles,
    then applies sph.-harm. transform at each radius.
    The first step is OpenMP-parallelized, so that it may be efficiently used
    for an input density profile that is expensive to compute.
    \param[in]  dens - the input density profile.
    \param[in]  ind  - indexing scheme for spherical-harmonic coefficients,
                which determines the order of expansion and its symmetry properties.
    \param[in]  gridRadii - the array of radial points for the output coefficients;
                must form an increasing sequence and start from r>0.
    \param[out] coefs - the array of sph.-harm. coefficients:
                coefs[k][c] is the value of c-th coefficient (where c is a single index 
                combining both l and m) at the radius r_k; will be resized as needed.
    \throws std::invalid_argument if gridRadii are not correct.
*/
void computeDensityCoefsSph(const BaseDensity& dens, 
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector<double> > &coefs);

/** Compute spherical-harmonic potential expansion coefficients,
    by first creating a sph.-harm.representation of the density profile,
    and then solving the Poisson equation.
    \param[in]  dens - the input density profile.
    \param[in]  ind  - indexing scheme for spherical-harmonic coefficients,
                which determines the order of expansion and its symmetry properties.
    \param[in]  gridRadii - the array of radial points for the output coefficients;
                must form an increasing sequence and start from r>0.
    \param[out] Phi  - the array of sph.-harm. coefficients for the potential:
                Phi[k][c] is the value of c-th coefficient (where c is a single index 
                combining both l and m) at the radius r_k; will be resized as needed.
    \param[out] dPhi - the array of radial derivatives of each sph.-harm. term:
                dPhi_{l,m}(r) = d(Phi_{l,m})/dr; will be resized as needed.
    \throws std::invalid_argument if gridRadii are not correct.
*/
void computePotentialCoefsSph(const BaseDensity &dens, 
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi);

/** Same as above, but compute coefficients from the potential directly,
    without solving Poisson equation */
void computePotentialCoefsSph(const BasePotential& pot,
    const math::SphHarmIndices& ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi);

/** Compute the coefficients of spherical-harmonic potential expansion
    from an N-body snapshot.
    \tparam ParticleT  is any of the 6 principal particle types
    (3 coordinate systems, with or without velocity data, which is not used anyway).
    \param[in] points  is the array of point masses.
    \param[in] ind   is the coefficient indexing scheme (defines the order of expansion
    and its symmetries).
    \param[in] gridRadii is the grid in spherical radius.
    \param[out] Phi, dPhi  will contain the arrays of computed coefficients
    for potential and its radial derivative, with the same convention as used
    in the constructor of `Multipole`; will be resized as needed.
*/
template<typename ParticleT>
void computePotentialCoefsSph(
    const particles::PointMassArray<ParticleT> &points,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi);


/** Spherical-harmonic expansion of density with coefficients being spline functions of radius */
class DensitySphericalHarmonic: public BaseDensity {
public:
    /** construct the object from the provided density model and grid parameters */
    static PtrDensity create(const BaseDensity& src, int lmax, int mmax,
        unsigned int gridSizeR, double rmin, double rmax);

    /** construct the object from stored coefficients */
    DensitySphericalHarmonic(const std::vector<double> &gridRadii,
        const std::vector< std::vector<double> > &coefs);

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensitySphericalHarmonic"; };

    /** return the radii of spline nodes and the array of density expansion coefficients */
    void getCoefs(std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray) const;

private:
    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// radial dependence of each sph.-harm. expansion term
    std::vector<math::CubicSpline> spl;

    /// logarithmic density slopes 's' at small and large radii (rho ~ r^s)
    std::vector<double> innerSlope, outerSlope;

    virtual double densityCar(const coord::PosCar &pos) const {
        return densitySph(toPosSph(pos)); }

    virtual double densityCyl(const coord::PosCyl &pos) const {
        return densitySph(toPosSph(pos)); }

    virtual double densitySph(const coord::PosSph &pos) const;  // this is the implementation

};  // class DensitySphericalHarmonic


/** Auxiliary potential class that represents an array of multipole terms,
    with each term being a sum of two power-law profiles.
    It is internally used by spherical-harmonic and azimuthal Fourier expansion potential classes,
    as an extrapolation to small or large radii beyond the definition region of the main potential.
    Each term with index {l,m} is given by
    \f$  \Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}}            + W_{l,m} * (r/r0)^v  \f$  if s!=v,
    \f$  \Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}} * ln(r/r0) + W_{l,m} * (r/r0)^v  \f$  if s==v.
    Here v=l for the inward extrapolation and v=-1-l for the outward extrapolation,
    so that the term W r^v represents the 'main' multipole component corresponding 
    to the Laplace equation, i.e. with zero density. The other term U r^s corresponds
    to a power-law density profile of the given harmonic component (rho ~ r^{s-2}). 
*/
class PowerLawMultipole: public BasePotentialSph {
public:
    /** Create the potential from the three arrays: amplitudes of harmonic coefficients (U, W)
        and the power-law slope of the coefficient U with nonzero Laplacian (S),
        the reference radius r0 and the flag choosing between inward and outward extrapolation */
    PowerLawMultipole(double r0, bool inner,
        const std::vector<double>& S,
        const std::vector<double>& U,
        const std::vector<double>& W);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "PowerLaw"; };
private:
    const math::SphHarmIndices ind; ///< indexing scheme for sph.-harm.coefficients
    double r0;                      ///< reference radius 
    bool inner;                     ///< whether this is an inward or outward extrapolation
    std::vector<double> S, U, W;    ///< sph.-harm.coefficients for extrapolation
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;
};

/// Multipole expansion for potentials
class Multipole: public BasePotentialSph{
public:
    /** create the potential from the analytic density or potential model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        It exists in two variants: the first one takes a density model as input
        and solves Poisson equation to find the potential sph.-harm. coefficients;
        the second one takes a potential model and computes these coefs directly.
        \param[in]  src        is the input density or potential model;
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent.
    */
    static PtrPotential create(const BaseDensity& src, int lmax, int mmax,
        unsigned int gridSizeR, double rmin, double rmax);

    /** same as above, but takes a potential model as an input */
    static PtrPotential create(const BasePotential& src, int lmax, int mmax,
        unsigned int gridSizeR, double rmin, double rmax);
    
    /** construct the potential from the set of spherical-harmonic coefficients.
        \param[in]  radii  is the grid in radius;
        \param[in]  Phi  is the matrix of harmonic coefficients for the potential;
                    its first dimension is equal to the number of radial grid points,
                    and the second gives the number of coefficients (lmax+1)^2;
        \param[in]  dPhi  is the matrix of radial derivatives of harmonic coefs
                    (same size as Phi, each element is  d Phi_{l,m}(r) / dr ).
    */
    Multipole(const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);

    /** return the array of spherical-harmonic expansion coefficients.
        \param[out] radii will contain the radii of grid nodes;
        \param[out] Phi   will contain the spherical-harmonic expansion coefficients
                    for the potential at the given radii;
        \param[out] dPhi  will contain the radial derivatives of these coefs.
    */
    void getCoefs(std::vector<double> &radii,
        std::vector<std::vector<double> > &Phi,
        std::vector<std::vector<double> > &dPhi) const;

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "Multipole"; };

private:
    /// radial grid
    const std::vector<double> gridRadii;

    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// actual potential implementation (based either on 1d or 2d interpolating splines)
    PtrPotential impl;

    /// asymptotic behaviour at small and large radii described by `PowerLawMultipole`
    PtrPotential asymptInner, asymptOuter;

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const;
};
    
}  // namespace
