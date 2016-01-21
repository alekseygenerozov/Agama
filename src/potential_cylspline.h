/** \file    potential_cylspline.h
    \brief   potential approximation based on 2d spline in cylindrical coordinates
    \author  Eugene Vasiliev
    \date    2014-2016
**/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_spline.h"
#include "smart.h"

namespace math {
    // smart pointer to a generic 2d interpolation class
    typedef std::tr1::shared_ptr<const BaseInterpolator2d> PtrInterpolator2d;
}

namespace potential {

// ------ old api, to be removed soon ------ //

/** Angular expansion of potential in azimuthal angle 
    with coefficients being 2d spline functions of R,z.
    This is a very flexible and powerful potential approximation that can be used 
    to represent arbitrarily flattened axisymmetric or non-axisymmetric 
    (if the number of azimuthal terms is >0) density profiles.
*/
class CylSplineExpOld: public BasePotentialCyl
{
public:
    /// init potential from analytic mass model specified by its density profile
    /// (using CPotentialDirect for intermediate potential computation)
    CylSplineExpOld(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const BaseDensity& density, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    /// init potential from analytic mass model specified by a potential-density pair
    CylSplineExpOld(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const BasePotential& potential, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    /// init potential from stored coefficients
    CylSplineExpOld(const std::vector<double>& gridR, const std::vector<double>& gridz, 
        const std::vector< std::vector<double> >& coefs);

    /// init potential from N-body snapshot
    CylSplineExpOld(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const particles::PointMassArray<coord::PosCyl>& points, coord::SymmetryType _sym=coord::ST_TRIAXIAL, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CylSplineExpOld"; };
    virtual coord::SymmetryType symmetry() const { return mysymmetry; };

    /** retrieve coefficients of potential approximation.
        \param[out] gridR will be filled with the array of R-values of grid nodes
        \param[out] gridz will be filled with the array of z-values of grid nodes
        \param[out] coefs will contain array of sequentially stored 2d arrays 
        (the size of the outer array equals the number of terms in azimuthal expansion,
        inner arrays contain gridR.size()*gridz.size() values). */
    void getCoefs(std::vector<double> &gridR, std::vector<double>& gridz, 
        std::vector< std::vector<double> > &coefs) const;

private:
    coord::SymmetryType mysymmetry;           ///< may have different type of symmetry
    std::vector<double> grid_R, grid_z;       ///< nodes of the grid in cylindrical radius and vertical direction
    double Rscale;                            ///< scaling coefficient for transforming the interpolated potential
    std::vector<math::CubicSpline2d> splines; ///< array of 2d splines for each m-component in the expansion
    double C00, C20, C40, C22;                ///< multipole coefficients for extrapolation beyond the grid

    /// compute potential and its derivatives
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

    /// create interpolation grid and compute potential at grid nodes
    void initPot(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, 
        unsigned int _Ncoefs_phi, const BasePotential& potential, 
        double radius_min, double radius_max, double z_min, double z_max);

    /// create 2d interpolation splines in scaled R,z grid
    void initSplines(const std::vector< std::vector<double> > &coefs);

    /// compute m-th azimuthal harmonic of potential 
    /// (either by Fourier transform or calling the corresponding method of DirectPotential)
    double computePhi_m(double R, double z, int m, const BasePotential& potential) const;
};


// ------ new api ------ //

/** Density profile expressed as a Fourier expansion in azimuthal angle (phi)
    with coefficients interpolated on a 2d grid in meridional plane (R,z).
*/
class DensityAzimuthalHarmonic: public BaseDensity {
public:
    static PtrDensity create(const BaseDensity& src, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax, 
        unsigned int gridSizez, double zmin, double zmax);
    
    /** construct the object from the array of coefficients */
    DensityAzimuthalHarmonic(
        const std::vector<double> &gridR,
        const std::vector<double> &gridz,
        const std::vector< math::Matrix<double> > &coefs);
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensityAzimuthalHarmonic"; };

    /** retrieve the values of density expansion coefficients 
        and the nodes of 2d grid used for interpolation */
    void getCoefs(std::vector<double> &gridR, std::vector<double> &gridz, 
        std::vector< math::Matrix<double> > &coefs) const;
    
    /** return the value of m-th Fourier harmonic at the given point in R,z plane */
    double rho_m(int m, double R, double z) const;

private:
    std::vector<math::CubicSpline2d> spl;  ///< spline for rho_m(R,z)
    coord::SymmetryType sym;  ///< type of symmetry deduced from coefficients

    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }

    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }

    /** Return density interpolated on the 2d grid, or zero if the point lies outside the grid */
    virtual double densityCyl(const coord::PosCyl &pos) const;
};

/** Angular expansion of potential in azimuthal angle (phi)
    with coefficients being 2d spline functions of R,z.
*/
class CylSplineExp: public BasePotentialCyl
{
public:

    /** Create the potential from the provided density model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential: it creates the grids, computes the coefficients
        and calls the actual class constructor.
        It exists in two variants: the first one takes a density model as input
        and solves Poisson equation to find the potential azimuthal harmonic coefficients;
        the second one takes a potential model and computes these coefs directly.
        \param[in]  src        is the input density or potential model;
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the number of grid nodes in cylindrical radius (semi-logarithmic);
        \param[in]  Rmin, Rmax give the radial grid extent
                    (first non-zero node and the outermost node);
        \param[in]  gridSizez  is the number of grid nodes in vertical direction;
        \param[in]  zmin, zmax give the vertical grid extent (first non-zero positive node
                    and the outermost node; if the source model is not symmetric w.r.t.
                    z-reflection, a mirrored extension of the grid to negative z will be created).
    */
    static PtrPotential create(const BaseDensity& src, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax, 
        unsigned int gridSizez, double zmin, double zmax);

    /** Same as above, but taking a potential model as an input. */
    static PtrPotential create(const BasePotential& src, int mmax,
       unsigned int gridSizeR, double Rmin, double Rmax, 
       unsigned int gridSizez, double zmin, double zmax);

    /** Construct the potential from previously computed coefficients.
        \param[in]  gridR  is the grid in cylindrical radius
        (nodes must start at 0 and be increasing with R);
        \param[in]  gridz  is the grid in the vertical direction:
        if it starts at 0 and covers the positive half-space, then the potential
        is assumed to be symmetric w.r.t. z-reflection, so that the internal grid
        will be extended to the negative half-space; in the opposite case it
        is assumed to be asymmetric and the grid must cover negative z too.
        \param[in]  Phi  is the 3d array of harmonic coefficients:
        the outermost dimension determines the order of expansion - the number of terms
        is 2*mmax+1, so that m runs from -mmax to mmax inclusive (i.e. the m=0 harmonic
        is contained in the element with index mmax).
        Each element of this array (apart from the one at mmax, i.e. for m=0) may be empty,
        in which case the corresponding harmonic is taken to be identically zero.
        Non-empty elements are matrices with dimension gridR.size() * gridz.size(),
        regardless of whether gridz covers only z>=0 or both positive and negative z.
        The indexing scheme is Phi[m+mmax](iR,iz) = Phi_m(gridR[iR], gridz[iz]).
        \param[in]  dPhidR  is the array of radial derivatives of the potential,
        with the same shape as Phi, and containing the same number of non-empty terms.
        \param[in]  dPhidz  is the array of vertical derivatives.
        If both dPhidR and dPhidz are empty arrays (not arrays with empty elements),
        then the potential is constructed using only the values of Phi at grid nodes,
        employing 2d cubic spline interpolation for each m term.
        If derivatives are provided, then the interpolation is based on quintic splines,
        improving the accuracy.
    */
    CylSplineExp(
        const std::vector<double> &gridR,
        const std::vector<double> &gridz, 
        const std::vector< math::Matrix<double> > &Phi,
        const std::vector< math::Matrix<double> > &dPhidR = std::vector< math::Matrix<double> >(),
        const std::vector< math::Matrix<double> > &dPhidz = std::vector< math::Matrix<double> >() );

    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CylSplineExp"; };
    virtual coord::SymmetryType symmetry() const { return sym; };

    /** retrieve coefficients of potential approximation.
        \param[out] gridR will be filled with the array of R-values of grid nodes;
        \param[out] gridz will be filled with the array of z-values of grid nodes:
        if the potential is symmetric w.r.t. z-reflection, then only the half-space with
        non-negative z is returned both in this array and in the coefficients.
        \param[out] Phi will contain array of sequentially stored 2d arrays 
        (the size of the outer array equals the number of terms in azimuthal expansion (2*mmax+1),
        inner arrays contain gridR.size()*gridz.size() values).
        \param[out] dPhidR will contain the array of derivatives in the radial direction,
        with the same shape as Phi.
        \param[out] dPhidz will contain the array of derivatives in the vertical direction.
    */
    void getCoefs(
        std::vector<double> &gridR,
        std::vector<double> &gridz, 
        std::vector< math::Matrix<double> > &Phi,
        std::vector< math::Matrix<double> > &dPhidR,
        std::vector< math::Matrix<double> > &dPhidz) const;

private:
    /// array of 2d splines (for each m-component in the expansion in azimuthal angle)
    std::vector<math::PtrInterpolator2d> spl;
    coord::SymmetryType sym;  ///< type of symmetry deduced from coefficients
    double Rscale;            ///< radial scaling factor

    /// asymptotic behaviour at large radii described by `PowerLawMultipole`
    PtrPotential asymptOuter;

    /// compute potential and its derivatives
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};


/** Compute the coefficients of azimuthal Fourier expansion of density profile,
    used for constructing a DensityAzimuthalHarmonic object.
    The input density values are taken at the nodes of 2d grid in (R,z) specified by
    two one-dimensional arrays, gridR and gridz, and nphi distinct values of phi, 
    where nphi=mmax+1 if the density is reflection-symmetric in y, or nphi=2*mmax+1 otherwise.
    The total number of density evaluations is gridR.size() * gridz.size() * nphi.
*/
void computeDensityCoefsCyl(const BaseDensity &dens,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &coefs);

/** Compute the coefficients of azimuthal Fourier expansion of potential by
    taking the values and derivatives of the source potential at nodes of 2d grid in (R,z)
    and equally-spaced nodes in phi (same as for `computeDensityCoefsCyl`).
    The input and output array conventions match those of the constructor of `CylSplineExp`;
    the output arrays will be resized as needed.
*/
void computePotentialCoefsCyl(const BasePotential &pot, 
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz);

/** Compute the coefficients of azimuthal Fourier expansion of potential
    from the given density profile, used for creating a CylSpline object.
    Unlike the overloaded function that accepts `BasePotential` as input,
    this one takes `BaseDensity` and thus solves the Poisson equation
    in cylindrical coordinates, by creating a Fourier expansion of density
    in azimuthal angle (phi) and using 2d numerical integration to compute
    the values and derivatives of each Fourier component of potential 
    at the nodes of 2d grid in R,z plane. This is a rather costly calculation.
    The input and output array conventions match those of the constructor of `CylSplineExp`;
    the output arrays will be resized as needed.
*/
void computePotentialCoefsCyl(const BaseDensity& dens, 
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz);
    
}  // namespace
