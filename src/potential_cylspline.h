/** \file    potential_cylspline.h
    \brief   potential approximation based on 2d spline in cylindrical coordinates
    \author  Eugene Vasiliev
    \date    2014-2015
**/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_spline.h"

namespace potential {

/** Angular expansion of potential in azimuthal angle 
    with coefficients being 2d spline functions of R,z.
    This is a very flexible and powerful potential approximation that can be used 
    to represent arbitrarily flattened axisymmetric or non-axisymmetric 
    (if the number of azimuthal terms is >0) density profiles.
*/
class CylSplineExp: public BasePotentialCyl
{
public:
    /// init potential from analytic mass model specified by its density profile
    /// (using CPotentialDirect for intermediate potential computation)
    CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const BaseDensity& density, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    /// init potential from analytic mass model specified by a potential-density pair
    CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const BasePotential& potential, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    /// init potential from stored coefficients
    CylSplineExp(const std::vector<double>& gridR, const std::vector<double>& gridz, 
        const std::vector< std::vector<double> >& coefs);

    /// init potential from N-body snapshot
    CylSplineExp(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
        const particles::PointMassArray<coord::PosCyl>& points, coord::SymmetryType _sym=coord::ST_TRIAXIAL, 
        double radius_min=0, double radius_max=0, double z_min=0, double z_max=0);

    ~CylSplineExp() {};
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "CylSplineExp"; };
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
    double Rscale;                            ///< scaling coefficient for transforming the interpolated potential; computed as -Phi(0)/Mtotal.
    std::vector<math::CubicSpline2d> splines; ///< array of 2d splines (for each m-component in the expansion in azimuthal angle)
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

/** Density profile expressed as a Fourier expansion in azimuthal angle (phi)
    with coefficients interpolated on a 2d grid in meridional plane (R,z).
*/
class DensityAzimuthalHarmonic: public BaseDensity {
public:
    /** construct the object from the array of coefficients */
    DensityAzimuthalHarmonic(const std::vector<double>& gridR, const std::vector<double>& gridz,
        const std::vector< std::vector<double> > &coefs);
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensityAzimuthalHarmonic"; };

    /** retrieve the values of density expansion coefficients 
        and the nodes of 2d grid used for interpolation */
    void getCoefs(std::vector<double> &gridR, std::vector<double>& gridz, 
        std::vector< std::vector<double> > &coefs) const;
    
    /** return the value of m-th Fourier harmonic at the given point in R,z plane */
    double rho_m(int m, double R, double z) const;

private:
    std::vector<math::CubicSpline2d> spl;  ///< spline for rho_m(R,z)
    coord::SymmetryType sym;

    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }

    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }

    /** Return density interpolated on the 2d grid, or zero if the point lies outside the grid */
    virtual double densityCyl(const coord::PosCyl &pos) const;
};

/** Compute the coefficients of azimuthal Fourier expansion of density profile,
    used for constructing a DensityAzimuthalHarmonic object.
    The input density values are taken at the nodes of 2d grid in (R,z) specified by
    two one-dimensional arrays, gridR and gridz, and nphi distinct values of phi, 
    where nphi=mmax+1 if the density is reflection-symmetric in y, or nphi=2*mmax+1 otherwise.
    The total number of density evaluations is gridR.size() * gridz.size() * nphi.
*/
void computeDensityCoefs(const BaseDensity& dens,
    const unsigned int mmax, const std::vector<double> &gridR, const std::vector<double> &gridz,
    std::vector< std::vector<double> > &coefs);

/** Compute the coefficients of azimuthal Fourier expansion of potential from
    from the given density profile, used for creating a CylSpline object.
*/
void computePotentialCoefs(const BaseDensity& dens, 
    const unsigned int mmax, const std::vector<double> &gridR, const std::vector<double> &gridz,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhidR, std::vector< std::vector<double> > &dPhidz);
    
}  // namespace
