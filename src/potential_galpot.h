/** \file    potential_galpot.h
    \brief   Walter Dehnen's GalaxyPotential code
    \author  Walter Dehnen, Paul McMillan, Eugene Vasiliev
    \date    1996-2015

Copyright Walter Dehnen, 1996-2005 
e-mail:   walter.dehnen@astro.le.ac.uk 
address:  Department of Physics and Astronomy, University of Leicester 
          University Road, Leicester LE1 7RH, United Kingdom 

Put into the Torus code (with a minimum of fuss) by Paul McMillan, Oxford 2010
email: p.mcmillan1@physics.ox.ac.uk

Modifications by Eugene Vasiliev, 2015-2016


The method, explained in Dehnen & Binney (1998, MNRAS, 294, 429) and based 
on the approach of Kuijken & Dubinski (1994, MNRAS, 269, 13), is applicable 
to any disk density profile which is separable in cylindrical coordinates.

Let the density profile of the disk be

\f$  \rho_d(R,z) = f(R) h(z)  \f$,

and let H(z) be the second integral of h(z) over z.
Then the potential of the disk can be written as a sum of 'main' and 'residual' parts:

\f$  \Phi(R,z) = 4\pi f(r) H(z) + \Phi_{res}  \f$,

where the argument of f is spherical rather than cylindrical radius, 
and the residual potential is generated by the following density profile:

\f$  \rho_{res} = [f(R)-f(r)] h(z) - f''(r) H(z) - 2 f'(r) [H(z) + z H'(z)]/r  \f$.

This residual potential is not strongly confined to the disk plane, and can be 
efficiently approximated by a multipole expanion, which, in turn, is represented 
by a two-dimensional quintic spline in (R,z) plane.

The original GalaxyPotential uses this method for any combination of disk components 
and additional, possibly flattened spheroidal components: the residual density of all 
disks and the entire density of spheroids serves as the source to the Multipole potential 
approximation.

In the present modification, the GalaxyPotential class is replaced by a more generic Composite 
potential, which contains one Multipole potential and possibly several DiskAnsatz components.
The latter come in pairs with DiskDensity density components, so that the difference between
the full input density and the one provided by DiskAnsatz is used in the multipole expansion.
A composite density model with all SpheroidDensity components and all pairs of DiskDensity minus 
DiskAnsatz components is used to initialize the Multipole potential.
Of course this input may be generalized to contain other density components, and the Composite
potential may also contain some other potential models apart from DiskAnsatz and Multipole. 

The Multipole potential solves the Poisson equation using the spherical-harmonic expansion
of its input density profile, and then stores the values and derivatives of potential on 
a 2d grid in (r,theta) plane, so that the potential evaluation uses 2d spline-interpolated 
values; however, if the radius lies outside the grid definition region, the potential is computed
by summing up appropriately extrapolated multipole components (unlike the original GalPot).

For compatibility with the original implementation, an utility function `readGalaxyPotential`
is provided in potential_factory.h, taking the name of parameter file and the Units object as parameters.
*/

#pragma once
#include "potential_base.h"
#include "smart.h"
#include <vector>

namespace potential{

/// \name  Separable disk density profile
///@{

/** parameters that describe a disk component.

    Specification of a disk density profile separable in R and z requires two auxiliary function,
    f(R) and H(z)  (the former essentially describes the surface density of the disk,
    and the latter is the second antiderivative of vertical density profile h(z) ).
    They are used by both DiskAnsatz potential and DiskDensity density classes.
    In the present implementation they are the same as in GalPot:

    \f$  \rho = f(R) h(z)  \f$,

    \f$  f(R) = \Sigma_0  \exp [ -R_0/R - R/R_d + \epsilon \cos(R/R_d) ]  \f$,

    \f$  h(z) = \delta(z)                 \f$  for  h=0, or 
    \f$  h(z) = 1/(2 h)  * exp(-|z/h|)    \f$  for  h>0, or
    \f$  h(z) = 1/(4|h|) * sech^2(|z/2h|) \f$  for  h<0.

    The corresponding second antiderivatives of h(z) are given in Table 2 of Dehnen&Binney 1998.
    Alternatively, one may provide two arbitrary 1d functions to be used in the separable profile.
*/
struct DiskParam{
    double surfaceDensity;      ///< surface density normalisation Sigma_0
    double scaleRadius;         ///< scale length R_d
    double scaleHeight;         ///< scale height h: 
    ///< For h<0 an isothermal (sech^2) profile is used, for h>0 an exponential one, 
    ///< and for h=0 the disk is infinitesimal thin
    double innerCutoffRadius;   ///< if nonzero, specifies the radius of a hole at the center R_0
    double modulationAmplitude; ///< a term eps*cos(R/R_d) is added to the radial exponent
    DiskParam(double _surfaceDensity=0, double _scaleRadius=1, double _scaleHeight=0,
        double _innerCutoffRadius=0, double _modulationAmplitude=0) :
        surfaceDensity(_surfaceDensity), scaleRadius(_scaleRadius), scaleHeight(_scaleHeight),
        innerCutoffRadius(_innerCutoffRadius), modulationAmplitude(_modulationAmplitude) {};
    double mass() const;        ///< return the total mass of a density profile with these parameters
};

/** helper routine to create an instance of radial density function */
math::PtrFunction createRadialDiskFnc(const DiskParam& params);

/** helper routine to create an instance of vertical density function */
math::PtrFunction createVerticalDiskFnc(const DiskParam& params);

/** Density profile of a separable disk model */
class DiskDensity: public BaseDensity {
public:
    /// construct the density profile with provided parameters
    DiskDensity(const DiskParam& _params) : 
        radialFnc  (createRadialDiskFnc(_params)),
        verticalFnc(createVerticalDiskFnc(_params)) {};
    /// construct a generic profile with user-specified radial and vertical functions
    DiskDensity(const math::PtrFunction& _radialFnc, const math::PtrFunction& _verticalFnc) :
        radialFnc(_radialFnc), verticalFnc(_verticalFnc) {}
    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "DiskDensity"; return text; }
private:
    math::PtrFunction radialFnc;     ///< function describing radial dependence of surface density
    math::PtrFunction verticalFnc;   ///< function describing vertical density profile
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densityCar(const coord::PosCar &pos) const
    {  return densityCyl(toPosCyl(pos)); }
    virtual double densitySph(const coord::PosSph &pos) const
    {  return densityCyl(toPosCyl(pos)); }
};

/** Part of the disk potential provided analytically as  4 pi f(r) H(z) */
class DiskAnsatz: public BasePotentialCyl {
public:
    DiskAnsatz(const DiskParam& _params) : 
        radialFnc  (createRadialDiskFnc(_params)),
        verticalFnc(createVerticalDiskFnc(_params)) {};
    DiskAnsatz(const math::PtrFunction& _radialFnc, const math::PtrFunction& _verticalFnc) :
        radialFnc(_radialFnc), verticalFnc(_verticalFnc) {};
    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "DiskAnsatz"; return text; }
private:
    math::PtrFunction radialFnc;     ///< function describing radial dependence of surface density
    math::PtrFunction verticalFnc;   ///< function describing vertical density profile
    /** Compute _part_ of disk potential: f(r)*H(z) */
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
    virtual double densityCyl(const coord::PosCyl &pos) const;
};

///@}
/// \name  Spheroical density profile
///@{

/** Parameters describing a spheroidal component with a Zhao(1996) alpha-beta-gamma
    density profile and an optional exponential cutoff:
    \f$  \rho = \rho_0  (r/r_0)^{-\gamma} ( 1 + (r/r_0)^\alpha )^{(\gamma-\beta) / \alpha}
    \exp[ -(r/r_{cut})^2], \f$,
    where  \f$ r = \sqrt{ x^2 + y^2/p^2 + z^2/q^2 } \f$  is the ellipsoidal radius.
*/
struct SphrParam{
    double densityNorm;         ///< density normalization rho_0
    double axisRatioY;          ///< axis ratio p (y/R)
    double axisRatioZ;          ///< axis ratio q (z/R)
    double alpha;               ///< steepness of transition alpha
    double beta;                ///< outer power slope beta
    double gamma;               ///< inner power slope gamma
    double scaleRadius;         ///< transition radius r_0
    double outerCutoffRadius;   ///< outer cut-off radius r_{cut}
    SphrParam(double _densityNorm=0, double _axisRatioY=1, double _axisRatioZ=1,
        double _alpha=1, double _beta=4, double _gamma=1,
        double _scaleRadius=1, double _outerCutoffRadius=0) :
        densityNorm(_densityNorm), axisRatioY(_axisRatioY), axisRatioZ(_axisRatioZ),
        alpha(_alpha), beta(_beta), gamma(_gamma),
        scaleRadius(_scaleRadius), outerCutoffRadius(_outerCutoffRadius) {};
    double mass() const;        ///< return the total mass of a density profile with these parameters
};

/** Density profile of a double-power-law model described by SphrParam */
class SpheroidDensity: public BaseDensity{
public:
    SpheroidDensity (const SphrParam &_params);
    virtual coord::SymmetryType symmetry() const { 
        return params.axisRatioY!=1 ? coord::ST_TRIAXIAL :
            params.axisRatioZ!=1 ? coord::ST_AXISYMMETRIC : coord::ST_SPHERICAL; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "SpheroidDensity"; return text; }
private:
    SphrParam params;
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densityCar(const coord::PosCar &pos) const
    {  return densityCyl(toPosCyl(pos)); }
    virtual double densitySph(const coord::PosSph &pos) const
    {  return densityCyl(toPosCyl(pos)); }
};

///@}

/** Construct an array of potential components consisting of a Multipole and a number of 
    DiskAnsatz components, using the provided arrays of parameters for disks and spheroids;
    this array should be passed to the constructor of CompositeCyl potential,
    after more components being added to it if needed.
*/
std::vector<PtrPotential> createGalaxyPotentialComponents(
    const std::vector<DiskParam>& DiskParams,
    const std::vector<SphrParam>& SphrParams);

/** Construct a CompositeCyl potential consisting of a Multipole and a number of DiskAnsatz 
    components, using the provided arrays of parameters for disks and spheroids
    (a simplified interface for the previous routine in the case that no additional 
    components are needed).
*/
PtrPotential createGalaxyPotential(
    const std::vector<DiskParam>& DiskParams,
    const std::vector<SphrParam>& SphrParams);

} // namespace potential
