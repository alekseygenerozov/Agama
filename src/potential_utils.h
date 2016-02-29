/** \file    potential_utils.h
    \brief   General routines for various tasks associated with potential classes
    \author  Eugene Vasiliev
    \date    2009-2015
*/
#pragma once
#include "potential_base.h"
#include "math_spline.h"

namespace potential{

/** Compute circular velocity at a given (cylindrical) radius R in equatorial plane */
double v_circ(const BasePotential& potential, double R);

/** Compute angular momentum of a circular orbit in equatorial plane for a given value of energy */
double L_circ(const BasePotential& potential, double energy);

/** Compute cylindrical radius of a circular orbit in equatorial plane for a given value of energy */
double R_circ(const BasePotential& potential, double energy);

/** Compute cylindrical radius of an orbit in equatorial plane for a given z-component
    of angular momentum */
double R_from_Lz(const BasePotential& potential, double L_z);


/** Compute epicycle frequencies for a circular orbit in the equatorial plane with radius R.
    \param[in]  potential is the instance of potential (must have axial symmetry)
    \param[in]  R     is the cylindrical radius 
    \param[out] kappa is the epicycle frequency of radial oscillations
    \param[out] nu    is the frequency of vertical oscillations
    \param[out] Omega is the azimuthal angular frequency (essentially v_circ/R)
*/
void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega);


/** Interpolator for epicyclic frequencies as functions of L_circ */
class InterpEpicycleFreqs{
public:
    /** The potential passed as parameter is only used to initialize the internal
        interpolation tables in the constructor, and is not used afterwards
        when interpolation is needed. */
    explicit InterpEpicycleFreqs(const BasePotential& potential);

    /// compute interpolated values of epicyclic frequencies for the given angular momentum
    void eval(double Lcirc, double& kappa, double& nu, double& Omega) const;
private:
    /// epicyclic frequencies as spline-interpolated functions of L_circ
    math::CubicSpline freqSum, freqKappa, freqNu;
};


/** Interpolator class for faster evaluation of L_circ(E) and R_circ(E) */
class InterpLcirc: public math::IFunction {
public:
    /** The potential passed as parameter is only used to initialize the internal
        interpolation tables in the constructor, and is not used afterwards
        when interpolation is needed. */
    explicit InterpLcirc(const BasePotential& potential);

    /// return L_circ(E) and optionally its first derivative
    virtual void evalDeriv(const double E, double* value=0, double* deriv=0, double* deriv2=0) const;
    virtual unsigned int numDerivs() const { return 1; }

    /// return R_circ(E)
    double Rcirc(const double E) const;
private:
    double Ein, Eout;           ///< boundaries of energy interval
    math::CubicSpline interpL;  ///< spline-interpolated scaled function for L_circ
    math::CubicSpline interpR;  ///< spline-interpolated scaled function for R_circ
};

}  // namespace potential
