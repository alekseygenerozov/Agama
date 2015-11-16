/** \file    df_disk.h
    \brief   Distribution function for the disk component
*/
#pragma once
#include "df_base.h"
#include "potential_utils.h"
#include "math_spline.h"

namespace df{

/// \name   Class for action-based pseudo-isothermal disk distribution function (DF)
///@{

/// Parameters that describe a pseudo-isothermal distribution function.
struct PseudoIsothermalParam{
    double norm;     ///< overall normalization factor
    double Rdisk;    ///< scale radius of the (exponential) disk surface density
    double L0;       ///< scale angular momentum determining the suppression of retrograde orbits
    double Sigma0;   ///< surface density normalization (value at origin)
    double sigmar0;  ///< normalization of radial velocity dispersion at Rdisk
    double sigmaz0;  ///< normalization of vertical velocity dispersion at Rdisk
    double sigmaMin; ///< lower limit on the radial velocity dispersion
};

/** Distribution function for quasi-isothermal disk, used in Binney&McMillan 2011:
    \f$  f(J) = f_r(J_r, J_\phi)  f_z(J_z, J_\phi)  f_\phi(J_\phi)  \f$, where
    \f$  f_r  = \Omega(R_c) \Sigma(R_c) / (\pi \kappa(R_c \sigma_r^2(R_c) )  \f$,
    \f$  f_z  = \nu / (2\pi \sigma_z^2(R_c) )  \f$,
    \f$  f_\phi = 1 + \tanh(J_\phi / L0)  \f$.
    It is defined in terms of surface density \f$ \Sigma \f$ and velocity dispersions
    \f$ \sigma_r, \sigma_z \f$, computed assuming an exponential disk density profile,
    and characteristic epicyclic frequencies \f$ \kappa, \nu, \Omega \f$, all evaluated
    at a radius R_c that corresponds to the radius of a circular orbit with angular momentum
    L_z = J_phi (recall that the DF is defined in terms of actions only, not coordinates).
    Note, however, that the while these frequencies are computed from a particular potential
    and passed as the interpolator object, the DF uses them simply as one-parameter functions
    of the azimuthal actions, without regard to whether they actually correspond to
    the epicyclic frequencies in the potential that this DF is used.
    In other words, action-based DF may only depend on actions and on some arbitrary function
    of them, but not explicitly on the potential.
*/
class PseudoIsothermal: public BaseDistributionFunction{
private:
    const PseudoIsothermalParam par;            ///< parameters of DF
    const potential::InterpEpicycleFreqs freq;  ///< interface providing the epicyclic frequencies
public:
    /** Create an instance of pseudo-isothermal distribution function with given parameters
        \param[in] params  are the parameters of DF;
        \param[in] freqs   is the instance of object that computes epicyclic frequencies:
        a copy of this object is kept in the DF, so that one may pass a temporary variable
        as the argument, like:
            df = new PseudoIsothermal(params, InterpEpicycleFreqs(potential));
        Since the potential itself is not used in the InterpEpicycleFreqs object,
        the DF is independent of the actual potential in which it is later used,
        which could be different from the one that was employed at construction.
        \throws std::invalid_argument exception if parameters are nonsense
    */
    PseudoIsothermal(const PseudoIsothermalParam& params, const potential::InterpEpicycleFreqs& freqs);

    /** return value of DF for the given set of actions
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;
};

///@}
}  // namespace df
