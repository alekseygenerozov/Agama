/** \file    df_halo.h
    \brief   Distribution functions for the spheroidal component (halo)
*/
#pragma once
#include "df_base.h"
#include "potential_utils.h"

namespace df{

/// \name   Classes for action-based double power-law distribution function (DF)
///@{

/// Parameters that describe a double power law distribution function.
struct DoublePowerLawParam{
double
    norm,  ///< normalization factor with the dimension of mass
    j0,    ///< break action (defines the transition between inner and outer regions)
    jcore, ///< core action (sets upper limit on DF at J<Jcore)
    jmax,  ///< cutoff action (sets exponential suppression at J>Jmax, 0 to disable)
    alpha, ///< power-law index for actions below the break action
    beta,  ///< power-law index for actions above the break action
    ar,    ///< weight on radial actions below the break action
    az,    ///< weight on z actions below the break action
    aphi,  ///< weight oh angular actions below the break action
    br,    ///< weight on radial actions above the break action
    bz,    ///< weight on z actions above the break action
    bphi,  ///< weight on angular actions above the break action
    b;     ///< alternative to the above six parameters: a single anisotropy coefficient
DoublePowerLawParam() :  ///< set default values for all fields
    norm(0), j0(0), jcore(0), jmax(0), alpha(0), beta(0),
    ar(1), az(1), aphi(1), br(1), bz(1), bphi(1), b(1) {}
};

/** General double power-law model.
    The distribution function is given by
    \f$  f(J) = (1 + J_0 / (h(J) + J_{core}) )^\alpha / (1 + (g(J) + J_{core}) / J_0 )^\beta
         \times \exp[ - (g(J) / J_{max})^2 ] \f$,
    where h(J) and g(J) are two functions that should be approximately linear combinations
    of actions, specified in the derived classes, that control the behaviour of the model
    in the inner region (below the break action J_0) and the outer region, respectively.
*/
class BaseDoublePowerLaw: public BaseDistributionFunction{
public:
    /** Create an instance of double-power-law distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    BaseDoublePowerLaw(const DoublePowerLawParam &params);

    /** return value of DF for the given set of actions.
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;

protected:
    DoublePowerLawParam par;  ///< parameters of DF

    /// the shape function controlling the inner region of the model
    virtual double h(const actions::Actions& J) const = 0;
    
    /// the shape function controlling the outer region of the model
    virtual double g(const actions::Actions& J) const = 0;
};

/** Posti et al(2015) double-power-law model. The functions h(J) and g(J) are given by
    \f$  h(J) = a_r J_r + a_z J_z + a_\phi |J_\phi|  \f$,
    \f$  g(J) = b_r J_r + b_z J_z + b_\phi |J_\phi|  \f$.
*/
class DoublePowerLaw: public BaseDoublePowerLaw{
public:
    explicit DoublePowerLaw(const DoublePowerLawParam &params);
private:
    virtual double h(const actions::Actions& J) const;
    virtual double g(const actions::Actions& J) const;
};

/** Piffl et al(2015) double-power-law model. The function h(J)=g(J) is given by
    \f$  h(J) = (1/A(J)) J_r + (1/B(J)) (Omega(J_\phi)/kappa(J_\phi)) (J_z + |J_\phi|)  \f$,
    \f$  A(J) = (b+1)/2 + (b-1)/2 * C(J)  \f$,
    \f$  B(J) = (b+1)/2 - (b-1)/2 * C(J)  \f$, 
    \f$  C(J) = \tanh^2 [ 1 - J_r / (J_r + J_z + |J_\phi|) ]. \f$
    It gives a somewhat simplified model that corresponds to a spherical halo with 
    a nearly constant velocity anisotropy coefficient (same for both large and small radii),
    if constructed in isolation (b=1 corresponds to isotropic systems, b>1 -- to radially
    biased and 0<b<1 -- to tangentially biased). It also requires an interpolation tables
    for epicyclic frequencies to be provided (constructed for a given potential, which itself
    does not appear in the DF).
*/
class DoublePowerLawSph: public BaseDoublePowerLaw{
public:
    DoublePowerLawSph(const DoublePowerLawParam &params, const potential::Interpolator& freqs);
private:
    const potential::Interpolator freq;  ///< interface providing the epicyclic frequencies
    virtual double h(const actions::Actions& J) const;
    virtual double g(const actions::Actions& J) const { return h(J); }
};
    
///@}
}  // namespace df
