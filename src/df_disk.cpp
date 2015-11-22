#include "df_disk.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

PseudoIsothermal::PseudoIsothermal(
    const PseudoIsothermalParam &params, const potential::InterpEpicycleFreqs& freqs) :
    par(params), freq(freqs)
{
    // sanity checks on parameters
    if(par.Rdisk<=0)
        throw std::invalid_argument("PseudoIsothermal DF: disk scale length must be positive");
    if(par.sigmar0<=0 || par.sigmaz0<=0)
        throw std::invalid_argument("PseudoIsothermal DF: velocity dispersion scale must be positive");
}

double PseudoIsothermal::value(const actions::Actions &J) const {
    double kappa, nu, Omega;   // characteristic epicyclic freqs
    freq.eval(fmax(par.Jphimin, J.Jphi), kappa, nu, Omega);
    double Rcirc     = J.Jphi!=0 ? sqrt(fabs(J.Jphi) / Omega) : 0;
    double exp_rad   = exp( -Rcirc / par.Rdisk );      // exponential profile in radius
    if(exp_rad<1e-100)   // we're too far out
        return 0;
    double sigmarsq  = pow_2(par.sigmar0) * exp_rad;   // radial velocity dispersion squared
    sigmarsq        += pow_2(par.sigmaMin);
    double sigmazsq  = pow_2(par.sigmaz0) * exp_rad;   // vertical velocity dispersion squared
    double Sigma     = par.Sigma0 * exp_rad;           // surface density
    double exp_act   = exp( -kappa * J.Jr / sigmarsq - nu * J.Jz / sigmazsq );
    double exp_Jphi  =                            // suppression factor for counterrotating orbits:
        par.L0 == INFINITY || J.Jphi == 0 ? 1. :  // do not distinguish the sign of Lz at all
        par.L0 == 0 ? (J.Jphi>0 ? 2. : 0.) :      // strictly use only orbits with positive Lz
        1 + tanh(J.Jphi / par.L0);                // intermediate regime, mildly cut off DF at negative Lz
    double numerator = par.norm * exp_act * exp_Jphi * Sigma * Omega * nu;
    if(numerator==0 || !math::isFinite(numerator))
       return 0;
    else
       return numerator / (4*M_PI*M_PI * kappa * sigmarsq * sigmazsq);
}

}  // namespace df
