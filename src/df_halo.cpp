#include "df_halo.h"
//#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>

namespace df{

BaseDoublePowerLaw::BaseDoublePowerLaw(const DoublePowerLawParam &inparams) :
    par(inparams)
{
    // sanity checks on parameters
    if(par.j0<=0)
        throw std::invalid_argument("DoublePowerLaw DF: break action j0 must be positive");
    if(par.jcore<0)
        throw std::invalid_argument("DoublePowerLaw DF: core action jcore must be non-negative");
    if(par.jmax<0)
        throw std::invalid_argument("DoublePowerLaw DF: cutoff action jmax must be non-negative");
    if(par.alpha<0)
        throw std::invalid_argument("DoublePowerLaw DF: inner slope alpha must be non-negative");
    if(par.beta<=3 && par.jmax==0)
        throw std::invalid_argument(
            "DoublePowerLaw DF: mass diverges at large J (outer slope beta must be > 3)");
    if(par.jcore==0 && par.alpha>=3)
        throw std::invalid_argument(
            "DoublePowerLaw DF: mass diverges at J->0 (inner slope alpha must be < 3)");
    par.norm /= pow_3(2*M_PI); 
    //   * math::gamma(3-par.alpha) * math::gamma(par.beta-3) / math::gamma(par.beta-par.alpha);
}

double BaseDoublePowerLaw::value(const actions::Actions &J) const {
    // linear combination of actions in the inner part of the model (for J<J0)
    double hJ  = h(J);
    // linear combination of actions in the outer part of the model (for J>J0)
    double gJ  = g(J);
    double val = par.norm / pow_3(par.j0) *                // overall normalization factor
        pow(1. + par.j0 / (hJ + par.jcore), par.alpha) *   // numerator
        pow(1. + (gJ + par.jcore) / par.j0, -par.beta);    // denominator
    if(par.jmax>0)
        val *= exp(-pow_2(gJ / par.jmax));                 // exponential cutoff at large J
    return val;
}

// ------------ Posti et al. ----------- //
DoublePowerLaw::DoublePowerLaw(const DoublePowerLawParam &inparams) :
    BaseDoublePowerLaw(inparams)
{
    if( par.ar<=0 || par.az<=0 || par.aphi<=0 ||
        par.br<=0 || par.bz<=0 || par.bphi<=0 )
        throw std::invalid_argument(
            "DoublePowerLaw DF: coefficients in the linear combination of actions must be positive");
}

double DoublePowerLaw::h(const actions::Actions &J) const {
    return par.ar*J.Jr + par.az*J.Jz + par.aphi*fabs(J.Jphi);
}

double DoublePowerLaw::g(const actions::Actions &J) const {
    return par.br*J.Jr + par.bz*J.Jz + par.bphi*fabs(J.Jphi);
}

DoublePowerLawSph::DoublePowerLawSph(const DoublePowerLawParam &inparams,
    const potential::InterpEpicycleFreqs& freqs) :
    BaseDoublePowerLaw(inparams), freq(freqs)
{
    if( par.b<=0 )
        throw std::invalid_argument(
        "DoublePowerLaw DF: anisotropy coefficient 'b' should be positive");
}

double DoublePowerLawSph::h(const actions::Actions &J) const {
    double Jsum = J.Jr + J.Jz + fabs(J.Jphi);
    double kappa, nu, Omega;   // characteristic epicyclic freqs
    freq.eval(Jsum, kappa, nu, Omega);
    double s = par.b==1 ? 0 : (par.b-1)/2 * pow_2(tanh(1 - J.Jr/Jsum));
    double A = (par.b+1)/2 + s, B = (par.b+1)/2 - s;
    return J.Jr / A + Omega/kappa * (Jsum-J.Jr) / B;
}


}  // namespace df
