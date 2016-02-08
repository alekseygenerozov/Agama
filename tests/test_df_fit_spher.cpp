/** \file    test_df_fit.cpp
    \author  Eugene Vasiliev
    \date    August 2015

    This example demonstrates how to find best-fit parameters of an action-based
    distribution function that matches the given N-body snapshot.

    The N-body model itself corresponds to a spherically-symmetric isotropic
    Hernquist profile, and we fit it with a double-power-law distribution function
    of Posti et al.2015. We use the exact potential (i.e., do not compute it
    from the N-body model itself, nor try to vary its parameters, although
    both options are possible), and compute actions for all particles only once.
    Then we scan the parameter space of DF, finding the maximum of the likelihood
    function with a multidimensional minimization algorithm.
    This takes a few hundred iterations to converge.
*/
#include "potential_dehnen.h"
#include "actions_staeckel.h"
#include "df_halo.h"
#include "particles_base.h"
#include "math_fit.h"
#include "math_core.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <iostream>

const unsigned int NPARAMS = 4;
typedef std::vector<actions::Actions> ActionArray;

/// convert from parameter space to DF params: note that we apply
/// some non-trivial scaling to make the life easier for the minimizer
df::DoublePowerLawParam dfparams(const double vars[])
{
    df::DoublePowerLawParam params;
    params.alpha = vars[0];
    params.beta  = vars[1];
    params.j0    = exp(vars[2]);
    params.ar    = 3./(2+vars[3])*vars[3];
    params.az    = 3./(2+vars[3]);
    params.aphi  = 3./(2+vars[3]);  // ensure that sum of ar*Jr+az*Jz+aphi*Jphi doesn't depend on vars[3]
    params.norm  = 1.;
    return params;
}

class ModelSearchFncLM: public math::IFunctionNdimDeriv {
public:
    ModelSearchFncLM(const ActionArray& _points) : points(_points) {};    
    /// compute the deviations of Hamiltonian from its average value for an array of points
    /// with the provided (scaled) parameters of toy map
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        const double EPS = 1e-4;
        df::PtrDistributionFunction df[NPARAMS+1];
        double norm[NPARAMS+1];
        df::DoublePowerLawParam params = dfparams(vars);
        std::cout << "J0="<<params.j0<<", Jcore="<<params.jcore<<
            ", alpha="<<params.alpha<<", ar="<<params.ar<<", az="<<params.az<<", aphi="<<params.aphi<<
            ", beta=" <<params.beta << ": ";
        try{
            for(int p=0; p <= (derivs? NPARAMS : 0); p++) {
                double var[NPARAMS] = {vars[0], vars[1], vars[2], vars[3]};
                if(p>0)
                    var[p-1] = vars[p-1] * (1+EPS);
                df[p].reset(new df::DoublePowerLaw(dfparams(var)));
                norm[p] = df[p]->totalMass(1e-6, 1000000);
            }
            double sumlog = 0;
            for(unsigned int i=0; i<points.size(); i++) {
                double val = -log(df[0]->value(points[i]) / norm[0]);
                if(values)
                    values[i] = val;
                if(derivs) {
                    for(int p=0; p<NPARAMS; p++) {
                        double valp = -log(df[p+1]->value(points[i]) / norm[p+1]);
                        derivs[i*NPARAMS + p] = (valp-val) / (vars[p]*EPS);
                    }
                }
                sumlog += val;
            }
            std::cout << sumlog << "\n";
        }
        catch(std::exception& e) {
            std::cout << "Exception "<<e.what()<<"\n";
            if(values)
                values[0] = NAN;
            if(derivs)
                derivs[0] = NAN;
        }
    }
    virtual unsigned int numVars() const { return NPARAMS; }
    virtual unsigned int numValues() const { return points.size(); }
private:
    const ActionArray& points;
};

/// compute log-likelihood of DF with given params against an array of points
double modelLikelihood(const df::DoublePowerLawParam& params, const ActionArray& points)
{
    std::cout << "J0="<<params.j0<<", Jcore="<<params.jcore<<
        ", alpha="<<params.alpha<<", ar="<<params.ar<<", az="<<params.az<<", aphi="<<params.aphi<<
        ", beta=" <<params.beta <</*", br="<<params.br<<", bz="<<params.bz<<", bphi="<<params.bphi<<*/": ";
    double sumlog = 0;
    try{
        df::DoublePowerLaw dpl(params);
        double norm = dpl.totalMass(1e-4, 100000);
        for(unsigned int i=0; i<points.size(); i++)
            sumlog += log(dpl.value(points[i])/norm);
        std::cout << "LogL="<<sumlog<<", norm="<<norm<< std::endl;
        return sumlog;
    }
    catch(std::invalid_argument& e) {
        std::cout << "Exception "<<e.what()<<"\n";
        return -1000.*points.size();
    }
}

/// function to be minimized
class ModelSearchFnc: public math::IFunctionNdim{
public:
    ModelSearchFnc(const ActionArray& _points) : points(_points) {};
    virtual void eval(const double vars[], double values[]) const
    {
        values[0] = -modelLikelihood(dfparams(vars), points);
    }
    virtual unsigned int numVars() const { return NPARAMS; }
    virtual unsigned int numValues() const { return 1; }
private:
    const ActionArray& points;
};

/// analytic expression for the ergodic distribution function f(E)
/// in a Hernquist model with mass M, scale radius a, at energy E.
double dfHernquist(double M, double a, double E)
{
    double q = sqrt(-E*a/M);
    return M / (4 * pow(2 * M * a * M_PI*M_PI, 1.5) ) * pow(1-q*q, -2.5) *
        (3*asin(q) + q * sqrt(1-q*q) * (1-2*q*q) * (8*q*q*q*q - 8*q*q - 3) );
}

/// create an N-body representation of Hernquist model
particles::PointMassArrayCyl createHernquistModel(double M, double a, unsigned int nbody)
{
    particles::PointMassArrayCyl points;
    for(unsigned int i=0; i<nbody; i++) {
        // 1. choose position
        double f = math::random();   // fraction of enclosed mass chosen at random
        double r = 1/(1/sqrt(f)-1);  // and converted to radius, using the known inversion of M(r)
        double costheta = math::random()*2 - 1;
        double sintheta = sqrt(1-pow_2(costheta));
        double phi = math::random()*2*M_PI;
        // 2. assign velocity
        double pot = -M/(r+a);
        double fmax = 0.006/(r*r);  // magic number
        double E, fE;
        do{ // rejection algorithm
            E = math::random() * pot;
            f = math::random() * fmax;
            fE= dfHernquist(M, a, E) * sqrt(E-pot);
            assert(fE<fmax);  // we must have selected a safe upper bound on f(E)*sqrt(E-Phi)
        } while(f > fE);
        double v = sqrt(2*(E-pot));
        double vcostheta = math::random()*2 - 1;
        double vsintheta = sqrt(1-pow_2(vcostheta));
        double vphi = math::random()*2*M_PI;
        points.add(coord::PosVelCyl(r*sintheta, r*costheta, phi,
            v*vsintheta*cos(vphi), v*vsintheta*sin(vphi), v*vcostheta), 1./nbody);
    }
    return points;
}

int main(){
    potential::PtrPotential pot(new potential::Dehnen(1., 1., 1., 1., 1.));
    const actions::ActionFinderAxisymFudge actf(pot);
    particles::PointMassArrayCyl particles(createHernquistModel(1., 1., 100000));
    ActionArray particleActions(particles.size());
    for(unsigned int i=0; i<particles.size(); i++)
        particleActions[i] = actf.actions(particles.point(i));

    // do a parameter search to find best-fit distribution function describing these particles
    const double initparams[NPARAMS] = {2.0, 4.0, 1.0, 1.0};
    const double stepsizes[NPARAMS]  = {0.1, 0.1, 0.1, 0.1};
    const int maxNumIter = 1000;
    const double toler   = 1e-4;
    double bestparams[NPARAMS];
    //ModelSearchFncLM fncLM(particleActions);
    //math::nonlinearMultiFit(fncLM, initparams, toler, maxNumIter, bestparams);
    ModelSearchFnc fnc(particleActions);
    int numIter = math::findMinNdim(fnc, initparams, stepsizes, toler, maxNumIter, bestparams);
    std::cout << numIter << " iterations\n";

    return 0;
}
