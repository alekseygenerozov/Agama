/** \file    test_df_fit.cpp
    \author  Eugene Vasiliev
    \date    Aug-Nov 2015

    This example demonstrates how to find best-fit parameters of an action-based
    distribution function that matches the given N-body snapshot.

    The N-body model itself corresponds to an axisymmetric density profile generated
    self-consistently from a double-power-law distribution function (Posti et al.2015),
    with the `test_selfconsistentmodel` program.
    The test is supposed to recover the parameters of DF used to create the N-body model
    (we fit it with the same functional form but unknown a priori parameters).
    We compute the potential from the N-body model itself, and compute actions
    for all particles only once.
    Then we scan the parameter space of DF, finding the maximum of the likelihood
    function with a multidimensional minimization algorithm.
    This takes a few hundred iterations to converge.
*/
#include "potential_sphharm.h"
#include "actions_staeckel.h"
#include "df_halo.h"
#include "particles_io.h"
#include "math_fit.h"
#include "math_core.h"
#include "debug_utils.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

typedef std::vector<actions::Actions> ActionArray;

/// compute log-likelihood of DF with given params against an array of points
double modelLikelihood(const df::DoublePowerLawParam& params, const ActionArray& points)
{
    std::cout << "J0="<<params.j0<< //", Jcore="<<params.jcore<<
        ", alpha="<<params.alpha<<", ar="<<params.ar<<", az="<<params.az<<", aphi="<<params.aphi<<
        ", beta=" <<params.beta <<", br="<<params.br<<", bz="<<params.bz<<", bphi="<<params.bphi<<": ";
    double sumlog = 0;
    try{
        df::DoublePowerLaw dpl(params);
        double norm = dpl.totalMass(1e-4, 100000);
        for(unsigned int i=0; i<points.size(); i++) {
            double val = dpl.value(points[i]);
            sumlog += log(val/norm);
        }
        std::cout << "LogL="<<sumlog<<", norm="<<norm<< std::endl;
        return sumlog;
    }
    catch(std::invalid_argument& e) {
        std::cout << "Exception "<<e.what()<<"\n";
        return -1000.*points.size();
    }
}

/// convert from parameter space to DF params: note that we apply
/// some non-trivial scaling to make the life easier for the minimizer
df::DoublePowerLawParam dfparams(const double vars[])
{
    df::DoublePowerLawParam params;
    params.jcore = 0;
    params.alpha = vars[0];
    params.beta  = vars[1];
    params.j0    = vars[2];
    params.ar    = 3./(1+vars[3]+vars[4])*vars[3];
    params.az    = 3./(1+vars[3]+vars[4])*vars[4];
    params.aphi  = 3./(1+vars[3]+vars[4]);  // ar+az+aphi = 3
    params.br    = 3./(1+vars[5]+vars[6])*vars[5];
    params.bz    = 3./(1+vars[5]+vars[6])*vars[6];
    params.bphi  = 3./(1+vars[5]+vars[6]);
    params.norm  = 1.;
    return params;
}

/// function to be minimized
class ModelSearchFnc: public math::IFunctionNdim{
public:
    ModelSearchFnc(const ActionArray& _points) : points(_points) {};
    virtual void eval(const double vars[], double values[]) const
    {
        values[0] = -modelLikelihood(dfparams(vars), points);
    }
    virtual unsigned int numVars() const { return 7; }
    virtual unsigned int numValues() const { return 1; }
private:
    const ActionArray& points;
};

int main(){
    particles::PointMassArrayCar particles;
    readSnapshot("sampled_model.txt", units::ExternalUnits(), particles);
    potential::PtrPotential pot(new potential::SplineExp(20, 4, particles, coord::ST_AXISYMMETRIC, 1.));
    const actions::ActionFinderAxisymFudge actf(pot);
    ActionArray particleActions(particles.size());
    for(unsigned int i=0; i<particles.size(); i++) {
        particleActions[i] = actf.actions(toPosVelCyl(particles.point(i)));
        if(!math::isFinite(particleActions[i].Jr+particleActions[i].Jz)) {
            std::cout << particleActions[i] <<" for "<< particles.point(i) <<"\n";
            particleActions[i] = particleActions[i>0?i-1:0];  // put something reasonable (unless i==0)
        }
    }

    // do a parameter search to find best-fit distribution function describing these particles
    ModelSearchFnc fnc(particleActions);
    const double initparams[] = {2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const double stepsizes[]  = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    const int maxNumIter = 1000;
    const double toler   = 1e-4;
    double bestparams[7];
    int numIter = math::findMinNdim(fnc, initparams, stepsizes, toler, maxNumIter, bestparams);
    std::cout << numIter << " iterations\n";

    return 0;
}
