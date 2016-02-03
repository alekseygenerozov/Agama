#include "actions_newtorus.h"
#include "actions_isochrone.h"
#include "math_core.h"
#include "math_fit.h"
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace actions{

/// create an array of angles uniformly covering the range [0:2pi)
static std::vector<Angles> makeGridAngles(unsigned int nr, unsigned int nz, unsigned int nphi=1)
{
    std::vector<Angles> vec(nr*nz*nphi);
    for(unsigned int ir=0; ir<nr; ir++) {
        double thetar = ir * 2*M_PI / nr;
        for(unsigned int iz=0; iz<nz; iz++) {
            double thetaz = iz * 2*M_PI / nz;
            for(unsigned int iphi=0; iphi<nphi; iphi++)
                vec[ (ir*nz + iz) * nphi + iphi ] = Angles(thetar, thetaz, iphi * 2*M_PI / nphi);
        }
    }
    return vec;
}

/** Function to be provided to Levenberg-Marquardt minimization routine:
    compute the difference between values of Hamiltonian `H_k` evaluated in the target potential
    at points `k` provided by toy map, and the average Hamiltonian `<H>` over all points
    (entire range of toy angles).
    For each point it returns the relative difference between `H_k` and `<H>`, and optionally
    computes the derivatives of `H_k` w.r.t. the parameters of toy map.
*/
class FitMappingParams: public math::IFunctionNdimDeriv {
public:
    FitMappingParams(const Actions& _acts, const std::vector<Angles>& _angs,
        const potential::BasePotential& _pot) :
        acts(_acts), angs(_angs), pot(_pot) {}

    /// compute the deviations of Hamiltonian from its average value for an array of points
    /// with the provided (scaled) parameters of toy map
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        const unsigned int nPoints = angs.size();
        const double EPS=1e-8;
        // create a toy map with the provided (un-scaled) parameters
        const double M = exp(vars[0]), b = exp(vars[1]);
        ToyMapIsochrone toyMap(M, b),
        toyMap1(M*(1+EPS), b), toyMap2(M, b*(1+EPS));
        std::vector<double> tmpvalues;
        if(!values) {  // we need to store the values of H_k even if this is not requested
            tmpvalues.resize(nPoints);
            values = &tmpvalues.front();
        }
        std::vector<coord::PosVelCyl> derivParam(toyMap.numParams());  // d(x,v)/dP
        std::vector<double> dHavgdP(numVars());
        double Havg = 0;
        for(unsigned int k=0; k<nPoints; k++) {
            coord::PosVelCyl point = toyMap.mapDeriv(ActionAngles(acts, angs[k]),
                NULL, NULL, NULL, derivs ? &derivParam.front() : NULL);
            coord::GradCyl grad;
            double H;
            pot.eval(point, &H, derivs ? &grad : NULL);
            H += 0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
            Havg += H;
            values[k] = H;
            if(derivs) {
#if 0
                for(unsigned int p=0; p<toyMap.numParams(); p++) {
                    double dHdP =
                    grad.dR    * derivParam[p].R +
                    grad.dz    * derivParam[p].z + // skip d/dphi for an axisymmetric potential
                    point.vR   * derivParam[p].vR +
                    point.vz   * derivParam[p].vz +
                    point.vphi * derivParam[p].vphi;
                    derivs[k * numVars() + p] = dHdP;
                    dHavgdP[p] += dHdP;
                }
#else
                // a naive way to compute derivatives by finite difference
                double dH1 = (totalEnergy(pot, toyMap1.map(ActionAngles(acts, angs[k]))) - H) / EPS;
                double dH2 = (totalEnergy(pot, toyMap2.map(ActionAngles(acts, angs[k]))) - H) / EPS;
                derivs[k * 2] = dH1;
                derivs[k*2+1] = dH2;
                dHavgdP[0] += dH1;
                dHavgdP[1] += dH2;
#endif
            }
        }
        // convert from  H_k  to  deltaH_k = H_k / <H> - 1
        Havg /= nPoints;
        double disp = 0;
        for(unsigned int k=0; k<nPoints; k++) {
            values[k] = values[k] / Havg - 1;;
            disp += pow_2(values[k]);
        }
        std::cout << "M="<<M<<", b="<<b<<
        "; Havg="<<Havg<<", DeltaH="<<sqrt(disp/nPoints)<<"\n";
        // convert derivatives:  d(deltaH_k) / dP_p = (1/<H>) dH_k / dP_p - (H_k / <H>^2) d<H> / dP_p
        if(derivs) {
            for(unsigned int p=0; p<numVars(); p++)
                dHavgdP[p] /= nPoints;  // now contains d<H> / dP_p
            for(unsigned int k=0; k<nPoints; k++)
                for(unsigned int p=0; p<numVars(); p++) {
                    unsigned int index = k * numVars() + p;  // index in 2d Jacobian matrix
                    derivs[index] = (derivs[index] - (values[k]+1) * dHavgdP[p]) / Havg;
                }
        }
    }
    virtual unsigned int numVars() const { return ToyMapIsochrone::nParams; }
    virtual unsigned int numValues() const { return angs.size(); }
private:
    const Actions acts;                  ///< values of real actions
    const std::vector<Angles> angs;      ///< grid of toy angles
    const potential::BasePotential& pot; ///< real potential
};

ActionMapperNewTorus::ActionMapperNewTorus(const potential::BasePotential& pot, const Actions& _acts) :
    acts(_acts)
{
    if(!isAxisymmetric(pot))
        throw std::invalid_argument("ActionMapperNewTorus only works for axisymmetric potentials");
    const unsigned int maxNumIter = 10;   // number of iterations in Levenberg-Marquardt algorithm
    const double relToler = 0.01;
    std::vector<Angles> angs = makeGridAngles(10, 10);
    std::vector<double> toyMapParams(ToyMapIsochrone::nParams);
    FitMappingParams fnc(acts, angs, pot);
    //unsigned int numIter =
    math::nonlinearMultiFit(fnc, &toyMapParams[0], relToler, maxNumIter, &toyMapParams[0]);
    toyMap = PtrToyMap(new ToyMapIsochrone(toyMapParams[0], toyMapParams[1]));
}

coord::PosVelCyl ActionMapperNewTorus::map(const ActionAngles& actAng, Frequencies* freq) const
{
    // make sure that the input actions are the same as in the Torus object
    if( math::fcmp(actAng.Jr,   acts.Jr) != 0 ||
        math::fcmp(actAng.Jz,   acts.Jz) != 0 ||
        math::fcmp(actAng.Jphi, acts.Jphi) != 0 )
        throw std::invalid_argument("ActionMapperNewTorus: "
            "values of actions are different from those provided to the constructor");
    return toyMap->map(actAng, freq);
}

}  // namespace actions
