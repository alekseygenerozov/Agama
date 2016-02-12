#include "actions_newtorus.h"
#include "actions_isochrone.h"
#include "actions_genfnc.h"
#include "math_core.h"
#include "math_fit.h"
#include <stdexcept>
#include <cmath>

#include <iostream>

namespace actions{

namespace {  // internal routines

/** Interface for toy map used during torus fitting.
    It computes the position and velocity from the given values of toy actions and angles
    and the given array of toy map parameters (possibly scaled in some way),
    and also provides the derivatives of pos/vel w.r.t. actions and toy map params.
    Unlike the ordinary BaseToyMap class, it does not fix the parameters of toy map
    at construction, but rather takes them at each invocation, because they change
    in the process of least-square fitting.
*/
class BaseToyMapFit{
public:
    virtual ~BaseToyMapFit() {}

    /** number of free parameters in the toy map to be optimized during the fit */
    virtual unsigned int numParams() const = 0;

    /** convert toy actions/angles into position/velocity for the given parameters of toy map.
        \param[in]  actAng are the toy actions and angles;
        \param[in]  params are the (scaled) parameters of toy map;
        \param[out] derivAct if not NULL, return derivatives of pos/vel w.r.t. toy actions;
        \param[out] derivParam if not NULL, return derivatives of pos/vel w.r.t. scaled
        parameters of toy map (must point to an existing array with numParams elements);
        \return  position and velocity point.
    */
    virtual coord::PosVelCyl mapDeriv(
        const ActionAngles& actAng,
        const double params[],
        DerivAct* derivAct=0,
        coord::PosVelCyl* derivParam=0) const = 0;

    /** create an instance of ordinary ToyMap from the provided (scaled) parameters;
        this should be used after the torus fitting procedure finishes and returns
        the optimal params, which then will be used for performing the torus mapping itself.
    */
    virtual PtrToyMap create(const double params[]) const = 0;
};

/** Specialization of toy map for the case of Isochrone */
class ToyMapFitIsochrone: public BaseToyMapFit{
public:
    virtual unsigned int numParams() const { return 2; }
    virtual coord::PosVelCyl mapDeriv(
        const ActionAngles& actAng,
        const double params[],
        DerivAct* derivAct=0,
        coord::PosVelCyl* derivParam=0) const
    {
        // input parameters are log-scaled
        double M = exp(params[0]), b = exp(params[1]);
        coord::PosVelCyl result = ToyMapIsochrone(M, b).mapDeriv(
            actAng, NULL, derivAct, NULL, derivParam);
        if(derivParam) {  // multiply derivs by additional factor for converting d/dP to d/dln(P)
            derivParam[0].R   *= M;  derivParam[1].R   *= b;
            derivParam[0].z   *= M;  derivParam[1].z   *= b;
            derivParam[0].phi *= M;  derivParam[1].phi *= b;
            derivParam[0].vR  *= M;  derivParam[1].vR  *= b;
            derivParam[0].vz  *= M;  derivParam[1].vz  *= b;
            derivParam[0].vphi*= M;  derivParam[1].vphi*= b;
        }
        return result;
    }
    virtual PtrToyMap create(const double params[]) const
    {
        return PtrToyMap(new ToyMapIsochrone(exp(params[0]), exp(params[1])));
    }
};

/** compute the derivative of Hamiltonian by toy actions:
    dH/dJ = dH/d{x,v} d{x,v}/dJ, where the lhs is a covector of length 3,
    the first term on rhs is a covector of length 6 (the gradient dPhi/dx and the velocity),
    and the second term is a 6x3 matrix of partial derivs provided by the toy map.
*/
static inline Actions dHbydJ(
    const coord::PosVelCyl& vel, const coord::GradCyl& grad, const DerivAct& derivAct)
{
    return Actions(
        derivAct.dbyJr.R * grad.dR + derivAct.dbyJr.z * grad.dz + derivAct.dbyJr.phi * grad.dphi +
        derivAct.dbyJr.vR * vel.vR + derivAct.dbyJr.vz * vel.vz + derivAct.dbyJr.vphi * vel.vphi,
        derivAct.dbyJz.R * grad.dR + derivAct.dbyJz.z * grad.dz + derivAct.dbyJz.phi * grad.dphi +
        derivAct.dbyJz.vR * vel.vR + derivAct.dbyJz.vz * vel.vz + derivAct.dbyJz.vphi * vel.vphi,
        derivAct.dbyJphi.R * grad.dR + derivAct.dbyJphi.z * grad.dz + derivAct.dbyJphi.phi * grad.dphi +
        derivAct.dbyJphi.vR * vel.vR + derivAct.dbyJphi.vz * vel.vz + derivAct.dbyJphi.vphi * vel.vphi);
}

/** Function to be provided to Levenberg-Marquardt minimization routine:
    compute the difference between values of Hamiltonian `H_k` evaluated in the target potential
    at points `k` provided by toy map, and the average Hamiltonian `<H>` over all points
    (entire range of toy angles).
    For each point it returns the relative difference between `H_k` and `<H>`, and optionally
    computes the derivatives of `H_k` w.r.t. the parameters of toy map and generating function.
    It also provides the method for computing the coefficients of angle map.
*/
class FitMappingParams: public math::IFunctionNdimDeriv {
private:
    const BaseToyMapFit& toyMap;         ///< map between toy action/angles and position/velocity
    const GenFncFit& genFnc;             ///< map between real and toy actions at a grid of angles
    const potential::BasePotential& pot; ///< real potential
    const unsigned int numParamsToyMap;  ///< # of params of toy potential to fit
    const unsigned int numParamsGenFnc;  ///< # of terms in generating function to fit
    const unsigned int numParams;        ///< total # of parameters to fit
    const unsigned int numPoints;        ///< number of points in the grid of toy angles
public:
    FitMappingParams(const BaseToyMapFit& _toyMap,
        const GenFncFit& _genFnc, const potential::BasePotential& _pot) :
        toyMap(_toyMap), genFnc(_genFnc), pot(_pot),
        numParamsToyMap(toyMap.numParams()),
        numParamsGenFnc(genFnc.numParams()),
        numParams(numParamsToyMap + numParamsGenFnc),
        numPoints(genFnc.numPoints()) {}
    virtual unsigned int numVars() const { return numParams; }
    virtual unsigned int numValues() const { return numPoints; }

    /** Compute the deviations of Hamiltonian from its average value for an array of points
        with the provided parameters of toy map and coefficients of generating function,
        given as a single array of input variables.
        Instances of a toy map and a generating function with these parameters are created
        and used to map each element of toy angle array to {x,v}, which are then used to
        compute the actual value of Hamiltonian at each point H(theta_k).
        The output values are the differences between H(theta_k) and <H>,
        and the output derivatives is the Jacobian matrix of derivatives of H(theta_k)
        w.r.t. input parameters of toy map and gen.fnc.
        This method is called from the Levenberg-Marquardt solver.
    */
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        // we need to store the values of H_k even if this is not requested,
        // because they are used to correct the entries of the Jacobian matrix
        // to account for the fact that the mean <H> also depends on the parameters
        std::vector<double> tmpvalues(numPoints);
        if(!values)
            values = &tmpvalues.front();
        // temp.storage for derivs of position/velocity by parameters of toy map: d(x,v)/dP
        std::vector<coord::PosVelCyl> derivParam(numParamsToyMap);
        // accumulators for the average Hamiltonian and its derivs by all parameters
        double Havg = 0;
        std::vector<double> dHavgdP(numPoints);

        // loop over grid of toy angles
        for(unsigned int k=0; k<numPoints; k++) {
            // Generating function computes the toy actions from the real actions
            // at the given point in the grid of toy angles grid, for the given parameters
            // (provided in the array of input vars right after the parameters of toy map)
            ActionAngles toyAA = genFnc.toyActionAngles(k, vars+numParamsToyMap);

            // Toy map computes the position and velocity from the toy actions and angles,
            // and optionally their derivatives w.r.t. toy actions and toy map parameters,
            // for the given toy map parameters provided by the first numParamsToyMap
            // elements of array of input vars
            DerivAct derivAct;
            coord::PosVelCyl point = toyMap.mapDeriv(toyAA, vars,
                derivs && numParamsGenFnc>0 ? &derivAct : NULL,
                derivs ? &derivParam.front() : NULL);

            // obtain the value and gradient of real potential at the given point
            coord::GradCyl grad;
            double H;
            pot.eval(point, &H, derivs ? &grad : NULL);
            H += 0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));

            // accumulate the average value and store the output
            Havg += H;
            values[k] = H;
            
            // compute the derivatives of H at the given toy angles w.r.t. input params
            if(derivs) {
                for(unsigned int p = 0; p<numParamsToyMap; p++) {
                    // derivative of Hamiltonian w.r.t. p-th parameter of toy map
                    double dHdP =
                    grad.dR   * derivParam[p].R   +
                    grad.dz   * derivParam[p].z   +
                    grad.dphi * derivParam[p].phi +
                    point.vR  * derivParam[p].vR  +
                    point.vz  * derivParam[p].vz  +
                    point.vphi* derivParam[p].vphi;
                    derivs[k * numParams + p] = dHdP;
                    dHavgdP[p] += dHdP;
                }
                if(numParamsGenFnc>0) {
                    // derivative of Hamiltonian by toy actions
                    Actions dHby = dHbydJ(point, grad, derivAct);
                    for(unsigned int p = numParamsToyMap; p<numParams; p++) {
                        // derivs of toy actions by gen.fnc.params
                        Actions dbyS = genFnc.deriv(k, p - numParamsToyMap);
                        // derivs of Hamiltonian by gen.fnc.params
                        double  dHdS = dHby.Jr * dbyS.Jr + dHby.Jz * dbyS.Jz + dHby.Jphi * dbyS.Jphi;
                        derivs[k * numParams + p] = dHdS;
                        dHavgdP[p] += dHdS;
                    }
                }
            }
        }
        // convert from  H_k  to  deltaH_k = H_k / <H> - 1
        Havg /= numPoints;
        double disp = 0;
        for(unsigned int k=0; k<numPoints; k++) {
            values[k] = values[k] / Havg - 1;;
            disp += pow_2(values[k]);
        }
        // convert derivatives:  d(deltaH_k) / dP_p = (1/<H>) dH_k / dP_p - (H_k / <H>^2) d<H> / dP_p
        if(derivs) {
            for(unsigned int p=0; p<numParams; p++)
                dHavgdP[p] /= numPoints;  // now contains d<H> / dP_p
            for(unsigned int k=0; k<numPoints; k++)
                for(unsigned int p=0; p<numParams; p++) {
                    unsigned int index = k * numParams + p;  // index in 2d Jacobian matrix
                    derivs[index] = (derivs[index] - (values[k]+1) * dHavgdP[p]) / Havg;
                }
        }
        std::cout << "M="<<exp(vars[0])<<", b="<<exp(vars[1])<<
        "; Havg="<<Havg<<", DeltaH="<<sqrt(disp/numPoints)<<"\n";
    }

    /** Compute the derivatives of generating function by real actions, used in angle mapping.
        The three arrays of derivatives dS_i/dJ_{r,z,phi}, i=0..numParamsGenFnc-1,
        together with three frequencies Omega_{r,z,phi}, are the solutions of
        an overdetermined system of linear equations:
        \f$  M_{k,i} X_{i} = RHS_{k}, k=0..numPoints-1  \f$,
        where numPoints is the number of individual triplets of toy angles,
        \f$  X_i  \f$ is the solution vector {Omega, dS_i/dJ} for each direction (r,z,theta),
        \f$  RHS_k = dH/dJ(\theta_k)  \f$, again for three directions independently, and
        \f$  M_{k,i}  \f$ is the matrix of coefficients shared between all three equation systems:
        \f$  M_{k,0} = 1, M_{k,i+1} = -dH/dS_i(\theta_k)  \f$.
        The matrix M and three RHS vectors are filled using the same approach as during
        the Levenberg-Marquardt minimization, from the provided parameters of toy map and
        generating function (in the same order as for `evalDeriv` or as returned by
        `math::nonlinearFitNdim`); then the three linear systems are solved using
        the singular-value decomposition of the shared coefficient matrix,
        and the output frequencies and gen.fnc.derivatives are returned.
    */
    void fitAngleMap(const double vars[],
        actions::Frequencies& freqs, actions::GenFncDerivs& derivs) const
    {
        // the matrix of coefficients shared between three linear systems
        math::Matrix<double> coefsdHdS(numPoints, numParamsGenFnc+1);
        // derivs of Hamiltonian by toy actions (RHS vectors)
        std::vector <double> dHdJr(numPoints), dHdJz(numPoints), dHdJphi(numPoints);
        // loop over grid of toy angles
        for(unsigned int k=0; k<numPoints; k++) {
            ActionAngles toyAA = genFnc.toyActionAngles(k, vars+numParamsToyMap);
            DerivAct derivAct;
            coord::PosVelCyl point = toyMap.mapDeriv(toyAA, vars, &derivAct);
            // obtain the gradient of real potential at the given point
            coord::GradCyl grad;
            pot.eval(point, NULL, &grad);
            // derivative of Hamiltonian by toy actions
            Actions dHby = dHbydJ(point, grad, derivAct);
            // fill the elements of each of three rhs vectors
            dHdJr  [k] = dHby.Jr;
            dHdJz  [k] = dHby.Jz;
            dHdJphi[k] = dHby.Jphi;
            // fill the k-th matrix row
            coefsdHdS(k, 0) = 1;  // matrix coef for omega
            for(unsigned int p=0; p<numParamsGenFnc; p++) {
                // derivs of toy actions by gen.fnc.params
                Actions dbyS = genFnc.deriv(k, p);
                // derivs of Hamiltonian by gen.fnc.params
                double  dHdS = dHby.Jr * dbyS.Jr + dHby.Jz * dbyS.Jz + dHby.Jphi * dbyS.Jphi;
                coefsdHdS(k, p+1) = -dHdS;  // matrix coef for dS_p/dJ
            }
        }
        // solve the overdetermined linear system in the least-square sense:
        // step 1: prepare the SVD of coefs matrix
        math::Matrix<double> tmpV;
        std::vector <double> tmpSV, dSdJr, dSdJz, dSdJphi;
        math::singularValueDecomp(coefsdHdS, tmpV, tmpSV);
        // step 2: solve three linear systems with the same matrix but different rhs
        math::linearSystemSolveSVD(coefsdHdS, tmpV, tmpSV, dHdJr, dSdJr);
        math::linearSystemSolveSVD(coefsdHdS, tmpV, tmpSV, dHdJz, dSdJz);
        math::linearSystemSolveSVD(coefsdHdS, tmpV, tmpSV, dHdJphi, dSdJphi);
        freqs.Omegar   = dSdJr[0];
        freqs.Omegaz   = dSdJz[0];
        freqs.Omegaphi = dSdJphi[0];
        derivs.resize(numParamsGenFnc);
        for(unsigned int p=0; p<numParamsGenFnc; p++) {
            derivs[p].Jr   = dSdJr[p+1];
            derivs[p].Jz   = dSdJz[p+1];
            derivs[p].Jphi = dSdJphi[p+1];
        }
    }
};

/// create an array of angles uniformly covering the range [0:pi)  (NB: why not 2pi?)
static std::vector<Angles> makeGridAngles(unsigned int nr, unsigned int nz, unsigned int nphi=1)
{
    std::vector<Angles> vec(nr*nz*nphi);
    for(unsigned int ir=0; ir<nr; ir++) {
        double thetar = ir * M_PI / nr;
        for(unsigned int iz=0; iz<nz; iz++) {
            double thetaz = iz * M_PI / nz;
            for(unsigned int iphi=0; iphi<nphi; iphi++)
                vec[ (ir*nz + iz) * nphi + iphi ] = Angles(thetar, thetaz, iphi * M_PI / nphi);
        }
    }
    return vec;
}

/// create the array of indices of the generating function with some default terms
static GenFncIndices makeDefaultIndices()
{
    GenFncIndices indices;
    indices.push_back(GenFncIndex(0,-4, 0));
    indices.push_back(GenFncIndex(0,-2, 0));
    indices.push_back(GenFncIndex(1,-2, 0));
    indices.push_back(GenFncIndex(1, 0, 0));
    indices.push_back(GenFncIndex(1, 2, 0));
    indices.push_back(GenFncIndex(2,-2, 0));
    indices.push_back(GenFncIndex(2, 0, 0));
    indices.push_back(GenFncIndex(2, 2, 0));
    indices.push_back(GenFncIndex(3, 0, 0));
    return indices;
}

/// create the array of indices of the generating function with all terms up to the given maximum order
static GenFncIndices makeGridIndices(int irmax, int izmax)
{
    GenFncIndices indices;
    for(int ir=0; ir<=irmax; ir++)
        for(int iz=-izmax; iz<=(ir==0?-2:izmax); iz+=2)
            indices.push_back(GenFncIndex(ir, iz, 0));
    return indices;
}

}  // internal namespace

ActionMapperNewTorus::ActionMapperNewTorus(const potential::BasePotential& pot, const Actions& _acts) :
    acts(_acts)
{
    if(!isAxisymmetric(pot))
        throw std::invalid_argument("ActionMapperNewTorus only works for axisymmetric potentials");
    // number of iterations in Levenberg-Marquardt algorithm
    const unsigned int maxNumIter = 20;
    // stopping criterion for LM fit
    const double relToler = 0.01;
    // grid in toy angles
    std::vector<Angles> angs = makeGridAngles(12, 12);
    // indices of non-trivial terms of generating function
    GenFncIndices indices = makeGridIndices(8, 8); //makeDefaultIndices();
    // instance of generating function to be used during the fit
    GenFncFit genFncFit(indices, acts, angs);
    // instance of toy map to be used during the fit
    ToyMapFitIsochrone toyMapFit;
    // routine that computes the scatter in Hamiltonian among the points in angle grid
    FitMappingParams fnc(toyMapFit, genFncFit, pot);
    // initial values of parameters of toy map and generating function
    std::vector<double> fitParams(fnc.numVars(), 0.);
    // perform the Levenberg-Marquardt minimization and store best-fit parameters in fitParams
    math::nonlinearMultiFit(fnc, &fitParams[0], relToler, maxNumIter, &fitParams[0]);
    // compute dS/dJ used in the angle map
    GenFncDerivs derivs;
    fnc.fitAngleMap(&fitParams[0], freqs, derivs);
    // create the toy map with best-fit params
    toyMap = toyMapFit.create(&fitParams[0]);
    // create the generating function with best-fit params
    genFnc = PtrCanonicalMap(
        new GenFnc(indices, &fitParams[toyMapFit.numParams()], &derivs[0]));
}

coord::PosVelCyl ActionMapperNewTorus::map(const ActionAngles& actAng, Frequencies* freq) const
{
    // make sure that the input actions are the same as in the Torus object
    if( math::fcmp(actAng.Jr,   acts.Jr) != 0 ||
        math::fcmp(actAng.Jz,   acts.Jz) != 0 ||
        math::fcmp(actAng.Jphi, acts.Jphi) != 0 )
        throw std::invalid_argument("ActionMapperNewTorus: "
            "values of actions are different from those provided to the constructor");
    if(freq)
        *freq = freqs;
    return toyMap->map(genFnc->map(actAng));
}

}  // namespace actions
