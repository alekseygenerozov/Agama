#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_spline.h"
#include "math_specfunc.h"
#include "math_sphharm.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential {

// internal definitions
namespace{

const unsigned int CYLSPLINE_MIN_GRID_SIZE = 4;
const unsigned int CYLSPLINE_MAX_GRID_SIZE = 1024;
const unsigned int CYLSPLINE_MAX_ANGULAR_HARMONIC = 12;

/// minimum number of terms in Fourier expansion used to compute coefficients
/// of a non-axisymmetric density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)
const unsigned int MMIN_AZIMUTHAL_FOURIER = 16;

/// maximum allowed order of azimuthal Fourier expansion
const unsigned int MMAX_AZIMUTHAL_FOURIER = 64;

/// lower cutoff in radius for Legendre Q function
const double MIN_R = 1e-10;

/// max number of function evaluations in multidimensional integration
const unsigned int MAX_NUM_EVAL = 10000;

/// relative accuracy of potential computation (integration tolerance parameter)
const double EPSREL_POTENTIAL_INT = 1e-6;

// ------- Fourier expansion of density or potential ------- //
// The routine 'computeFourierCoefs' can work with both density and potential classes,
// computes the azimuthal Fourier expansion for either density (in the first case),
// or potential and its R- and z-derivatives (in the second case).
// To avoid code duplication, the function that actually retrieves the relevant quantity
// is separated into a dedicated routine 'storeValue', which stores either one or three
// values for each input point. The 'computeFourierCoefs' routine is templated on both
// the type of input data and the number of quantities stored for each point.

template<class BaseDensityOrPotential>
void storeValue(const BaseDensityOrPotential& src, const coord::PosCyl& pos, double values[]);

template<>
inline void storeValue(const BaseDensity& src, const coord::PosCyl& pos, double values[]) {
    *values = src.density(pos);
}

template<>
inline void storeValue(const BasePotential& src, const coord::PosCyl& pos, double values[]) {
    coord::GradCyl grad;
    src.eval(pos, values, &grad);
    values[(2*MMAX_AZIMUTHAL_FOURIER+1)]   = grad.dR;
    values[(2*MMAX_AZIMUTHAL_FOURIER+1)*2] = grad.dz;
}

template<class BaseDensityOrPotential, int NQuantities>
void computeFourierCoefs(const BaseDensityOrPotential &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> >* coefs[])
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR<2 || sizez<2)
        throw std::invalid_argument("computeFourierCoefs: incorrect grid size");
    if(mmax > MMAX_AZIMUTHAL_FOURIER)
        throw std::invalid_argument("computeFourierCoefs: mmax is too large");
    if(!isZReflSymmetric(src) && gridz[0]==0)
        throw std::invalid_argument("computeFourierCoefs: input density is not symmetric "
            "under z-reflection, the grid in z must cover both positive and negative z");
    int mmin = isYReflSymmetric(src) ? 0 : -1*mmax;
    bool useSine = mmin<0;
    math::FourierTransformForward trans(mmax, useSine);
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, src.symmetry());
    unsigned int numHarmonicsComputed = indices.size();
    int numPoints = sizeR * sizez;
    for(int q=0; q<NQuantities; q++) {
        coefs[q]->resize(mmax*2+1);
        for(unsigned int i=0; i<numHarmonicsComputed; i++)
            coefs[q]->at(indices[i]+mmax).resize(sizeR, sizez);
    }
    std::string errorMsg;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numPoints; n++) {
        int iR = n % sizeR;  // index in radial grid
        int iz = n / sizeR;  // index in vertical direction
        double values [(2*MMAX_AZIMUTHAL_FOURIER+1) * NQuantities];
        double coefs_m[(2*MMAX_AZIMUTHAL_FOURIER+1)];
        try{
            for(unsigned int i=0; i<trans.size(); i++)
                storeValue<BaseDensityOrPotential>(src,
                    coord::PosCyl(gridR[iR], gridz[iz], trans.phi(i)), values+i);
            for(int q=0; q<NQuantities; q++) {
                trans.transform(&values[q*(2*MMAX_AZIMUTHAL_FOURIER+1)], coefs_m);
                for(unsigned int i=0; i<numHarmonicsComputed; i++) {
                    int m = indices[i];
                    coefs[q]->at(m+mmax)(iR, iz) =
                        iR==0 && m!=0 ? 0 :   // at R=0, all non-axisymmetric harmonics must vanish
                        coefs_m[useSine ? m+mmax : m] / (m==0 ? 2*M_PI : M_PI);
                }
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computeFourierCoefs: "+errorMsg);
}

// ------- Computation of potential from density ------- //
// The routines below solve the Poisson equation by computing the Fourier harmonics
// of potential via direct 2d integration over (R,z) plane. 
// If the input density is axisymmetric, then the value of density at phi=0 is taken,
// otherwise the density must first be Fourier-transformed itself and represented
// as an instance of DensityAzimuthalHarmonic class, which provides the member function
// returning the value of m-th harmonic at the given (R,z).

inline double density_rho_m(const BaseDensity& dens, int m, double R, double z) {
    if(dens.name() == DensityAzimuthalHarmonic::myName())
        return static_cast<const DensityAzimuthalHarmonic&>(dens).rho_m(m, R, z);
    return m==0 ? dens.density(coord::PosCyl(R, z, 0)) : 0;
}

// Routine that computes the contribution tothe  m-th harmonic potential at location (R0,z0)
// from the point at (R,z) with given 'mass' (or, rather, mass times trig(m phi)
// in the discrete case, or density times jacobian times trig(m phi) in the continuous case).
// This routine is used both in AzimuthalHarmonicIntegrand to compute the potential from
// a continuous density distribution, and in ComputePotentialCoefsFromPoints to obtain
// the potential from a discrete point mass collection.
inline void computePotentialHarmonicAtPoint(int m, double R, double z, double R0, double z0,
    double mass, bool useDerivs, double values[])
{
    // the contribution to the potential is given by
    // rho * \int_0^\infty dk J_m(k R) J_m(k R0) exp(-k|z-z0|)
    double t = R*R + R0*R0 + pow_2(z0-z);
    if(R > MIN_R && R0 > MIN_R) {  // normal case
        double sq = 1 / (M_PI * sqrt(R*R0));
        double u  = t / (2*R*R0);
        if(u < 1+MIN_R)
            return;  // close to singularity
        double dQ;
        double Q  = math::legendreQ(m-0.5, u, useDerivs ? &dQ : NULL);
        values[0]+= -sq * mass * Q;
        if(useDerivs) {
            values[1]+= -sq * mass * (dQ/R - (Q/2 + u*dQ)/R0);
            values[2]+= -sq * mass * dQ * (z0-z) / (R*R0);
        }
    } else      // degenerate case
    if(m==0) {  // here only m=0 harmonic survives
        if(t < 1e-15)
            return;    // close to singularity
        double s  = 1/sqrt(t);
        values[0]+= -mass * s;
        if(useDerivs)
            values[2]+=  mass * s * (z0-z) / t;
    }
}

// the N-dimensional integrand for computing the potential harmonics from density
class AzimuthalHarmonicIntegrand: public math::IFunctionNdim {
public:
    AzimuthalHarmonicIntegrand(const BaseDensity& _dens, int _m,
        double _R0, double _z0, bool _useDerivs) :
        dens(_dens), m(_m), R0(_R0), z0(_z0), useDerivs(_useDerivs) {}
    // evaluate the function at a given (R,z) point (scaled)
    virtual void eval(const double pos[], double values[]) const
    {
        for(unsigned int c=0; c<numValues(); c++)
            values[c] = 0;
        // unscale input coordinates
        const double s = pos[0];
        const double r = exp( 1/(1-s) - 1/s );
        if(!math::withinReasonableRange(r))
            return;  // scaled coords point at 0 or infinity
        const double th= pos[1] * M_PI/2;
        const double R = r*cos(th);
        const double z = r*sin(th);
        const double jac = pow_2(M_PI*r) * R * (1/pow_2(1-s) + 1/pow_2(s));
        
        // get the values of density at (R,z) and (R,-z):
        // here the density evaluation may be a computational bottleneck,
        // so in the typical case of z-reflection symmetry we save on using
        // the same value of density for both positive and negative z1.
        double rho = jac * density_rho_m(dens, m, R, z);
        computePotentialHarmonicAtPoint(m, R, z, R0, z0, rho, useDerivs, values);
        if(!isZReflSymmetric(dens))
               rho = jac * density_rho_m(dens, m, R,-z);
        computePotentialHarmonicAtPoint(m, R,-z, R0, z0, rho, useDerivs, values);

#ifdef HAVE_CUBA
        // workaround for CUBA n-dimensional quadrature routine: it seems to be unable 
        // to properly handle cases when one of components of the integrand is identically zero,
        // that's why we output 1 instead, and zero it out later
        if(R0==0)
            values[1] = 1;
        if(isZReflSymmetric(dens) && z0==0)
            values[2] = 1;
#endif
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return useDerivs ? 3 : 1; }
private:
    const BaseDensity& dens;
    const int m;
    // the point at which the integral is computed, also defines the toroidal coordinate system
    const double R0, z0;
    const bool useDerivs;
};

void computePotentialCoefsFromDensity(const BaseDensity &src,
    unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    bool useDerivs,
    std::vector< math::Matrix<double> >* output[])
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR == 0 || sizez == 0 || mmax > CYLSPLINE_MAX_ANGULAR_HARMONIC)
        throw std::invalid_argument("computePotentialCoefsCyl: invalid grid parameters");
    if(isZRotSymmetric(src))
        mmax = 0;
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, src.symmetry());
    unsigned int numQuantitiesOutput = useDerivs ? 3 : 1;  // Phi only, or Phi plus two derivs
    // the number of output coefficients is always a full set even if some of them are empty
    for(unsigned int q=0; q<numQuantitiesOutput; q++) {
        output[q]->resize(2*mmax+1);
        for(unsigned int i=0; i<indices.size(); i++)  // only allocate those coefs that will be used
            output[q]->at(indices[i]+mmax).resize(sizeR, sizez);
    }

    PtrDensity densInterp;  // pointer to an internally created interpolating object if it is needed
    const BaseDensity* dens = &src;  // pointer to either the original density or the interpolated one
    // For an axisymmetric potential we don't use interpolation,
    // as the Fourier expansion of density trivially has only one harmonic;
    // also, if the input density is already a Fourier expansion, use it directly.
    // Otherwise, we need to create a temporary DensityAzimuthalHarmonic interpolating object.
    if(!isZRotSymmetric(src) && src.name() != DensityAzimuthalHarmonic::myName()) {
        double Rmax = gridR.back() * 100;
        double Rmin = gridR[1] * 0.01;
        double zmax = gridz.back() * 100;
        double zmin = gridz[0]==0 ? gridz[1] * 0.01 :
            gridz[sizez/2]==0 ? gridz[sizez/2+1] * 0.01 : Rmin;
        double delta=0.1;  // relative difference between grid nodes = log(x[n+1]/x[n])
        // create a density interpolator; it will be automatically deleted upon return
        densInterp = DensityAzimuthalHarmonic::create(
            src, mmax, log(Rmax/Rmin)/delta, Rmin, Rmax, log(zmax/zmin)/delta, zmin, zmax);
        // and use it for computing the potential
        dens = densInterp.get();
    }

    int numPoints = sizeR * sizez;
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {  // combined index variable
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
        try{
            // integration box in scaled coords - r range is slightly smaller than 0:1
            // due to exponential scaling (rscaled=0.045 corresponds to r<1e-9)
            double Rzmin[2]={0.045,0.}, Rzmax[2]={0.955,1.};
            double result[3], error[3];
            int numEval;
            for(unsigned int i=0; i<indices.size(); i++) {
                int m = indices[i];
                AzimuthalHarmonicIntegrand fnc(*dens, m, gridR[iR], gridz[iz], useDerivs);
                math::integrateNdim(fnc, Rzmin, Rzmax, 
                    EPSREL_POTENTIAL_INT, MAX_NUM_EVAL,
                    result, error, &numEval);
                if(isZReflSymmetric(*dens) && gridz[iz]==0)
                    result[2] = 0;
                if(gridR[iR]==0)
                    result[1] = 0;
                for(unsigned int q=0; q<numQuantitiesOutput; q++)
                    output[q]->at(m+mmax)(iR,iz) += result[q];
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computePotentialCoefsCyl: "+errorMsg);
}

// transform an N-body snapshot to an array of Fourier harmonic coefficients
template<typename ParticleT>
void computeAzimuthalHarmonics(
    const particles::PointMassArray<ParticleT>& points,
    const std::vector<int>& indices,
    std::vector<std::vector<double> >& harmonics,
    std::vector<std::pair<double, double> > &Rz)
{
    assert(harmonics.size()>0 && indices.size()>0);
    unsigned int nbody = points.size();
    unsigned int nind  = indices.size();
    int mmax = (harmonics.size()-1)/2;
    bool needSine = false;
    for(unsigned int i=0; i<nind; i++) {
        needSine |= indices[i]<0;
        harmonics[indices[i]+mmax].resize(nbody);
    }
    Rz.resize(nbody);
    std::vector<double> tmpharm(2*mmax);
    for(unsigned int b=0; b<nbody; b++) {
        const coord::PosCyl pc = coord::toPosCyl(points.point(b));
        Rz[b].first = pc.R;
        Rz[b].second= pc.z;
        math::trigMultiAngle(pc.phi, mmax, needSine, &tmpharm.front());
        for(unsigned int i=0; i<nind; i++) {
            int m = indices[i];
            harmonics[m+mmax][b] = points.mass(b) *
                (m==0 ? 1 : m>0 ? 2*tmpharm[m-1] : 2*tmpharm[mmax-m-1]);
        }
    }
}

void computePotentialCoefsFromPoints(
    const std::vector<int>& indices,
    const std::vector<std::vector<double> > &harmonics,
    const std::vector<std::pair<double, double> > &Rz,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    bool useDerivs,
    std::vector< math::Matrix<double> >* output[])
{
    assert(harmonics.size()>0 && indices.size()>0);
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    int mmax = (harmonics.size()-1)/2;
    bool zsym = gridz[0]==0;  // whether we assume z-reflection symmetry, deduced from the grid
    unsigned int numQuantitiesOutput = useDerivs ? 3 : 1;  // Phi only, or Phi plus two derivs
    for(unsigned int q=0; q<numQuantitiesOutput; q++) {
        output[q]->resize(2*mmax+1);
        for(unsigned int i=0; i<indices.size(); i++)
            output[q]->at(indices[i]+mmax).resize(sizeR, sizez);
    }
    int nbody = Rz.size();
    int numPoints = sizeR * sizez;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
        for(int b=0; b<nbody; b++) {
            for(unsigned int i=0; i<indices.size(); i++) {
                int m = indices[i];
                double values[3] = {0,0,0};
                computePotentialHarmonicAtPoint(m, Rz[b].first, Rz[b].second,
                    gridR[iR], gridz[iz], harmonics[m+mmax][b], useDerivs, values);
                if(zsym) {  // add symmetric contribution from -z
                    computePotentialHarmonicAtPoint(m, Rz[b].first, -Rz[b].second,
                        gridR[iR], gridz[iz], harmonics[m+mmax][b], useDerivs, values);
                    for(unsigned int q=0; q<numQuantitiesOutput; q++)
                        values[q] *= 0.5;  // average with the one from +z
                }
                for(unsigned int q=0; q<numQuantitiesOutput; q++)
                    output[q]->at(m+mmax)(iR,iz) += values[q];
            }
        }
    }
}

}  // internal namespace

// the driver functions that use the templated routines defined above

// density coefs from density
void computeDensityCoefsCyl(const BaseDensity& src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &output)
{
    std::vector< math::Matrix<double> > *coefs = &output;
    computeFourierCoefs<BaseDensity, 1>(src, mmax, gridR, gridz, &coefs);
    // the value at R=0,z=0 might be undefined, in which case we take it from nearby points
    for(unsigned int iz=0; iz<gridz.size(); iz++)
        if(gridz[iz] == 0 && !math::isFinite(output[mmax](0, iz))) {
            double d1 = output[mmax](0, iz+1);  // value at R=0,z>0
            double d2 = output[mmax](1, iz);    // value at R>0,z=0
            for(unsigned int mm=0; mm<output.size(); mm++)
                if(output[mm].numCols()>0)  // loop over all non-empty harmonics
                    output[mm](0, iz) = mm==mmax ? (d1+d2)/2 : 0;  // only m=0 survives
        }
}

// potential coefs from potential
void computePotentialCoefsCyl(const BasePotential &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz)
{
    std::vector< math::Matrix<double> > *coefs[3] = {&Phi, &dPhidR, &dPhidz};
    computeFourierCoefs<BasePotential, 3>(src, mmax, gridR, gridz, coefs);
    // assign potential derivatives at R=0 or z=0 to zero, depending on the symmetry
    for(unsigned int iz=0; iz<gridz.size(); iz++) {
        if(gridz[iz] == 0 && isZReflSymmetric(src)) {
            for(unsigned int iR=0; iR<gridR.size(); iR++)
                for(unsigned int mm=0; mm<dPhidz.size(); mm++)
                    if(dPhidz[mm].numCols()>0)  // loop over all non-empty harmonics
                        dPhidz[mm](iR, iz) = 0; // z-derivative is zero in the symmetry plane
        }
        for(unsigned int mm=0; mm<dPhidR.size(); mm++)
            if(dPhidR[mm].numCols()>0)  // loop over all non-empty harmonics
                dPhidR[mm](0, iz) = 0;  // R-derivative at R=0 should always be zero
    }
}

// potential coefs from density, with derivatves
void computePotentialCoefsCyl(const BaseDensity &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz)
{
    std::vector< math::Matrix<double> > *coefs[3] = {&Phi, &dPhidR, &dPhidz};
    computePotentialCoefsFromDensity(src, mmax, gridR, gridz, true, coefs);
}

// potential coefs from density, without derivatves
void computePotentialCoefsCyl(const BaseDensity &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi)
{
    std::vector< math::Matrix<double> > *coefs = &Phi;
    computePotentialCoefsFromDensity(src, mmax, gridR, gridz, false, &coefs);
}

// potential coefs from N-body array, with derivatives
template<typename ParticleT>
void computePotentialCoefsCyl(
    const particles::PointMassArray<ParticleT>& points,
    coord::SymmetryType sym,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz)
{
    if(gridR.size() < CYLSPLINE_MIN_GRID_SIZE || gridz.size() < CYLSPLINE_MIN_GRID_SIZE ||
        mmax > CYLSPLINE_MAX_ANGULAR_HARMONIC ||
        (isZReflSymmetric(sym) && gridz[0] != 0) )
        throw std::invalid_argument("computePotentialCoefsCyl: invalid grid parameters");
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, sym);
    std::vector<std::vector<double> > harmonics(2*mmax+1);
    std::vector<std::pair<double, double> > Rz;
    computeAzimuthalHarmonics(points, indices, harmonics, Rz);
    std::vector< math::Matrix<double> >* output[] = {&Phi, &dPhidR, &dPhidz};
    computePotentialCoefsFromPoints(indices, harmonics, Rz, gridR, gridz, true, output);
}

// template instantiations
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosCar>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&, std::vector< math::Matrix<double> >&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosCyl>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&, std::vector< math::Matrix<double> >&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosSph>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&, std::vector< math::Matrix<double> >&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosVelCar>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&, std::vector< math::Matrix<double> >&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosVelCyl>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&, std::vector< math::Matrix<double> >&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosVelSph>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&, std::vector< math::Matrix<double> >&,
    std::vector< math::Matrix<double> >&);

// potential coefs from N-body array, without derivatives
template<typename ParticleT>
void computePotentialCoefsCyl(
    const particles::PointMassArray<ParticleT>& points,
    coord::SymmetryType sym,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi)
{
    if(gridR.size() < CYLSPLINE_MIN_GRID_SIZE || gridz.size() < CYLSPLINE_MIN_GRID_SIZE ||
        mmax > CYLSPLINE_MAX_ANGULAR_HARMONIC ||
        (isZReflSymmetric(sym) && gridz[0] != 0) )
        throw std::invalid_argument("computePotentialCoefsCyl: invalid grid parameters");
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, sym);
    std::vector<std::vector<double> > harmonics(2*mmax+1);
    std::vector<std::pair<double, double> > Rz;
    computeAzimuthalHarmonics(points, indices, harmonics, Rz);
    std::vector< math::Matrix<double> >* output = &Phi;
    computePotentialCoefsFromPoints(indices, harmonics, Rz, gridR, gridz, false, &output);
}

// template instantiations
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosCar>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosCyl>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosSph>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosVelCar>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosVelCyl>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&);
template void computePotentialCoefsCyl(const particles::PointMassArray<coord::PosVelSph>&,
    coord::SymmetryType, const unsigned int, const std::vector<double>&, const std::vector<double>&,
    std::vector< math::Matrix<double> >&);

// -------- public classes: DensityAzimuthalHarmonic --------- //

PtrDensity DensityAzimuthalHarmonic::create(const BaseDensity& src, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax, 
    unsigned int gridSizez, double zmin, double zmax)
{
    if(gridSizeR<=2 || Rmin<=0 || Rmax<=Rmin || gridSizez<=2 || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("DensityAzimuthalHarmonic: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(src))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > coefs;
    // to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle:
    // the number of output harmonics remains the same, but the accuracy of approximation increases.
    int mmaxFourier = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, MMIN_AZIMUTHAL_FOURIER);
    computeDensityCoefsCyl(src, mmaxFourier, gridR, gridz, coefs);
    if(mmaxFourier > (int)mmax) {
        // remove extra coefs: (mmaxFourier-mmax) from both heads and tails of arrays
        coefs.erase(coefs.begin() + mmaxFourier+mmax+1, coefs.end());
        coefs.erase(coefs.begin(), coefs.begin() + mmaxFourier-mmax);
    }
    return PtrDensity(new DensityAzimuthalHarmonic(gridR, gridz, coefs));
}

DensityAzimuthalHarmonic::DensityAzimuthalHarmonic(
    const std::vector<double>& gridR_orig, const std::vector<double>& gridz_orig,
    const std::vector< math::Matrix<double> > &coefs)
{
    unsigned int sizeR = gridR_orig.size(), sizez_orig = gridz_orig.size(), sizez = sizez_orig;
    if(sizeR<2 || sizez<2 || 
        coefs.size()%2 == 0 || coefs.size() > 2*MMAX_AZIMUTHAL_FOURIER+1)
        throw std::invalid_argument("DensityAzimuthalHarmonic: incorrect grid size");
    int mysym = coord::ST_AXISYMMETRIC;
    // grid in z may only cover half-space z>=0 if the density is z-reflection symmetric
    std::vector<double> gridR=gridR_orig, gridz;
    if(gridz_orig[0] == 0) {
        gridz = math::mirrorGrid(gridz_orig);
        sizez = 2*sizez_orig-1;
    } else {  // if the original grid covered both z>0 and z<0, we assume that the symmetry is broken
        gridz = gridz_orig;
        mysym &= ~coord::ST_ZREFLECTION;
    }
    for(unsigned int iR=0; iR<sizeR; iR++)
        gridR[iR] = log(1+gridR[iR]);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = log(1+fabs(gridz[iz]))*math::sign(gridz[iz]);

    math::Matrix<double> val(sizeR, sizez);
    int mmax = (coefs.size()-1)/2;
    spl.resize(2*mmax+1);
    for(int mm=0; mm<=2*mmax; mm++) {
        if(coefs[mm].numRows() == 0 && coefs[mm].numCols() == 0)
            continue;
        if(coefs[mm].numRows() != sizeR || coefs[mm].numCols() != sizez_orig)
            throw std::invalid_argument("DensityAzimuthalHarmonic: incorrect coefs array size");
        double sum=0;
        for(unsigned int iR=0; iR<sizeR; iR++)
            for(unsigned int iz=0; iz<sizez_orig; iz++) {
                double value = coefs[mm](iR, iz);
                if((mysym & coord::ST_ZREFLECTION) == coord::ST_ZREFLECTION) {
                    val(iR, sizez_orig-1-iz) = value;
                    val(iR, sizez_orig-1+iz) = value;
                } else
                    val(iR, iz) = value;
                sum += fabs(value);
            }
        if(sum>0) {
            spl[mm] = math::CubicSpline2d(gridR, gridz, val);
            int m = mm-mmax;
            if(m!=0)  // no z-rotation symmetry because m!=0 coefs are non-zero
                mysym &= ~coord::ST_ZROTATION;
            if(m<0)
                mysym &= ~(coord::ST_YREFLECTION | coord::ST_REFLECTION);
            if((m<0) ^ (m%2 != 0))
                mysym &= ~(coord::ST_XREFLECTION | coord::ST_REFLECTION);
        }
    }
    sym = static_cast<coord::SymmetryType>(mysym);
}

double DensityAzimuthalHarmonic::rho_m(int m, double R, double z) const
{
    int mmax = (spl.size()-1)/2;
    double lR = log(1+R), lz = log(1+fabs(z))*math::sign(z);
    if(math::abs(m)>mmax || spl[m+mmax].isEmpty() ||
        //R<spl[m+mmax].xmin() || z<spl[m+mmax].ymin() || R>spl[m+mmax].xmax() || z>spl[m+mmax].ymax())
       lR<spl[mmax].xmin() || lz<spl[mmax].ymin() ||
       lR>spl[mmax].xmax() || lz>spl[mmax].ymax() )
        return 0;
    return spl[m+mmax].value(lR, lz);
}

double DensityAzimuthalHarmonic::densityCyl(const coord::PosCyl &pos) const
{
    int mmax = (spl.size()-1)/2;
    double lR = log(1+pos.R), lz = log(1+fabs(pos.z))*math::sign(pos.z);
    if( lR<spl[mmax].xmin() || lz<spl[mmax].ymin() ||
        lR>spl[mmax].xmax() || lz>spl[mmax].ymax() )
        return 0;
    double result = 0;
    double trig[2*MMAX_AZIMUTHAL_FOURIER];
    if(!isZRotSymmetric(sym)) {
        bool needSine = !isYReflSymmetric(sym);
        math::trigMultiAngle(pos.phi, mmax, needSine, trig);
    }
    for(int m=-mmax; m<=mmax; m++)
        if(!spl[m+mmax].isEmpty()) {
            double trig_m = m==0 ? 1 : m>0 ? trig[m-1] : trig[mmax-1-m];
            result += spl[m+mmax].value(lR, lz) * trig_m;
        }
    return result;
}

void DensityAzimuthalHarmonic::getCoefs(
    std::vector<double> &gridR, std::vector<double> &gridz, 
    std::vector< math::Matrix<double> > &coefs) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(mmax>=0 && spl.size() == mmax*2+1 && !spl[mmax].isEmpty());
    coefs.resize(2*mmax+1);
    gridR = spl[mmax].xvalues();
    unsigned int sizeR = gridR.size();
    unsigned int sizez = spl[mmax].yvalues().size();
    if(isZReflSymmetric(sym)) {
        // output only coefs for half-space z>=0
        sizez = (sizez+1) / 2;
        gridz.assign(spl[mmax].yvalues().begin() + sizez-1, spl[mmax].yvalues().end());
    } else
        gridz = spl[mmax].yvalues();
    for(unsigned int mm=0; mm<=2*mmax; mm++)
        if(!spl[mm].isEmpty()) {
            coefs[mm].resize(sizeR, sizez);
            for(unsigned int iR=0; iR<sizeR; iR++)
                for(unsigned int iz=0; iz<sizez; iz++)
                    coefs[mm](iR, iz) = spl[mm].value(gridR[iR], gridz[iz]);
            //math::eliminateNearZeros(coefs[mm]);
        }
}

// -------- public classes: CylSpline --------- //

namespace {  // internal routines

// This routine constructs an spherical-harmonic expansion describing
// asymptotic behaviour of the potential beyond the grid definition region.
// It takes the values of potential at the outer edge of the grid in (R,z) plane,
// and finds a combination of SH coefficients that approximate the theta-dependence
// of each m-th azimuthal harmonic term, in the least-square sense.
// In doing so, we must assume that the coefficients behave like C_{lm} ~ r^{-1-l},
// which is valid for empty space, but is not able to describe the residual density;
// thus this asymptotic form describes the potential and forces rather well,
// but returns zero density. Unfortunately we can't do any better without knowing
// the slope of density fall-off in the first place, or in other words,
// without determining the second radial derivative of each harmonic term,
// as is possible for the Multipole potential (even there it requires great care
// to accurately determine the 2nd derivative from values and first derivatives of
// each term at three consecutive radial points, something that is hardly possible here).
static PtrPotential determineAsympt(
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    const std::vector< math::Matrix<double> > &Phi)
{
    bool zsym = gridz[0]==0;
    unsigned int sizeR = gridR.size();
    unsigned int sizez = gridz.size();
    std::vector<coord::PosSph> points;     // coordinates of boundary points
    std::vector<unsigned int> indR, indz;  // indices of these points in the Phi array

    // assemble the boundary points and their indices
    for(unsigned int iR=0; iR<sizeR-1; iR++) {
        // first run along R at the max-z and min-z edges
        unsigned int iz=sizez-1;
        points. push_back(coord::toPosSph(coord::PosCyl(gridR[iR], gridz[iz], 0)));
        indR.push_back(iR);
        indz.push_back(iz);
        if(zsym) {  // min-z edge is the negative of max-z edge
            points. push_back(coord::toPosSph(coord::PosCyl(gridR[iR], -gridz[iz], 0)));
            indR.push_back(iR);
            indz.push_back(iz);
        } else {  // min-z edge must be at the beginning of the array
            iz = 0;
            points. push_back(coord::toPosSph(coord::PosCyl(gridR[iR], gridz[iz], 0)));
            indR.push_back(iR);
            indz.push_back(iz);
        }
    }
    for(unsigned int iz=0; iz<sizez; iz++) {
        // next run along z at max-R edge
        unsigned int iR=sizeR-1;
        points. push_back(coord::toPosSph(coord::PosCyl(gridR[iR], gridz[iz], 0)));
        indR.push_back(iR);
        indz.push_back(iz);
        if(zsym && iz>0) {
            points. push_back(coord::toPosSph(coord::PosCyl(gridR[iR], -gridz[iz], 0)));
            indR.push_back(iR);
            indz.push_back(iz);
        }
    }
    unsigned int npoints = points.size();
    int mmax = (Phi.size()-1)/2;  // # of angular(phi) harmonics in the original potential
    int lmax_fit = 8;             // # of meridional harmonics to fit - don't set too large
    int mmax_fit = std::min<int>(lmax_fit, mmax);
    unsigned int ncoefs = pow_2(lmax_fit+1);
    std::vector<double> Plm(lmax_fit+1);     // temp.storage for sph-harm functions
    std::vector<double> W(ncoefs), zeros(ncoefs);
    double r0 = fmin(gridR.back(), gridz.back());

    // find values of spherical harmonic coefficients
    // that best match the potential at the array of boundary points
    for(int m=-mmax_fit; m<=mmax_fit; m++)
        if(Phi[m+mmax].numCols()*Phi[m+mmax].numRows()>0) {
            // for m-th harmonic, we may have lmax-m+1 different l-terms
            int absm = math::abs(m);
            math::Matrix<double> matr(npoints, lmax_fit-absm+1);
            std::vector<double>  rhs(npoints);
            std::vector<double>  sol;
            // The linear system to solve in the least-square sense is M_{p,l} * S_l = R_p,
            // where R_p = Phi at p-th boundary point (0<=p<npoints),
            // M_{l,p}   = value of l-th harmonic coefficient at p-th boundary point,
            // S_l       = the amplitude of l-th coefficient to be determined.
            for(unsigned int p=0; p<npoints; p++) {
                rhs[p] = Phi[m+mmax](indR[p], indz[p]);
                math::sphHarmArray(lmax_fit, absm, points[p].theta, &Plm.front());
                for(int l=absm; l<=lmax_fit; l++)
                    matr(p, l-absm) =
                        Plm[l-absm] * math::powInt(points[p].r/r0, -l-1) *
                        (m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2);
            }
            math::linearMultiFit(matr, rhs, NULL, sol);
            for(int l=absm; l<=lmax_fit; l++) {
                unsigned int c = math::SphHarmIndices::index(l, m);
                W[c] = sol[l-absm];
            }
        }
    // safeguarding against possible problems
    if(!math::isFinite(W[0])) {
        // something went wrong - at least return a correct value for the l=0 term
        math::Averager avg;
        for(unsigned int p=0; p<npoints; p++)
            avg.add(Phi[mmax](indR[p], indz[p]) * points[p].r / r0);
        W.assign(ncoefs, 0);
        W[0] = avg.mean();
    }
    math::eliminateNearZeros(W);
    return PtrPotential(new PowerLawMultipole(r0, false /*not inner*/, zeros, zeros, W));
}

} // internal namespace

PtrPotential CylSplineExp::create(const BaseDensity& src, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax, 
    unsigned int gridSizez, double zmin, double zmax, bool useDerivs)
{
    if(gridSizeR<=2 || Rmin<=0 || Rmax<=Rmin || gridSizez<=2 || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("Error in CylSplineExp: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(src))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    if(useDerivs)
        computePotentialCoefsCyl(src, mmax, gridR, gridz, Phi, dPhidR, dPhidz);
    else
        computePotentialCoefsCyl(src, mmax, gridR, gridz, Phi);
    return PtrPotential(new CylSplineExp(gridR, gridz, Phi, dPhidR, dPhidz));
}

PtrPotential CylSplineExp::create(const BasePotential& src, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax, 
    unsigned int gridSizez, double zmin, double zmax)
{
    if(gridSizeR<=2 || Rmin<=0 || Rmax<=Rmin || gridSizez<=2 || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("Error in CylSplineExp: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(src))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    std::vector< math::Matrix<double> > *coefs[3] = {&Phi, &dPhidR, &dPhidz};
    // to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle;
    int mmaxFourier = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, MMIN_AZIMUTHAL_FOURIER);
    computeFourierCoefs<BasePotential, 3>(src, mmaxFourier, gridR, gridz, coefs);
    if(mmaxFourier > (int)mmax) {
        // remove extra coefs: (mmaxFourier-mmax) from both heads and tails of arrays
        for(int q=0; q<3; q++) {
            coefs[q]->erase(coefs[q]->begin() + mmaxFourier+mmax+1, coefs[q]->end());
            coefs[q]->erase(coefs[q]->begin(), coefs[q]->begin() + mmaxFourier-mmax);
        }
    }
    return PtrPotential(new CylSplineExp(gridR, gridz, Phi, dPhidR, dPhidz));
}

CylSplineExp::CylSplineExp(
    const std::vector<double> &gridR_orig,
    const std::vector<double> &gridz_orig,
    const std::vector< math::Matrix<double> > &Phi,
    const std::vector< math::Matrix<double> > &dPhidR,
    const std::vector< math::Matrix<double> > &dPhidz)
{
    unsigned int sizeR = gridR_orig.size(), sizez = gridz_orig.size(), sizez_orig = sizez;
    bool haveDerivs = dPhidR.size() > 0 && dPhidz.size() > 0;
    if(sizeR<4 || sizez<4 || gridR_orig[0]!=0 ||
        Phi.size()%2 == 0 || Phi.size() > 2*MMAX_AZIMUTHAL_FOURIER+1 ||
        (haveDerivs && (Phi.size() != dPhidR.size() || Phi.size() != dPhidz.size())) )
        throw std::invalid_argument("CylSplineExp: incorrect grid size");
    int mmax  = (Phi.size()-1)/2;
    int mysym = coord::ST_AXISYMMETRIC;
    bool zsym = true;
    double Phi0;  // potential at R=0,z=0
    // grid in z may only cover half-space z>=0 if the density is z-reflection symmetric:
    std::vector<double> gridR = gridR_orig, gridz;
    if(gridz_orig[0] == 0) {
        gridz = math::mirrorGrid(gridz_orig);
        sizez = 2*sizez_orig-1;
        Phi0  = Phi[mmax](0, 0);
    } else {  // if the original grid covered both z>0 and z<0, we assume that the symmetry is broken
        gridz = gridz_orig;
        mysym&= ~coord::ST_ZREFLECTION;
        zsym  = false;
        Phi0  = Phi[mmax](0, (sizez+1)/2);
    }

    asymptOuter = determineAsympt(gridR_orig, gridz_orig, Phi);
    // at large radii, Phi(r) ~= -Mtotal/r
    double Mtot = -(asymptOuter->value(coord::PosSph(1000*gridR.back(), 0, 0)) * 1000*gridR.back());
    if(Phi0 < 0 && Mtot > 0)     // assign Rscale so that it approximately equals -Mtotal/Phi(r=0),
        Rscale  = -Mtot / Phi0;  // i.e. would equal the scale radius of a Plummer potential
    else
        Rscale  = 1.;  // rather arbitrary

    // transform the grid to log-scaled coordinates
    for(unsigned int i=0; i<sizeR; i++) {
        gridR[i] = log(1+gridR[i]/Rscale);
    }
    for(unsigned int i=0; i<sizez; i++) {
        gridz[i] = log(1+fabs(gridz[i])/Rscale) * math::sign(gridz[i]);
    }

    math::Matrix<double> val(sizeR, sizez), derR(sizeR, sizez), derz(sizeR, sizez);
    spl.resize(2*mmax+1);
    for(int mm=0; mm<=2*mmax; mm++) {
        if(Phi[mm].numRows() == 0 && Phi[mm].numCols() == 0)
            continue;
        if((   Phi[mm].numRows() != sizeR ||    Phi[mm].numCols() != sizez_orig) || (haveDerivs && 
           (dPhidR[mm].numRows() != sizeR || dPhidR[mm].numCols() != sizez_orig  ||
            dPhidz[mm].numRows() != sizeR || dPhidz[mm].numCols() != sizez_orig)))
            throw std::invalid_argument("CylSplineExp: incorrect coefs array size");
        double sum=0;
        for(unsigned int iR=0; iR<sizeR; iR++) {
            double R = gridR_orig[iR];
            for(unsigned int iz=0; iz<sizez_orig; iz++) {
                double z = gridz_orig[iz];
                unsigned int iz1 = zsym ? sizez_orig-1+iz : iz;  // index in the internal 2d grid
                // values of potential and its derivatives are represented as scaled 2d functions:
                // the amplitude is scaled by 'amp', while both coordinates are log-scaled.
                // thus the values passed to the constructor of 2d spline must be properly modified,
                // and the derivatives additionally transformed to the scaled coordinates.
                double amp = sqrt(pow_2(Rscale)+pow_2(R)+pow_2(z));
                val (iR, iz1) =  Phi[mm](iR,iz) * amp;
                if(haveDerivs) {
                    derR(iR, iz1) = (dPhidR[mm](iR,iz) * amp + Phi[mm](iR,iz) * R / amp) * (R+Rscale);
                    derz(iR, iz1) = (dPhidz[mm](iR,iz) * amp + Phi[mm](iR,iz) * z / amp) * (fabs(z)+Rscale);
                }
                if(zsym) {
                    assert(z>=0);  // source data only covers upper half-space
                    unsigned int iz2 = sizez_orig-1-iz;  // index of the reflected cell
                    val (iR, iz2) = val (iR, iz1);
                    derR(iR, iz2) = derR(iR, iz1);
                    derz(iR, iz2) =-derz(iR, iz1);
                }
                sum += fabs(Phi[mm](iR, iz));
            }
        }
        if(sum>0) {
            spl[mm] = haveDerivs ? 
                math::PtrInterpolator2d(new math::QuinticSpline2d(gridR, gridz, val, derR, derz)) :
                math::PtrInterpolator2d(new math::CubicSpline2d(gridR, gridz, val, 0, NAN, NAN, NAN));
            // check if this non-trivial harmonic breaks any symmetry
            int m = mm-mmax;
            if(m!=0)  // no z-rotation symmetry because m!=0 coefs are non-zero
                mysym &= ~coord::ST_ZROTATION;
            if(m<0)
                mysym &= ~(coord::ST_YREFLECTION | coord::ST_REFLECTION);
            if((m<0) ^ (m%2 != 0))
                mysym &= ~(coord::ST_XREFLECTION | coord::ST_REFLECTION);
        }
    }
    sym = static_cast<coord::SymmetryType>(mysym);
}

void CylSplineExp::evalCyl(const coord::PosCyl &pos,
    double* val, coord::GradCyl* der, coord::HessCyl* der2) const
{
    int mmax = (spl.size()-1)/2;
    double Rscaled = log(1+pos.R/Rscale);
    double zscaled = log(1+fabs(pos.z)/Rscale) * math::sign(pos.z);
    if( Rscaled<spl[mmax]->xmin() || zscaled<spl[mmax]->ymin() ||
        Rscaled>spl[mmax]->xmax() || zscaled>spl[mmax]->ymax() ) {
        // outside the grid definition region, use the asymptotic expansion
        asymptOuter->eval(pos, val, der, der2);
        return;
    }

    // only compute those quantities that will be needed in output
    bool needPhi  = true;
    bool needGrad = der !=NULL || der2!=NULL;
    bool needHess = der2!=NULL;
    double trig_arr[2*MMAX_AZIMUTHAL_FOURIER];
    if(!isZRotSymmetric(sym)) {
        bool needSine = needGrad || !isYReflSymmetric(sym);
        math::trigMultiAngle(pos.phi, mmax, needSine, trig_arr);
    }
    
    // total scaled potential, gradient and hessian in scaled coordinates
    double Phi = 0;
    coord::GradCyl grad;
    coord::HessCyl hess;
    grad.dR  = grad.dz = grad.dphi = 0;
    hess.dR2 = hess.dz2 = hess.dphi2 = hess.dRdz = hess.dRdphi = hess.dzdphi = 0;

    // loop over azimuthal harmonics and compute the temporary (scaled) values
    for(int mm=0; mm<=2*mmax; mm++) {
        if(!spl[mm])  // empty harmonic
            continue;
        // scaled value, gradient and hessian of m-th harmonic in scaled coordinates
        double Phi_m;
        coord::GradCyl dPhi_m;
        coord::HessCyl d2Phi_m;
        spl[mm]->evalDeriv(Rscaled, zscaled,
            needPhi  ?   &Phi_m      : NULL, 
            needGrad ?  &dPhi_m.dR   : NULL,
            needGrad ?  &dPhi_m.dz   : NULL,
            needHess ? &d2Phi_m.dR2  : NULL,
            needHess ? &d2Phi_m.dRdz : NULL,
            needHess ? &d2Phi_m.dz2  : NULL);
        int m = mm-mmax;
        double trig  = m==0 ? 1. : m>0 ? trig_arr[m-1] : trig_arr[mmax-1-m];  // cos or sin
        double dtrig = m==0 ? 0. : m>0 ? -m*trig_arr[mmax+m-1] : -m*trig_arr[-m-1];
        double d2trig = -m*m*trig;
        Phi += Phi_m * trig;
        if(needGrad) {
            grad.dR   += dPhi_m.dR *  trig;
            grad.dz   += dPhi_m.dz *  trig;
            grad.dphi +=  Phi_m    * dtrig;
        }
        if(needHess) {
            hess.dR2    += d2Phi_m.dR2  *   trig;
            hess.dz2    += d2Phi_m.dz2  *   trig;
            hess.dRdz   += d2Phi_m.dRdz *   trig;
            hess.dRdphi +=  dPhi_m.dR   *  dtrig;
            hess.dzdphi +=  dPhi_m.dz   *  dtrig;
            hess.dphi2  +=   Phi_m      * d2trig;
        }
    }

    // unscale both amplitude of all quantities and their coordinate derivatives
    double r2 = pow_2(pos.R) + pow_2(pos.z);
    double S  = 1 / sqrt(pow_2(Rscale)+r2);  // scaling of the amplitude
    if(val)
        *val = S * Phi;
    if(!needGrad)
        return;
    double dSdr_over_r = -S*S*S;    
    double dRscaleddR = 1/(Rscale+pos.R);
    double dzscaleddz = 1/(Rscale+fabs(pos.z));
    if(der) {
        der->dR   = S * grad.dR * dRscaleddR + dSdr_over_r * Phi * pos.R;
        der->dz   = S * grad.dz * dzscaleddz + dSdr_over_r * Phi * pos.z;
        der->dphi = S * grad.dphi;
    }
    if(der2) {
        double d2RscaleddR2 = -pow_2(dRscaleddR);
        double d2zscaleddz2 = -pow_2(dzscaleddz) * math::sign(pos.z);
        double d2Sdr2 = (pow_2(Rscale) - 2 * r2) * dSdr_over_r * S * S;
        r2 += 1e-100;  // prevent 0/0 indeterminacy if r2==0
        der2->dR2 =
            (d2Sdr2 * pow_2(pos.R) + dSdr_over_r * pow_2(pos.z)) / r2 * Phi + 
            dSdr_over_r * 2 * pos.R * dRscaleddR * grad.dR +
            S * (hess.dR2 * pow_2(dRscaleddR) + grad.dR * d2RscaleddR2);
        der2->dz2 =
            (d2Sdr2 * pow_2(pos.z) + dSdr_over_r * pow_2(pos.R)) / r2 * Phi + 
            dSdr_over_r * 2 * pos.z * dzscaleddz * grad.dz +
            S * (hess.dz2 * pow_2(dzscaleddz) + grad.dz * d2zscaleddz2);
        der2->dRdz =
            (d2Sdr2 - dSdr_over_r) * pos.R * pos.z / r2 * Phi +
            dSdr_over_r * (pos.z * dRscaleddR * grad.dR + pos.R * dzscaleddz * grad.dz) +
            S * hess.dRdz * dRscaleddR * dzscaleddz;
        der2->dRdphi =
            dSdr_over_r * pos.R * grad.dphi +
            S * dRscaleddR * hess.dRdphi;
        der2->dzdphi =
            dSdr_over_r * pos.z * grad.dphi +
            S * dzscaleddz * hess.dzdphi;
        der2->dphi2 = S * hess.dphi2;
    }
}

void CylSplineExp::getCoefs(
    std::vector<double> &gridR, std::vector<double> &gridz, 
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(mmax>=0 && spl.size() == mmax*2+1 && !spl[mmax]->isEmpty());
    const std::vector<double>& scaledR = spl[mmax]->xvalues();
    const std::vector<double>& scaledz = spl[mmax]->yvalues();
    unsigned int sizeR = scaledR.size();
    unsigned int sizez = scaledz.size();
    unsigned int iz0   = 0;
    if(isZReflSymmetric(sym)) {
        // output only coefs for half-space z>=0
        sizez = (sizez+1) / 2;
        iz0   = sizez-1;  // index of z=0 value in the internal scaled coordinate axis array
    }
    // unscale the coordinates
    gridR.resize(sizeR);
    for(unsigned int iR=0; iR<sizeR; iR++)
        gridR[iR] = Rscale * (exp(scaledR[iR]) - 1);
    gridz.resize(sizez);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = Rscale * (exp(fabs(scaledz[iz+iz0])) - 1) * math::sign(scaledz[iz+iz0]);

    Phi.resize(2*mmax+1);
    dPhidR.resize(2*mmax+1);
    dPhidz.resize(2*mmax+1);
    for(unsigned int mm=0; mm<=2*mmax; mm++) {
        if(!spl[mm]) 
            continue;    
        Phi   [mm].resize(sizeR, sizez);
        dPhidR[mm].resize(sizeR, sizez);
        dPhidz[mm].resize(sizeR, sizez);
        for(unsigned int iR=0; iR<sizeR; iR++)
            for(unsigned int iz=0; iz<sizez; iz++) {
                double Rscaled = scaledR[iR];     // coordinates in the internal scaled coords array
                double zscaled = scaledz[iz+iz0];
                // scaling of the amplitude
                double S = 1 / sqrt(pow_2(Rscale) + pow_2(gridR[iR]) + pow_2(gridz[iz]));
                double dSdr_over_r = -S*S*S;
                // scaling of derivatives
                double dRscaleddR  = 1 / (Rscale + gridR[iR]);
                double dzscaleddz  = 1 / (Rscale + fabs(gridz[iz]));
                double val, dR, dz;
                spl[mm]->evalDeriv(Rscaled, zscaled, &val, &dR, &dz);
                Phi   [mm](iR,iz) = S * val;
                dPhidR[mm](iR,iz) = S * dR * dRscaleddR + dSdr_over_r * val * gridR[iR];
                dPhidz[mm](iR,iz) = S * dz * dzscaleddz + dSdr_over_r * val * gridz[iz];
            }
    }
}

// ------ old api, to be removed ------ //

//-------- Auxiliary direct-integration potential --------//

/** Direct computation of potential for any density profile, using double integration over space.
    Not suitable for orbit integration, as it does not provide expressions for forces;
    only used for computing potential harmonics (coefficients of Fourier expansion in 
    azimuthal angle phi) at any point in (R,z) plane, for initializing the Cylindrical Spline 
    potential approximation. 
    It can be used in either of the two modes: 
    1) with a smooth input density model, which may or may not be axisymmetric itself;
       in the latter case an intermediate representation of its angular(azimuthal) harmonics
       is created and interpolated on a 2d grid covering almost all of (R,z) plane,
       to speed up computation of potential integral.
    2) with a discrete point mass array, in which case the integral is evaluated by summing 
       the contribution from each particle
*/
class DirectPotential: public BasePotentialCyl
{
public:
    /// init potential from analytic mass model
    DirectPotential(const BaseDensity& _density, unsigned int mmax);

    /// init potential from N-body snapshot
    DirectPotential(const particles::PointMassArray<coord::PosCyl>& _points, 
        unsigned int mmax, coord::SymmetryType sym);

    virtual ~DirectPotential() {};
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "Direct"; };
    virtual coord::SymmetryType symmetry() const { return mysymmetry; }

    /// compute m-th azimuthal harmonic of potential
    double Rho_m(double R, double z, int m) const;

    /// return m-th azimuthal harmonic of density, either by interpolating 
    /// the pre-computed 2d spline, or calculating it on-the-fly using computeRho_m()
    double Phi_m(double R, double z, int m) const;

    /// redefine the following two routines to count particles in the point-mass-set regime
    virtual double enclosedMass(const double radius) const;
    virtual double totalMass() const;

private:
    /// pointer to the input density model (if provided) - not owned by this object;
    const BaseDensity* density;

    /// input discrete point mass set (if provided)
    const particles::PointMassArray<coord::PosCyl>* points;

    /// symmetry type of the input density model (axisymmetric or not)
    coord::SymmetryType mysymmetry;

    /// interpolating splines for Fourier harmonics Rho_m(R,z), 
    /// in case that the input density is not axisymmetric
    std::vector<math::CubicSpline2d> splines;

    virtual void evalCyl(const coord::PosCyl& pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

    virtual double densityCyl(const coord::PosCyl& pos) const;
};

DirectPotential::DirectPotential(const BaseDensity& _density, unsigned int _mmax) :
    density(&_density), points(NULL), mysymmetry(_density.symmetry())
{
    if((mysymmetry & coord::ST_AXISYMMETRIC)==coord::ST_AXISYMMETRIC)
        return;  // no further action necessary
    // otherwise prepare interpolating splines in (R,z) for Fourier expansion of density in angle phi
    int mmax=_mmax;
    splines.resize(mmax*2+1);
    // set up reasonable min/max values: if they are inappropriate, it only will slowdown the computation 
    // but not deteriorate its accuracy, because the interpolation is not used outside the grid
    double totalmass = density->totalMass();
    if(!math::isFinite(totalmass))
        throw std::invalid_argument("DirectPotential: source density model has infinite mass");
    double Rmin = getRadiusByMass(*density, totalmass*0.01)*0.02;
    double Rmax = getRadiusByMass(*density, totalmass*0.99)*50.0;
    double delta=0.05;  // relative difference between grid nodes = log(x[n+1]/x[n]) 
    unsigned int numNodes = static_cast<unsigned int>(log(Rmax/Rmin)/delta);
    std::vector<double> grid = math::createNonuniformGrid(numNodes, Rmin, Rmax, true);
    std::vector<double> gridz(2*grid.size()-1);
    for(unsigned int i=0; i<grid.size(); i++) {
        gridz[grid.size()-1-i] =-grid[i];
        gridz[grid.size()-1+i] = grid[i];
    }
    math::Matrix<double> values(grid.size(), gridz.size());
    // whether densities at z and -z are different
    bool zsymmetry = (density->symmetry()&coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL;
    // if triaxial symmetry, do not use sine terms which correspond to m<0
    int mmin = (density->symmetry() & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL ? 0 :-1;
    // if triaxial symmetry, use only even m
    int mstep= (density->symmetry() & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL ? 2 : 1;
    for(int m=mmax*mmin; m<=mmax; m+=mstep) {
        for(unsigned int iR=0; iR<grid.size(); iR++)
            for(unsigned int iz=0; iz<grid.size(); iz++) {
                double val = computeRho_m(*density, grid[iR], grid[iz], m);
                if(!math::isFinite(val)) {
                    if(iR==0 && iz==0)  // may have a singularity at origin,
                                        // substitute the infinite density with something reasonable
                        val = std::max<double>(computeRho_m(*density, grid[1], grid[0], m), 
                                               computeRho_m(*density, grid[0], grid[1], m));
                    else val=0;
                }
                values(iR, grid.size()-1+iz) = val;
                if(!zsymmetry && iz>0) {
                    val = computeRho_m(*density, grid[iR], -grid[iz], m);
                    if(!math::isFinite(val)) val=0;  // don't let rubbish in
                }
                values(iR, grid.size()-1-iz) = val;
            }
        splines[mmax+m] = math::CubicSpline2d(grid, gridz, values);
    }
}

DirectPotential::DirectPotential(const particles::PointMassArray<coord::PosCyl>& _points, 
    unsigned int , coord::SymmetryType sym) :
    density(NULL), points(&_points), mysymmetry(sym) 
{
    if(points->size()==0)
        throw std::invalid_argument("DirectPotential: empty input array of particles");
};

double DirectPotential::totalMass() const
{
    assert((density!=NULL) ^ (points!=NULL));  // either of the two regimes
    if(density!=NULL) 
        return density->totalMass();
    else
        return points->totalMass();
}

double DirectPotential::enclosedMass(const double r) const
{
    assert((density!=NULL) ^ (points!=NULL));  // either of the two regimes
    if(density!=NULL)
        return density->enclosedMass(r);
    else {
        double mass=0;
        for(particles::PointMassArray<coord::PosCyl>::ArrayType::const_iterator pt=points->data.begin(); 
            pt!=points->data.end(); pt++) 
        {
            if(pow_2(pt->first.R)+pow_2(pt->first.z) <= pow_2(r))
                mass+=pt->second;
        }
        return mass;
    }
}

double DirectPotential::densityCyl(const coord::PosCyl& pos) const
{
    assert(density!=NULL);  // not applicable in discrete point set mode
    if(splines.size()==0)   // no interpolation
        return density->density(pos); 
    else {
        double val=0;
        int mmax=splines.size()/2;
        for(int m=-mmax; m<=mmax; m++)
            val += Rho_m(pos.R, pos.z, m) * (m>=0 ? cos(m*pos.phi) : sin(-m*pos.phi));
        return std::max<double>(val, 0);
    }
}

double DirectPotential::Rho_m(double R, double z, int m) const
{
    if(splines.size()==0) {  // source density is axisymmetric
        assert(m==0 && density!=NULL);
        return density->density(coord::PosCyl(R, z, 0));
    }
    size_t mmax=splines.size()/2;
    if(splines[mmax+m].isEmpty())
        return 0;
    if( R<splines[mmax+m].xmin() || R>splines[mmax+m].xmax() || 
        z<splines[mmax+m].ymin() || z>splines[mmax+m].ymax() )
        // outside interpolating grid -- compute directly by integration
        return computeRho_m(*density, R, z, m);
    else
        return splines[mmax+m].value(R, z);
}

/** Compute the following integral for a fixed integer value of m>=0 and arbitrary a>=0, b>=0, c:
    \f$  \int_0^\infty J_m(a x) J_m(b x) \exp(-|c| x) dx  \f$,  where J_m are Bessel functions. */
double besselInt(double a, double b, double c, int m)
{
    assert(a>=0 && b>=0);
    if(fabs(a)<1e-10 || fabs(b)<1e-10)
        return m==0 ? 1/sqrt(a*a + b*b + c*c) : 0;
    else {
        double x = (a*a+b*b+c*c)/(2*a*b);
        return math::legendreQ(m-0.5, x) / (M_PI * sqrt(a*b));
    }
}
    
/// integration for potential computation
class DirectPotentialIntegrand: public math::IFunctionNdim {
public:
    DirectPotentialIntegrand(const DirectPotential& _potential, 
        double _R, double _z, int _m) :
        potential(_potential), R(_R), z(_z), m(_m) {};
    // evaluate the function at a given (R1,z1) point (scaled)
    virtual void eval(const double Rz[], double values[]) const
    {
        if(Rz[0]>=1. || Rz[1]>=1.) {  // scaled coords point at infinity
            values[0] = 0;
            return;
        }
        const double R1 = Rz[0]/(1-Rz[0]);  // un-scale input coordinates
        const double z1 = Rz[1]/(1-Rz[1]);
        const double jac = pow_2((1-Rz[0])*(1-Rz[1]));  // jacobian of scaled coord transformation
        double result = 0;
        if(R1!=R || z1!=z)
            result = -2*M_PI*R1 * (m==0 ? 1 : 2) * 
            potential.Rho_m(R1, z1, m) * ( besselInt(R, R1, z-z1, m) +
            /*potential.Rho_m(R1,-z1, m)*/ besselInt(R, R1, z+z1, m) ) / jac;
        values[0] = result;
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 1; }
private:
    const DirectPotential& potential;
    double R, z;
    int m;
};

double DirectPotential::Phi_m(double R, double Z, int m) const
{
    if(density==NULL) {  // invoked in the discrete point set mode
        assert(points->size()>0);
        double val=0;
        for(particles::PointMassArray<coord::PosCyl>::ArrayType::const_iterator pt=points->data.begin(); 
            pt!=points->data.end(); pt++) 
        {
            const coord::PosCyl& pc = pt->first;
            double val1 = besselInt(R, pc.R, Z-pc.z, math::abs(m));
            if((mysymmetry & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL)   // add symmetric contribution from -Z
                val1 = (val1 + besselInt(R, pc.R, Z+pc.z, math::abs(m)))/2.;
            if(math::isFinite(val1))
                val += pt->second * val1 * (m==0 ? 1 : m>0 ? 2*cos(m*pc.phi) : 2*sin(-m*pc.phi) );
        }
        return -val;
    }
    // otherwise invoked in the smooth density profile mode
    int mmax = splines.size()/2;
    if(splines.size()>0 && splines[mmax+m].isEmpty())
        return 0;  // using splines for m-components of density but it is identically zero at this m
    DirectPotentialIntegrand fnc(*this, R, Z, m);
    double Rzmin[2]={0.,0.}, Rzmax[2]={1.,1.}; // integration box in scaled coords
    double result, error;
    int numEval;
    math::integrateNdim(fnc, Rzmin, Rzmax, 
        EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, &result, &error, &numEval);
    return result;
};

void DirectPotential::evalCyl(const coord::PosCyl& pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    if(deriv!=NULL || deriv2!=NULL)
        throw std::invalid_argument("DirectPotential: derivatives not implemented");
    assert(potential!=NULL);
    *potential = 0;
    int mmax=splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        *potential += Phi_m(pos.R, pos.z, m) * (m>=0 ? cos(m*pos.phi) : sin(-m*pos.phi));
}

//----------------------------------------------------------------------------//
// Cylindrical spline potential 

CylSplineExpOld::CylSplineExpOld(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const BaseDensity& srcdensity, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    DirectPotential pot_tmp(srcdensity, Ncoefs_phi);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
}

CylSplineExpOld::CylSplineExpOld(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const BasePotential& potential, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, potential, radius_min, radius_max, z_min, z_max);
}

CylSplineExpOld::CylSplineExpOld(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const particles::PointMassArray<coord::PosCyl>& points, coord::SymmetryType _sym, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    mysymmetry=_sym;
    if(Ncoefs_phi==0)
        mysymmetry=(coord::SymmetryType)(mysymmetry | coord::ST_ZROTATION);
    DirectPotential pot_tmp(points, Ncoefs_phi, mysymmetry);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
}

CylSplineExpOld::CylSplineExpOld(const std::vector<double> &gridR, const std::vector<double>& gridz, 
    const std::vector< std::vector<double> > &coefs)
{
    if( gridR.size()<CYLSPLINE_MIN_GRID_SIZE || gridz.size()<CYLSPLINE_MIN_GRID_SIZE || 
        gridR.size()>CYLSPLINE_MAX_GRID_SIZE || gridz.size()>CYLSPLINE_MAX_GRID_SIZE ||
        coefs.size()==0 || coefs.size()%2!=1 || 
        coefs[coefs.size()/2].size()!=gridR.size()*gridz.size()) {
        throw std::invalid_argument("CylSplineExp: Invalid parameters in the constructor");
    } else {
        grid_R=gridR;
        grid_z=gridz;
        if(gridz[0] == 0) {  // z-reflection symmetry
            grid_z = math::mirrorGrid(gridz);
            unsigned int nR = gridR.size(), nz = gridz.size();
            std::vector<std::vector<double> > coefs1(coefs.size());
            for(unsigned int i=0; i<coefs.size(); i++)
                if(coefs[i].size() == nR*nz) {
                    coefs1[i].resize(nR * (nz*2-1));
                    for(unsigned int iR=0; iR<nR; iR++)
                        for(unsigned int iz=0; iz<nz; iz++) {
                        coefs1[i][(nz-1+iz)*nR+iR] = coefs[i][iz*nR+iR];
                        coefs1[i][(nz-1-iz)*nR+iR] = coefs[i][iz*nR+iR];
                    }
                }
            initSplines(coefs1);
        } else
            initSplines(coefs);
        // check symmetry
        int mmax=static_cast<int>(splines.size()/2);
        mysymmetry=mmax==0 ? coord::ST_AXISYMMETRIC : coord::ST_TRIAXIAL;
        for(int m=-mmax; m<=mmax; m++)
            if(!splines[m+mmax].isEmpty()) {
                if(m<0 || (m>0 && m%2==1))
                    mysymmetry = coord::ST_NONE;//(SymmetryType)(mysymmetry & ~ST_TRIAXIAL & ~ST_ZROTSYM);
            }
    }
}

class PotentialAzimuthalAverageIntegrand: public math::IFunctionNoDeriv {
public:
    PotentialAzimuthalAverageIntegrand(const BasePotential& _pot, double _R, double _z, int _m) :
    pot(_pot), R(_R), z(_z), m(_m) {};
    virtual double value(double phi) const {
        double val;
        pot.eval(coord::PosCyl(R, z, phi), &val);
        return val * (m==0 ? 1 : m>0 ? 2*cos(m*phi) : 2*sin(-m*phi));
    }
private:
    const BasePotential& pot;
    double R, z, m;
};

double CylSplineExpOld::computePhi_m(double R, double z, int m, const BasePotential& potential) const
{
    if(potential.name()==DirectPotential::myName()) {
        return dynamic_cast<const DirectPotential&>(potential).Phi_m(R, z, m);
    } else {  
        // compute azimuthal Fourier harmonic coefficient for the given m
        // by averaging the input potential over phi
        if(R==0 && m!=0) return 0;
        double phimax=(potential.symmetry() & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL ? M_PI_2 : 2*M_PI;
        return math::integrate(PotentialAzimuthalAverageIntegrand(potential, R, z, m),
            0, phimax, EPSREL_POTENTIAL_INT) / phimax;
    }
}

void CylSplineExpOld::initPot(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
    const BasePotential& potential, double radius_min, double radius_max, double z_min, double z_max)
{
    if( _Ncoefs_R<CYLSPLINE_MIN_GRID_SIZE || _Ncoefs_z<CYLSPLINE_MIN_GRID_SIZE || 
        _Ncoefs_R>CYLSPLINE_MAX_GRID_SIZE || _Ncoefs_z>CYLSPLINE_MAX_GRID_SIZE || 
        _Ncoefs_phi>CYLSPLINE_MAX_ANGULAR_HARMONIC)
        throw std::invalid_argument("CylSplineExp: invalid grid size");
    mysymmetry = potential.symmetry();
    // whether we need to compute potential at z<0 independently from z>0
    bool zsymmetry= (mysymmetry & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL;
    int mmax = (mysymmetry & coord::ST_AXISYMMETRIC) == coord::ST_AXISYMMETRIC ? 0 : _Ncoefs_phi;
    // if triaxial symmetry, do not use sine terms which correspond to m<0
    int mmin = (mysymmetry & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL ? 0 :-1;
    // if triaxial symmetry, use only even m
    int mstep= (mysymmetry & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL ? 2 : 1;
    if(radius_max==0 || radius_min==0) {
        double totalmass = potential.totalMass();
        if(!math::isFinite(totalmass))
            throw std::invalid_argument("CylSplineExp: source density model has infinite mass");
        if(radius_max==0)
            radius_max = getRadiusByMass(potential, totalmass*(1-1.0/(_Ncoefs_R*_Ncoefs_z)));
        if(!math::isFinite(radius_max)) 
            throw std::runtime_error("CylSplineExp: cannot determine outer radius for the grid");
        if(radius_min==0) 
            radius_min = std::min<double>(radius_max/_Ncoefs_R, 
                getRadiusByMass(potential, totalmass/(_Ncoefs_R*_Ncoefs_z)));
        if(!math::isFinite(radius_min)) 
            //radius_min = radius_max/_Ncoefs_R;
            throw std::runtime_error("CylSplineExp: cannot determine inner radius for the grid");
    }
    std::vector<double> splineRad = math::createNonuniformGrid(_Ncoefs_R, radius_min, radius_max, true);
    grid_R = splineRad;
    if(z_max==0) z_max=radius_max;
    if(z_min==0) z_min=radius_min;
    z_min = std::min<double>(z_min, z_max/_Ncoefs_z);
    splineRad = math::createNonuniformGrid(_Ncoefs_z, z_min, z_max, true);
    grid_z.assign(2*_Ncoefs_z-1,0);
    for(size_t i=0; i<_Ncoefs_z; i++) {
        grid_z[_Ncoefs_z-1-i] =-splineRad[i];
        grid_z[_Ncoefs_z-1+i] = splineRad[i];
    }
    size_t Ncoefs_R=grid_R.size();
    size_t Ncoefs_z=grid_z.size();
    std::vector< std::vector<double> > coefs(2*mmax+1);
    for(int m=mmax*mmin; m<=mmax; m+=mstep) {
        coefs[mmax+m].assign(Ncoefs_R*Ncoefs_z,0);
    }
    int numPoints = Ncoefs_R * (Ncoefs_z/2+1);  // total # of points in 2d grid
    bool badValueEncountered = false;
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {  // combined index variable
        unsigned int iR = ind % Ncoefs_R;
        unsigned int iz = ind / Ncoefs_R;
        for(int m=mmax*mmin; m<=mmax; m+=mstep) {
            try{
                double val = computePhi_m(grid_R[iR], grid_z[Ncoefs_z/2+iz], m, potential);
                coefs[mmax+m][(Ncoefs_z/2+iz)*Ncoefs_R+iR] = val;
                if(!zsymmetry && iz>0 && math::isFinite(val))   // no symmetry about x-y plane
                    // compute potential at -z independently
                    val = computePhi_m(grid_R[iR], grid_z[Ncoefs_z/2-iz], m, potential);
                coefs[mmax+m][(Ncoefs_z/2-iz)*Ncoefs_R+iR] = val;
                if(!math::isFinite(val)) {
                    errorMsg = "Invalid potential value "
                        "at R=" + utils::convertToString(grid_R[iR]) +
                        ", z="  + utils::convertToString(grid_z[iz+Ncoefs_z/2])+
                        ", m="  + utils::convertToString(m);
                    badValueEncountered = true;
                }
            }
            catch(std::exception& e) {
                errorMsg = e.what();
                badValueEncountered = true;
            }
        }
    }
    if(badValueEncountered)
        throw std::runtime_error("Error in CylSplineExp: "+errorMsg);
    initSplines(coefs);
}

void CylSplineExpOld::initSplines(const std::vector< std::vector<double> > &coefs)
{
    size_t Ncoefs_R=grid_R.size();
    size_t Ncoefs_z=grid_z.size();
    int mmax=coefs.size()/2;
    assert(coefs[mmax].size()==Ncoefs_R*Ncoefs_z);  // check that at least m=0 coefficients are present
    assert(Ncoefs_R>=CYLSPLINE_MIN_GRID_SIZE && Ncoefs_z>=CYLSPLINE_MIN_GRID_SIZE);
    // compute multipole coefficients for extrapolating the potential and forces beyond the grid,
    // by fitting them to the potential at the grid boundary
    C00=C20=C22=C40=0;
    bool fitm2=mmax>=2 && coefs[mmax+2].size()==Ncoefs_R*Ncoefs_z;  // whether to fit m=2
    size_t npointsboundary=2*(Ncoefs_R-1)+Ncoefs_z;
    math::Matrix<double> X0(npointsboundary, 3); // matrix of coefficients  for m=0
    std::vector<double> Y0(npointsboundary);     // vector of r.h.s. values for m=0
    std::vector<double> W0(npointsboundary);     // vector of weights
    std::vector<double> X2(npointsboundary);     // vector of coefficients  for m=2
    std::vector<double> Y2(npointsboundary);     // vector of r.h.s. values for m=2
    bool allzero = true;
    for(size_t i=0; i<npointsboundary; i++) {
        size_t iR=i<2*Ncoefs_R ? i/2 : Ncoefs_R-1;
        size_t iz=i<2*Ncoefs_R ? (i%2)*(Ncoefs_z-1) : i-2*Ncoefs_R+1;
        double R=grid_R[iR];
        double z=grid_z[iz];
        double oneoverr=1/sqrt(R*R+z*z);
        Y0[i] = coefs[mmax][iz*Ncoefs_R+iR];
        X0(i, 0) = oneoverr;
        X0(i, 1) = pow(oneoverr,5.0) * (2*z*z-R*R);
        X0(i, 2) = pow(oneoverr,9.0) * (8*pow(z,4.0)-24*z*z*R*R+3*pow(R,4.0));
        // weight proportionally to the value of potential itself
        // (so that we minimize sum of squares of relative differences)
        W0[i] = 1.0/pow_2(Y0[i]);
        allzero &= (Y0[i] == 0);
        if(fitm2) {
            X2[i] = R*R*pow(oneoverr,5.0);
            Y2[i] = coefs[mmax+2][iz*Ncoefs_R+iR];
        }
    }
    // fit m=0 by three parameters
    std::vector<double> fit;
    math::linearMultiFit(X0, Y0, &W0, fit);
    C00 = fit[0];  // C00 ~= -Mtotal
    C20 = fit[1];
    C40 = fit[2];
    // fit m=2 if necessary
    if(fitm2)
        C22 = math::linearFitZero(X2, Y2, NULL);
    // assign Rscale so that it approximately equals -Mtotal/Phi(r=0)
    Rscale = C00 / coefs[mmax][(Ncoefs_z/2)*Ncoefs_R];
    if(allzero)  // empty model, mass=0
        Rscale = 1.;
    else if(Rscale<=0 || !math::isFinite(Rscale+C00+C20+C40+C22))
        throw std::runtime_error("CylSplineExp: cannot determine scaling factor");
        //Rscale=std::min<double>(grid_R.back(), grid_z.back())*0.5; // shouldn't occur?
#ifdef DEBUGPRINT
    my_message(FUNCNAME,  "Rscale="+convertToString(Rscale)+
        ", C00="+convertToString(C00)+", C20="+convertToString(C20)+
        ", C22="+convertToString(C22)+", C40="+convertToString(C40));
#endif

    std::vector<double> grid_Rscaled(Ncoefs_R);
    std::vector<double> grid_zscaled(Ncoefs_z);
    for(size_t i=0; i<Ncoefs_R; i++) {
        grid_Rscaled[i] = log(1+grid_R[i]/Rscale);
    }
    for(size_t i=0; i<Ncoefs_z; i++) {
        grid_zscaled[i] = log(1+fabs(grid_z[i])/Rscale)*(grid_z[i]>=0?1:-1);
    }
    splines.resize(coefs.size());
    math::Matrix<double> values(Ncoefs_R, Ncoefs_z);
    for(size_t m=0; m<coefs.size(); m++) {
        if(coefs[m].size() != Ncoefs_R*Ncoefs_z) 
            continue;
        allzero=true;
        for(size_t iR=0; iR<Ncoefs_R; iR++) {
            for(size_t iz=0; iz<Ncoefs_z; iz++) {
                double scaling = sqrt(pow_2(Rscale)+pow_2(grid_R[iR])+pow_2(grid_z[iz]));
                double val = coefs[m][iz*Ncoefs_R+iR] * scaling;
                values(iR, iz) = val;
                allzero &= (val==0);
            }
        }
        if(!allzero)
            // specify derivative at R=0 to be zero
            splines[m] = math::CubicSpline2d(grid_Rscaled, grid_zscaled, values, 0, NAN, NAN, NAN);
    }
}

void CylSplineExpOld::getCoefs(std::vector<double>& gridR,
    std::vector<double>& gridz, std::vector< std::vector<double> >& coefs) const
{
    gridR = grid_R;
    gridz = grid_z;
    coefs.resize(splines.size());
    for(size_t m=0; m<splines.size(); m++)
        if(!splines[m].isEmpty())
            coefs[m].assign(grid_z.size()*grid_R.size(), 0);
    for(size_t iz=0; iz<grid_z.size(); iz++)
        for(size_t iR=0; iR<grid_R.size(); iR++) {
            double Rscaled = log(1 + grid_R[iR] / Rscale);
            double zscaled = log(1 + fabs(grid_z[iz]) / Rscale) * (grid_z[iz]>=0?1:-1);
            for(size_t m=0; m<splines.size(); m++)
                if(!splines[m].isEmpty()) {
                    double scaling = sqrt(pow_2(Rscale)+pow_2(grid_R[iR])+pow_2(grid_z[iz]));
                    coefs[m][iz*grid_R.size()+iR] = splines[m].value(Rscaled, zscaled) / scaling;
                }
        }
}

void CylSplineExpOld::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    if(pos.R>=grid_R.back() || fabs(pos.z)>=grid_z.back()) 
    {   // fallback mechanism for extrapolation beyond grid definition region
        double Z2 = pow_2(pos.z), R2 = pow_2(pos.R), r2 = R2+Z2;
        if(r2>1e100) {  // special, honorary treatment...
            if(potential!=NULL)
                *potential = 0;
            if(grad!=NULL)
                grad->dR = grad->dz = grad->dphi = 0;
            if(hess!=NULL)
                hess->dR2 = hess->dz2 = hess->dphi2 = hess->dRdz = hess->dRdphi = hess->dzdphi = 0;
            return;
        }
        double R2r2 = R2/r2, Z2r2 = Z2/r2;
        double cos2phi = cos(2*pos.phi);
        double sin2phi = sin(2*pos.phi);
        double oneoverr3 = 1 / (r2 * sqrt(r2));
        double mulC2 = (2 * Z2r2 - R2r2) * C20 + R2r2 * cos2phi * C22;
        double mulC4 = (35 * pow_2(Z2r2) - 30 * Z2r2 + 3) / r2 * C40;
        if(potential!=NULL)
            *potential = (C00 * r2 + mulC2 + mulC4) * oneoverr3;
        if(grad!=NULL) {
            double commonTerm = C00 + (5 * mulC2 + 9 * mulC4) / r2;
            grad->dR   = -pos.R * oneoverr3 *
                (commonTerm + (2*C20 - 2*C22*cos2phi + C40*(60*Z2r2-12)/r2) / r2);
            grad->dz   = -pos.z * oneoverr3 *
                (commonTerm + (-4*C20 + C40*(48-80*Z2r2)/r2) / r2);
            grad->dphi = -2*sin2phi * R2r2 * oneoverr3 * C22;
        }
        if(hess!=NULL) {
            double oneoverr4 = 1/pow_2(r2);
            double commonTerm1 = 3 * C00 + (35 * mulC2 + 99 * mulC4) / r2;
            double commonTerm2 = C00 + 5 * mulC2 / r2;
            hess->dR2  = oneoverr3 * ( commonTerm1 * R2r2 - commonTerm2
                + (20 * R2r2 - 2) * (C20 - C22 * cos2phi) / r2 
                + (-207 * pow_2(R2r2) + 1068 * R2r2 * Z2r2 - 120 * pow_2(Z2r2) ) * C40 * oneoverr4 );
            hess->dz2  = oneoverr3 * ( commonTerm1 * Z2r2 - commonTerm2
                + (-40 * Z2r2 + 4) * C20 / r2 
                + (-75 * pow_2(R2r2) + 1128 * R2r2 * Z2r2 - 552 * pow_2(Z2r2) ) * C40 * oneoverr4 );
            hess->dRdz = oneoverr3 * pos.R * pos.z / r2 * ( commonTerm1
                - 10 * (C20 + C22 * cos2phi) / r2
                + (228 * R2r2 + 48 * Z2r2) * C40 * oneoverr4 );
            double commonTerm3 = oneoverr3 / r2 * sin2phi * C22;
            hess->dRdphi = pos.R * commonTerm3 * (10 * R2r2 - 4);
            hess->dzdphi = pos.z * commonTerm3 *  10 * R2r2;
            hess->dphi2  = -4 * oneoverr3 * R2r2 * cos2phi * C22;
        }
        return;
    }
    double Rscaled = log(1+pos.R/Rscale);
    double zscaled = log(1+fabs(pos.z)/Rscale)*(pos.z>=0?1:-1);
    double Phi_tot = 0;
    coord::GradCyl sGrad;   // gradient in scaled coordinates
    sGrad.dR = sGrad.dz = sGrad.dphi = 0;
    coord::HessCyl sHess;   // hessian in scaled coordinates
    sHess.dR2 = sHess.dz2 = sHess.dphi2 = sHess.dRdz = sHess.dRdphi = sHess.dzdphi = 0;
    int mmax = splines.size()/2;
    for(int m=-mmax; m<=mmax; m++)
        if(!splines[m+mmax].isEmpty()) {
            double cosmphi = m>=0 ? cos(m*pos.phi) : sin(-m*pos.phi);
            double sinmphi = m>=0 ? sin(m*pos.phi) : cos(-m*pos.phi);
            double Phi_m, dPhi_m_dRscaled, dPhi_m_dzscaled,
            d2Phi_m_dRscaled2, d2Phi_m_dRscaleddzscaled, d2Phi_m_dzscaled2;
            splines[m+mmax].evalDeriv(Rscaled, zscaled, &Phi_m, 
                &dPhi_m_dRscaled, &dPhi_m_dzscaled, &d2Phi_m_dRscaled2,
                &d2Phi_m_dRscaleddzscaled, &d2Phi_m_dzscaled2);
            Phi_tot += Phi_m*cosmphi;
            if(grad!=NULL || hess!=NULL) {
                sGrad.dR   += dPhi_m_dRscaled*cosmphi;
                sGrad.dz   += dPhi_m_dzscaled*cosmphi;
                sGrad.dphi += Phi_m * -m*sinmphi;
            }
            if(hess!=NULL) {
                sHess.dR2    += d2Phi_m_dRscaled2 * cosmphi;
                sHess.dz2    += d2Phi_m_dzscaled2 * cosmphi;
                sHess.dphi2  += Phi_m * -m*m*cosmphi;
                sHess.dRdz   += d2Phi_m_dRscaleddzscaled * cosmphi;
                sHess.dRdphi += dPhi_m_dRscaled* -m*sinmphi;
                sHess.dzdphi += dPhi_m_dzscaled* -m*sinmphi;
            }
        }
    double r2 = pow_2(pos.R) + pow_2(pos.z);
    double S  = 1/sqrt(pow_2(Rscale)+r2);  // scaling
    double dSdr_over_r = -S*S*S;
    double dRscaleddR = 1/(Rscale+pos.R);
    double dzscaleddz = 1/(Rscale+fabs(pos.z));
    if(potential!=NULL)
        *potential = S * Phi_tot;
    if(grad!=NULL) {
        grad->dR   = S * sGrad.dR * dRscaleddR + Phi_tot * pos.R * dSdr_over_r;
        grad->dz   = S * sGrad.dz * dzscaleddz + Phi_tot * pos.z * dSdr_over_r;
        grad->dphi = S * sGrad.dphi;
    }
    if(hess!=NULL)
    {
        double d2RscaleddR2 = -pow_2(dRscaleddR);
        double d2zscaleddz2 = -pow_2(dzscaleddz) * (pos.z>=0?1:-1);
        double d2Sdr2 = (pow_2(Rscale) - 2 * r2) * dSdr_over_r * S * S;
        hess->dR2 =
            (pow_2(pos.R) * d2Sdr2 + pow_2(pos.z) * dSdr_over_r) / r2 * Phi_tot + 
            dSdr_over_r * 2 * pos.R * dRscaleddR * sGrad.dR +
            S * (sHess.dR2*pow_2(dRscaleddR) + sGrad.dR*d2RscaleddR2);
        hess->dz2 =
            (pow_2(pos.z) * d2Sdr2 + pow_2(pos.R) * dSdr_over_r) / r2 * Phi_tot +
            dSdr_over_r * 2 * pos.z * dzscaleddz * sGrad.dz +
            S * (sHess.dz2 * pow_2(dzscaleddz) + sGrad.dz * d2zscaleddz2);
        hess->dRdz =
            (d2Sdr2 - dSdr_over_r) * pos.R * pos.z / r2 * Phi_tot +
            dSdr_over_r * (pos.z * dRscaleddR * sGrad.dR + pos.R * dzscaleddz * sGrad.dz) +
            S * sHess.dRdz * dRscaleddR * dzscaleddz;
        hess->dRdphi =
            dSdr_over_r * pos.R * sGrad.dphi +
            S * dRscaleddR * sHess.dRdphi;
        hess->dzdphi =
            dSdr_over_r * pos.z * sGrad.dphi +
            S * dzscaleddz * sHess.dzdphi;
        hess->dphi2 = S * sHess.dphi2;
    }
}
    
}; // namespace
