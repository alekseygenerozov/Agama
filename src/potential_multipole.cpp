#include "potential_multipole.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cfloat>
#include <cassert>
#include <algorithm>
#include <stdexcept>

#ifdef VERBOSE_REPORT
#include <iostream>
#include <fstream>
#include "utils.h"
#endif

namespace potential {

// internal definitions
namespace{

/// minimum number of terms in sph.-harm. expansion used to compute coefficients
/// of a non-spherical density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)    
const int LMIN_SPHHARM = 16;

/// maximum allowed order of sph.-harm. expansion
const int LMAX_SPHHARM = 64;

/// minimum number of grid nodes
const unsigned int MULTIPOLE_MIN_GRID_SIZE = 2;

/// order of Gauss-Legendre quadrature for computing the radial integrals in Multipole
const unsigned int ORDER_RAD_INT = 15;

/// safety factor to avoid roundoff errors near grid boundaries
const double SAFETY_FACTOR = 8*DBL_EPSILON;

// Helper function to deduce symmetry from the list of non-zero coefficients;
// combine the array of coefficients at different radii into a single array
// and then call the corresponding routine from math::.
// This routine is templated on the number of arrays that it handles:
// each one should have identical number of elements (SH coefs at different radii),
// and each element of each array should have the same dimension (number of SH coefs).
template<int N>
static math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> >* C[N])
{
    unsigned int numRadii=0, numCoefs=0;
    bool correct=true;
    for(int n=0; n<N; n++) {
        if(n==0) {
            numRadii = C[n]->size();
            if(numRadii>0)
                numCoefs = C[n]->at(0).size();
        } else
            correct &= C[n]->size() == numRadii;
    }
    std::vector<double> D(numCoefs);
    for(unsigned int k=0; correct && k<numRadii; k++) {
        for(int n=0; n<N; n++) {
            if(C[n]->at(k).size() == numCoefs)
                // if any of the elements is non-zero,
                // then the combined array will have non-zero c-th element too.
                for(unsigned int c=0; c<numCoefs; c++)
                    D[c] += fabs(C[n]->at(k)[c]);
            else
                correct = false;
        }
    }
    if(!correct)
        throw std::invalid_argument("Error in SphHarmIndices: invalid size of input arrays");
    return math::getIndicesFromCoefs(D);
}

static math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> > &C)
{
    const std::vector< std::vector<double> >* A = &C;
    return getIndicesFromCoefs<1>(&A);
}
static math::SphHarmIndices getIndicesFromCoefs(
    const std::vector< std::vector<double> > &C1, const std::vector< std::vector<double> > &C2)
{
    const std::vector< std::vector<double> >* A[2] = {&C1, &C2};
    return getIndicesFromCoefs<2>(A);
}

// resize the array of coefficients down to the requested order
static void restrictSphHarmCoefs(int lmax, int mmax, std::vector<double>& coefs)
{
    coefs.resize(pow_2(lmax+1), 0);
    math::SphHarmIndices ind(lmax, mmax, coord::ST_NONE);
    for(unsigned int c=0; c<coefs.size(); c++)
        if(abs(ind.index_m(c))>mmax)
            coefs[c] = 0;
}

// ------- Spherical-harmonic expansion of density or potential ------- //
// The routine `computeSphHarmCoefs` can work with both density and potential classes,
// computes the sph-harm expansion for either density (in the first case),
// or potential and its r-derivative (in the second case).
// To avoid code duplication, the function that actually retrieves the relevant quantity
// is separated into a dedicated routine `storeValue`, which stores either one or two
// values for each input point. The `computeSphHarmCoefsSph` routine is templated on both
// the type of input data and the number of quantities stored for each point.

template<class BaseDensityOrPotential>
void storeValue(const BaseDensityOrPotential& src,
    const coord::PosCyl& pos, double values[], int arraySize);

template<>
inline void storeValue(const BaseDensity& src,
    const coord::PosCyl& pos, double values[], int) {
    *values = src.density(pos);
}

template<>
inline void storeValue(const BasePotential& src,
    const coord::PosCyl& pos, double values[], int arraySize) {
    coord::GradCyl grad;
    src.eval(pos, values, &grad);
    double r = hypot(pos.R, pos.z);
    values[arraySize] = grad.dR * pos.R/r + grad.dz * pos.z/r;
}

template<class BaseDensityOrPotential, int NQuantities>
static void computeSphHarmCoefs(const BaseDensityOrPotential& src, 
    const math::SphHarmIndices& ind, const std::vector<double>& radii,
    std::vector< std::vector<double> > * coefs[])
{
    unsigned int numPointsRadius = radii.size();
    if(numPointsRadius<1)
        throw std::invalid_argument("computeSphHarmCoefs: radial grid size too small");
    //  initialize sph-harm transform
    math::SphHarmTransformForward trans(ind);

    // 1st step: collect the values of input quantities at a 2d grid in (r,theta);
    // loop over radii and angular directions, using a combined index variable for better load balancing.
    int numSamplesAngles = trans.size();  // size of array of density values at each r
    int numSamplesTotal  = numSamplesAngles * numPointsRadius;
    std::vector<double> values(numSamplesTotal * NQuantities);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numSamplesTotal; n++) {
        int indR    = n / numSamplesAngles;  // index in radial grid
        int indA    = n % numSamplesAngles;  // combined index in angular direction (theta,phi)
        double rad  = radii[indR];
        double z    = rad * trans.costheta(indA);
        double R    = sqrt(rad*rad - z*z);
        double phi  = trans.phi(indA);
        storeValue(src, coord::PosCyl(R, z, phi),
            &values[indR * numSamplesAngles + indA], numSamplesTotal);
    }

    // 2nd step: transform these values to spherical-harmonic expansion coefficients at each radius
    for(int q=0; q<NQuantities; q++) {
        coefs[q]->resize(numPointsRadius);
        for(unsigned int indR=0; indR<numPointsRadius; indR++) {
            coefs[q]->at(indR).assign(ind.size(), 0);
            trans.transform(&values[indR * numSamplesAngles + q * numSamplesTotal],
                &coefs[q]->at(indR).front());
            math::eliminateNearZeros(coefs[q]->at(indR));
        }
    }
}

// transform an N-body snapshot to an array of spherical-harmonic coefficients:
// input particles are sorted in radius, and for each k-th particle the array of
// sph.-harm. functions Y_lm(theta_k, phi_k) times the particle mass is computed
// and stored in the output array. The indexing of this array is reversed w.r.t.
// the one used for potential or density coefs, namely:
// C_lm(particle_k) = coefs[SphHarmIndices::index(l,m)][k].
// This saves memory, since only the arrays for harmonic coefficients allowed
// by the indexing scheme are allocated and returned.
template<typename ParticleT>
static void computeSphericalHarmonics(
    const particles::PointMassArray<ParticleT> &points,
    const math::SphHarmIndices &ind,
    std::vector<double> &pointRadii,
    std::vector< std::vector<double> > &coefs)
{
    unsigned int nbody = points.size();
    // sort points in radius
    std::vector<std::pair<double, unsigned int> > sortOrder(nbody);
    for(unsigned int k=0; k<nbody; k++)
        sortOrder[k] = std::make_pair(toPosSph(points.point(k)).r, k);
    std::sort(sortOrder.begin(), sortOrder.end());

    // allocate space
    pointRadii.reserve(nbody);
    coefs.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
            coefs[ind.index(l, m)].reserve(nbody);
    std::vector<double> tmpLeg(ind.lmax+1), tmpTrig(ind.mmax*2);
    bool needSine = ind.mmin()<0;

    // compute Y_lm for each point
    for(unsigned int i=0; i<nbody; i++) {
        unsigned int k = sortOrder[i].second;
        const coord::PosCyl pos = toPosCyl(points.point(k));
        double r   = hypot(pos.R, pos.z);
        double tau = pos.z / (r + pos.R);
        const double mass = points.mass(k);
        if(mass==0)
            continue;
        if(r==0)
            throw std::runtime_error("computeSphericalHarmonics: no massive particles at r=0 allowed");
        pointRadii.push_back(r);
        math::trigMultiAngle(pos.phi, ind.mmax, needSine, &tmpTrig.front());
        for(int m=0; m<=ind.mmax; m++) {
            math::sphHarmArray(ind.lmax, m, tau, &tmpLeg.front());
            for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
                coefs[ind.index(l, m)].push_back(mass * tmpLeg[l-m] * 2*M_SQRTPI *
                    (m==0 ? 1 : M_SQRT2 * tmpTrig[m-1]));
            if(needSine && m>0)
                for(int l=ind.lmin(-m); l<=ind.lmax; l+=ind.step)
                    coefs[ind.index(l, -m)].push_back(mass * tmpLeg[l-m] * 2*M_SQRTPI *
                        M_SQRT2 * tmpTrig[ind.mmax+m-1]);
        }
    }
}

static void computePotentialCoefsFromPoints(
    const std::vector<double> &pointRadii,
    const std::vector< std::vector<double> > &coefs,
    double smoothing,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi)
{
    unsigned int nbody = pointRadii.size();
    unsigned int gridSizeR = gridRadii.size();
    std::vector<double> pointLogRadii(nbody);
    for(unsigned int k=0; k<nbody; k++)
        pointLogRadii[k] = log(pointRadii[k]);

    // values of interior/exterior potential at particle radii:
    // each one is represented as a product of 'normalization' and 'value',
    // where norm is always positive and its log is being fit, and value is fit directly
    std::vector<double> PintN(nbody), PextN(nbody); //&PextN = PintN;
    std::vector<double> PintV(nbody), PextV(nbody); //&PextV = PintV;
    
    // scaled grid radii and interior/exterior potential at grid points
    std::vector<double> gridLogRadii(gridSizeR);
    for(unsigned int i=0; i<gridSizeR; i++)
        gridLogRadii[i] = log(gridRadii[i]);
    if(pointRadii[0] < gridRadii[0]) {  // need to add extra scaled grid node at front
        gridLogRadii.insert(gridLogRadii.begin(), pointLogRadii[0]);
    }
    if(pointRadii.back() > gridRadii.back()) {  // add extra node at back
        gridLogRadii.push_back(pointLogRadii.back());
    }

    // allocate space for output
    Phi.resize(gridSizeR);
    dPhi.resize(gridSizeR);
    for(unsigned int i=0; i<gridSizeR; i++) {
        Phi [i].resize(coefs.size());
        dPhi[i].resize(coefs.size());
    }

    // create the instance of smoothing spline creator
    math::SplineApprox approx(pointLogRadii, gridLogRadii);

    // loop over non-trivial SH indices
    for(unsigned int c=0; c<coefs.size(); c++) {
        if(coefs[c].empty())
            continue;
        int l = math::SphHarmIndices::index_l(c);
        // compute the interior and exterior potential coefficients at each particle's radius:
        //   Pint_{l,m}(r_k) = r_k^{-l-1} \sum_{i=0}^{k}   C_{l,m; i} r_i^l ,
        //   Pext_{l,m}(r_k) = r_k^l      \sum_{i=k}^{N-1} C_{l,m; i} r_i^{-1-l} ,
        double val = 0, norm = 0;
        for(unsigned int k=0; k<nbody; k++) {
            if(k>0) {
                val  *= math::powInt(pointRadii[k-1] / pointRadii[k], l+1);
                norm *= math::powInt(pointRadii[k-1] / pointRadii[k], l+1);
            }
            val  += coefs[c][k] / pointRadii[k];
            norm += coefs[0][k] / pointRadii[k];
            PintV[k] = val / norm;
            PintN[k] = log(norm);
        }
        // two-step procedure for each of the fitted values:
        // first compute the smoothing spline values at grid points and its derivs at both ends,
        // then create an ordinary cubic spline from this data, which will be later used 
        // to compute values at arbitrary r
        double derLeft, derRight;
        std::vector<double> tmp;  // temp.storage for spline values
        approx.fitDataOversmooth(PintV, smoothing, tmp, derLeft, derRight);
        math::CubicSpline SintV(gridLogRadii, tmp, derLeft, derRight);
        approx.fitDataOversmooth(PintN, smoothing, tmp, derLeft, derRight);
        math::CubicSpline SintN(gridLogRadii, tmp, derLeft, derRight);

        val = norm = 0;
        for(unsigned int k=nbody; k>0; k--) {
            if(k<nbody) {
                val  *= math::powInt(pointRadii[k-1] / pointRadii[k], l);
                norm *= math::powInt(pointRadii[k-1] / pointRadii[k], l);
            }
            val  += coefs[c][k-1] / pointRadii[k-1];
            norm += coefs[0][k-1] / pointRadii[k-1];
            PextV[k-1] = val / norm;
            PextN[k-1] = log(norm);
        }
        approx.fitDataOversmooth(PextV, smoothing, tmp, derLeft, derRight);
        math::CubicSpline SextV(gridLogRadii, tmp, derLeft, derRight);
        approx.fitData(PextN, 0, tmp, derLeft, derRight);
        math::CubicSpline SextN(gridLogRadii, tmp, derLeft, derRight);

        // Finally, put together the interior and exterior coefs to compute 
        // the potential and its radial derivative for each spherical-harmonic term
        double mul = -1. / (2*l+1);
        for(unsigned int i=0; i<gridSizeR; i++) {
            double CiN, CiV, dCiN, dCiV;
            SintN.evalDeriv(log(gridRadii[i]), &CiN, &dCiN);
            SintV.evalDeriv(log(gridRadii[i]), &CiV, &dCiV);
            double CeN, CeV, dCeN, dCeV;
            SextN.evalDeriv(log(gridRadii[i]), &CeN, &dCeN);
            SextV.evalDeriv(log(gridRadii[i]), &CeV, &dCeV);
            Phi [i][c] = (exp(CiN) * CiV + exp(CeN) * CeV) * mul;
            // the derivative may be computed in two possible ways
#if 0
            dPhi[i][c] = (exp(CiN) * CiV * (-1-l) + exp(CeN) * CeV * l) * mul / gridRadii[i];
#else
            dPhi[i][c] = (exp(CiN) * (dCiN*CiV + dCiV) + 
                          exp(CeN) * (dCeN*CeV + dCeV) ) * mul / gridRadii[i];
#endif
        }
#if 0
        std::ofstream strm(("mul"+utils::convertToString(c)).c_str());
        for(unsigned int k=0; k<nbody; k++) {
            double CiN, CiV, dCiN, dCiV;
            SintN.evalDeriv(log(pointRadii[k]), &CiN, &dCiN);
            SintV.evalDeriv(log(pointRadii[k]), &CiV, &dCiV);
            double CeN, CeV, dCeN, dCeV;
            SextN.evalDeriv(log(pointRadii[k]), &CeN, &dCeN);
            SextV.evalDeriv(log(pointRadii[k]), &CeV, &dCeV);
            strm << pointRadii[k] << '\t' << 
            PintV[k] << '\t' << PextV[k] << '\t' << 
            PintN[k] << '\t' << PextN[k] << '\t' << 
            CiV << '\t' << CeV << '\t' << CiN << '\t' << CeN << '\t' << 
            (mul * (-(l+1)*exp(CiN)*CiV + l*exp(CeN)*CeV) / pointRadii[k]) << '\t' <<
            (mul * (exp(CiN)*(dCiN*CiV+dCiV) + exp(CeN)*(dCeN*CeV+dCeV)) / pointRadii[k]) << '\n';
        }
        strm << '\n';
       /* for(unsigned int i=0; i<gridLogRadii.size(); i++) {
            strm << exp(gridLogRadii[i]) << '\t' << CintV[i] << '\t' << CextV[i] << '\t' <<
                CintN[i] << '\t' << CextN[i] << '\t' << Phi[i][c] << '\t' << dPhi[i][c] << '\n';
        }*/
#endif
    }
}

}  // internal namespace

// driver functions that call the above templated transformation routine

// density coefs from density
void computeDensityCoefsSph(const BaseDensity& src, 
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector<double> > &output)
{
    std::vector< std::vector<double> > *coefs = &output;
    computeSphHarmCoefs<BaseDensity, 1>(src, ind, gridRadii, &coefs);
}

// potential coefs from potential
void computePotentialCoefsSph(const BasePotential &src,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi)
{
    std::vector< std::vector<double> > *coefs[2] = {&Phi, &dPhi};
    computeSphHarmCoefs<BasePotential, 2>(src, ind, gridRadii, coefs);    
}

// potential coefs from density:
// core function to solve Poisson equation in spherical harmonics for a smooth density profile
void computePotentialCoefsSph(const BaseDensity& dens, 
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector<double> >& Phi,
    std::vector< std::vector<double> >& dPhi)
{
    int gridSizeR = gridRadii.size();
    if(gridSizeR < (int)MULTIPOLE_MIN_GRID_SIZE)
        throw std::invalid_argument("computePotentialCoefsSph: radial grid size too small");
    for(int k=0; k<gridSizeR; k++)
        if(gridRadii[k] <= (k==0 ? 0 : gridRadii[k-1]))
            throw std::invalid_argument("computePotentialCoefsSph: "
                "radii of grid points must be positive and sorted in increasing order");

    // several intermediate arrays are aliased with the output arrays,
    // but are denoted by different names to clearly mark their identity
    std::vector< std::vector<double> >& Qint = Phi;
    std::vector< std::vector<double> >& Qext = dPhi;
    std::vector< std::vector<double> >& Pint = Phi;
    std::vector< std::vector<double> >& Pext = dPhi;
    Phi .resize(gridSizeR);
    dPhi.resize(gridSizeR);
    for(int k=0; k<gridSizeR; k++) {
        Phi [k].assign(ind.size(), 0);
        dPhi[k].assign(ind.size(), 0);
    }

    // prepare tables for (non-adaptive) integration over radius
    std::vector<double> glx(ORDER_RAD_INT), glw(ORDER_RAD_INT);  // Gauss-Legendre nodes and weights
    math::prepareIntegrationTableGL(0, 1, ORDER_RAD_INT, &glx.front(), &glw.front());

    // prepare SH transformation
    math::SphHarmTransformForward trans(ind);

    // Loop over radial grid segments and compute integrals of rho_lm(r) times powers of radius,
    // for each interval of radii in the input grid (0 <= k < Nr):
    //   Qint[k][l,m] = \int_{r_{k-1}}^{r_k} \rho_{l,m}(r) (r/r_k)^{l+2} dr,  with r_{-1} = 0;
    //   Qext[k][l,m] = \int_{r_k}^{r_{k+1}} \rho_{l,m}(r) (r/r_k)^{1-l} dr,  with r_{Nr} = \infty.
    // Here \rho_{l,m}(r) are the sph.-harm. coefs for density at each radius.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int k=0; k<=gridSizeR; k++) {
        std::vector<double> densValues(trans.size());
        std::vector<double> tmpCoefs(ind.size());
        double rkminus1 = (k>0 ? gridRadii[k-1] : 0);
        double deltaGridR = k<gridSizeR ?
            gridRadii[k] - rkminus1 :  // length of k-th radial segment
            gridRadii.back();          // last grid segment extends to infinity

        // loop over ORDER_RAD_INT nodes of GL quadrature for each radial grid segment
        for(unsigned int s=0; s<ORDER_RAD_INT; s++) {
            double r = k<gridSizeR ?
                rkminus1 + glx[s] * deltaGridR :  // radius inside ordinary k-th segment
                // special treatment for the last segment which extends to infinity:
                // the integration variable is t = r_{Nr-1} / r
                gridRadii.back() / glx[s];

            // collect the values of density at all points of angular grid at the given radius
            for(unsigned int i=0; i<densValues.size(); i++)
                densValues[i] = dens.density(coord::PosCyl(
                    r * sqrt(1-pow_2(trans.costheta(i))), r * trans.costheta(i), trans.phi(i)));

            // compute density SH coefs
            trans.transform(&densValues.front(), &tmpCoefs.front());
            math::eliminateNearZeros(tmpCoefs);

            // accumulate integrals over density times radius in the Qint and Qext arrays
            for(int m=ind.mmin(); m<=ind.mmax; m++) {
                for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    if(k<gridSizeR)
                        // accumulate Qint for all segments except the one extending to infinity
                        Qint[k][c] += tmpCoefs[c] * glw[s] * deltaGridR *
                            math::powInt(r / gridRadii[k], l+2);
                    if(k>0)
                        // accumulate Qext for all segments except the innermost one
                        // (which starts from zero), with a special treatment for last segment
                        // that extends to infinity and has a different integration variable
                        Qext[k-1][c] += glw[s] * tmpCoefs[c] * deltaGridR *
                            (k==gridSizeR ? 1 / pow_2(glx[s]) : 1) * // jacobian of 1/r transform
                            math::powInt(r / gridRadii[k-1], 1-l);
                }
            }
        }
    }

    // Run the summation loop, replacing the intermediate values Qint, Qext
    // with the interior and exterior potential coefficients (stored in the same arrays):
    //   Pint_{l,m}(r) = r^{-l-1} \int_0^r \rho_{l,m}(s) s^{l+2} ds ,
    //   Pext_{l,m}(r) = r^l \int_r^\infty \rho_{l,m}(s) s^{1-l} ds ,
    // In doing so, we use a recurrent relation that avoids over/underflows when
    // dealing with large powers of r, by replacing r^n with (r/r_prev)^n.
    // Finally, compute the total potential and its radial derivative for each SH term.
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);

            // Compute Pint by summing from inside out, using the recurrent relation
            // Pint(r_{k+1}) r_{k+1}^{l+1} = Pint(r_k) r_k^{l+1} + Qint[k] * r_k^{l+2}
            double val = 0;
            for(int k=0; k<gridSizeR; k++) {
                if(k>0)
                    val *= math::powInt(gridRadii[k-1] / gridRadii[k], l+1);
                val += gridRadii[k] * Qint[k][c];
                Pint[k][c] = val;
            }

            // Compute Pext by summing from outside in, using the recurrent relation
            // Pext(r_k) r_k^{-l} = Pext(r_{k+1}) r_{k+1}^{-l} + Qext[k] * r_k^{1-l}
            val = 0;
            for(int k=gridSizeR-1; k>=0; k--) {
                if(k<gridSizeR-1)
                    val *= math::powInt(gridRadii[k] / gridRadii[k+1], l);
                val += gridRadii[k] * Qext[k][c];
                Pext[k][c] = val;
            }

            // Finally, put together the interior and exterior coefs to compute 
            // the potential and its radial derivative for each spherical-harmonic term
            double mul = -4*M_PI / (2*l+1);
            for(int k=0; k<gridSizeR; k++) {
                double tmpPhi = mul * (Pint[k][c] + Pext[k][c]);
                dPhi[k][c]    = mul * (-(l+1)*Pint[k][c] + l*Pext[k][c]) / gridRadii[k];
                // extra step needed because Phi/dPhi and Pint/Pext are aliased
                Phi[k][c]     = tmpPhi;
            }
        }
    }

    // polishing: zero out coefs with small magnitude at each radius
    for(int k=0; k<gridSizeR; k++) {
        math::eliminateNearZeros(Phi[k]);
        math::eliminateNearZeros(dPhi[k]);
    }
}

// potential coefs from N-body points
template<typename ParticleT>
void computePotentialCoefsSph(
    const particles::PointMassArray<ParticleT> &points,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi,
    double smoothing)
{
    unsigned int gridSizeR = gridRadii.size();
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE)
        throw std::invalid_argument("computePotentialCoefsSph: radial grid size too small");
    for(unsigned int k=0; k<gridSizeR; k++)
        if(gridRadii[k] <= (k==0 ? 0 : gridRadii[k-1]))
            throw std::invalid_argument("computePotentialCoefs: "
                "radii of grid points must be positive and sorted in increasing order");
    std::vector<std::vector<double> > harmonics(ind.size());
    std::vector<double> pointRadii;
    computeSphericalHarmonics(points, ind, pointRadii, harmonics);
    computePotentialCoefsFromPoints(pointRadii, harmonics, smoothing, gridRadii, Phi, dPhi);    
}


namespace {  // internal routines

// auto-assign min/max radii of the grid if they were not provided
static void chooseGridRadii(const BaseDensity& src, const unsigned int gridSizeR,
    double& rmin, double& rmax)
{
    if(rmax!=0 && rmin!=0)
        return;
    double rhalf = getRadiusByMass(src, 0.5 * src.totalMass());
    if(!math::isFinite(rhalf))
        throw std::invalid_argument("Multipole: failed to automatically determine grid extent");
    double spacing = 1 + sqrt(20./gridSizeR);  // ratio between consecutive grid nodes
    if(rmax==0)
        rmax = rhalf * pow(spacing,  0.5*gridSizeR);
    if(rmin==0)
        rmin = rhalf * pow(spacing, -0.5*gridSizeR);
#ifdef VERBOSE_REPORT
    std::cout << "Multipole: Grid in r=["<<rmin<<":"<<rmax<<"]\n";
#endif
}

template<typename ParticleT>
static void chooseGridRadii(const particles::PointMassArray<ParticleT>& points,
    unsigned int gridSizeR, double &rmin, double &rmax) 
{
    if(rmin!=0 && rmax!=0)
        return;
    unsigned int Npoints = points.size();
    std::vector<double> radii(Npoints);
    for(unsigned int i=0; i<Npoints; i++)
        radii[i] = toPosSph(points.point(i)).r;
    std::nth_element(radii.begin(), radii.begin() + Npoints/2, radii.end());
    double rhalf = radii[Npoints/2];   // half-mass radius (if all particles have equal mass)
    double spacing = 1 + sqrt(20./gridSizeR);
    int Nmin = log(Npoints+1)/log(2);  // # of points inside the first or outside the last grid node
    if(rmin==0) {
        std::nth_element(radii.begin(), radii.begin() + Nmin, radii.end());
        rmin = std::max(radii[Nmin], rhalf * pow(spacing, -0.5*gridSizeR));
    }
    if(rmax==0) {
        std::nth_element(radii.begin(), radii.end() - Nmin, radii.end());
        rmax = std::min(radii[Npoints-Nmin], rhalf * pow(spacing, 0.5*gridSizeR));
    }
#ifdef VERBOSE_REPORT
    std::cout << "Multipole: Grid in r=["<<rmin<<":"<<rmax<<"]\n";
#endif
}

/** helper function to determine the coefficients for potential extrapolation:
    assuming that 
        Phi(r) = W * (r/r1)^v + U * (r/r1)^s              if s!=v, or
        Phi(r) = W * (r/r1)^v + U * (r/r1)^s * ln(r/r1)   if s==v,
    and given v and the values of Phi(r1), Phi(r2) and dPhi/dr(r1),
    determine the coefficients s, U and W.
    Here v = l for the inward and v = -l-1 for the outward extrapolation.
    This corresponds to the density profile extrapolated as rho ~ r^(s-2).
*/
static void computeExtrapolationCoefs(double Phi1, double Phi2, double dPhi1,
    double r1, double r2, int v, double& s, double& U, double& W)
{
    double lnr = log(r2/r1);
    double A = lnr * (r1*dPhi1 - v*Phi1) / (Phi1 - Phi2 * exp(-v*lnr));
    if(!math::isFinite(A) || A >= 0)
    {   // no solution - output only the main multipole component (with zero Laplacian)
        U = 0;
        s = 0;
        W = Phi1;
        return;
    }
    // find x(A) such that  x = A * (1 - exp(x)),  where  x = (s-v) * ln(r2/r1)
    s = A==-1 ? v : v + (A - math::lambertW(A * exp(A), /*choice of branch*/ A>-1)) / lnr;
    // safeguard against weird slope determination
    if(v>=0 && (!math::isFinite(s) || s<=-1))
        s = 2;  // results in a constant-density core for the inward extrapolation
    if(v<0  && (!math::isFinite(s) || s>=0))
        s = -2; // results in a r^-4 falloff for the outward extrapolation
    if(s != v) {
        U = (r1*dPhi1 - v*Phi1) / (s-v);
        W = (r1*dPhi1 - s*Phi1) / (v-s);
    } else {
        U = r1*dPhi1 - v*Phi1;
        W = Phi1;
    }
}

/** construct asymptotic power-law potential for extrapolation to small or large radii
    from the given spherical-harmonic coefs and their derivatives at two consecutive radii */
static PtrPotential initAsympt(double r1, double r2, 
    const std::vector<double>& Phi1, const std::vector<double>& Phi2,
    const std::vector<double>& dPhi1, bool inner)
{
    unsigned int nc = Phi1.size();
    // limit the number of terms to consider
    const unsigned int lmax = 8;
    nc = std::min<unsigned int>(nc, pow_2(lmax+1));
    std::vector<double> S(nc), U(nc), W(nc);

    // determine the coefficients for potential extrapolation at small and large radii
    for(unsigned int c=0; c<nc; c++) {
        int l = math::SphHarmIndices::index_l(c);
        computeExtrapolationCoefs(Phi1[c], Phi2[c], dPhi1[c], r1, r2, inner ? l : -l-1,
                /*output*/ S[c], U[c], W[c]);
        // TODO: may need to constrain the slope of l>0 harmonics so that it doesn't exceed
        // that of the l=0 one; this is enforced in the constructor of PowerLawMultipole,
        // but the slope should already have been adjusted before computing the coefs U and W.
#ifdef VERBOSE_REPORT
        if(l==0) {
            std::cout << (inner?"Inner":"Outer")<<" slope of density profile is "<<(S[c]-2)<<'\n';
        }
#endif
    }
    return PtrPotential(new PowerLawMultipole(r1, inner, S, U, W));
}


/** transform Fourier components C_m(r, theta) and their derivs to the actual potential.
    C_m is an array of size nq*nm, where nm is the number of azimuthal harmonics 
    (either mmax+1, or 2*mmax+1, depending on the symmetry encoded in ind),
    and nq is the number of quantities to convert: either 1 (only the potential harmonics),
    3 (potential and its two derivs w.r.t. r and theta), or 6 (plus three second derivs w.r.t.
    r^2, r theta, and theta^2), all stored contiguously in the C_m array:
    first come the nm potential harmonics, then nm harmonics for dPhi/dr, and so on.
    How many quantities are processed is determined by grad and hess being NULL or non-NULL.
*/
static void fourierTransformAzimuth(const math::SphHarmIndices& ind, const double phi,
    const double C_m[], double *val, coord::GradSph *grad, coord::HessSph *hess)
{
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;  // number of quantities in C_m
    const bool useSine = ind.mmin()<0 || numQuantities>1;
    const int nm = ind.mmax - ind.mmin() + 1;  // number of azimuthal harmonics in C_m array
    double trig_m[2*LMAX_SPHHARM+1];
    if(ind.mmax>0)
        math::trigMultiAngle(phi, ind.mmax, useSine, trig_m);
    if(val)
        *val = 0;
    if(grad)
        grad->dr = grad->dtheta = grad->dphi = 0;
    if(hess)
        hess->dr2 = hess->dtheta2 = hess->dphi2 =
        hess->drdtheta = hess->drdphi = hess->dthetadphi = 0;
    for(int mm=0; mm<nm; mm++) {
        int m = mm + ind.mmin();
        if(ind.lmin(m)>ind.lmax)
            continue;  // empty harmonic
        double trig  = m==0 ? 1. : m>0 ? trig_m[m-1] : trig_m[ind.mmax-m-1];  // cos or sin
        double dtrig = m==0 ? 0. : m>0 ? -m*trig_m[ind.mmax+m-1] : -m*trig_m[-m-1];
        double d2trig = -m*m*trig;
        if(val)
            *val += C_m[mm] * trig;
        if(grad) {
            grad->dr     += C_m[mm+nm  ] *  trig;
            grad->dtheta += C_m[mm+nm*2] *  trig;
            grad->dphi   += C_m[mm]      * dtrig;
        }
        if(hess) {
            hess->dr2       += C_m[mm+nm*3] *   trig;
            hess->drdtheta  += C_m[mm+nm*4] *   trig;
            hess->dtheta2   += C_m[mm+nm*5] *   trig;
            hess->drdphi    += C_m[mm+nm  ] *  dtrig;
            hess->dthetadphi+= C_m[mm+nm*2] *  dtrig;
            hess->dphi2     += C_m[mm]      * d2trig;
        }
    }
}

/** transform sph.-harm. coefs of potential (C_lm) and its first (dC_lm) and second (d2C_lm)
    derivatives w.r.t. arbitrary function of radius to the value, gradient and hessian of 
    potential in spherical coordinates (w.r.t. the same function of radius).
*/
static void sphHarmTransformInverseDeriv(
    const math::SphHarmIndices& ind, const coord::PosCyl& pos,
    const double C_lm[], const double dC_lm[], const double d2C_lm[],
    double *val, coord::GradSph *grad, coord::HessSph *hess)
{
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;  // number of quantities in C_m
    const int nm = ind.mmax - ind.mmin() + 1;  // number of azimuthal harmonics in C_m array
    const double tau = pos.z / (hypot(pos.R, pos.z) + pos.R);
    // temporary storage for coefficients
    double    C_m[(LMAX_SPHHARM*2+1) * 6];
    double    P_lm[LMAX_SPHHARM+1];
    double   dP_lm_arr[LMAX_SPHHARM+1];
    double  d2P_lm_arr[LMAX_SPHHARM+1];
    double*  dP_lm = numQuantities>=3 ?  dP_lm_arr : NULL;
    double* d2P_lm = numQuantities==6 ? d2P_lm_arr : NULL;
    for(int mm=0; mm<nm; mm++) {
        int m = mm + ind.mmin();
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;  // extra factor sqrt{2} for m!=0 trig fncs
        for(int q=0; q<numQuantities; q++)
            C_m[mm + q*nm] = 0;
        int absm = abs(m);
        math::sphHarmArray(ind.lmax, absm, tau, P_lm, dP_lm, d2P_lm);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m), p = l-absm;
            C_m[mm] += P_lm[p] * C_lm[c] * mul;
            if(numQuantities>=3) {
                C_m[mm + nm  ] +=  P_lm[p] * dC_lm[c] * mul;   // dPhi_m/dr
                C_m[mm + nm*2] += dP_lm[p] *  C_lm[c] * mul;   // dPhi_m/dtheta
            }
            if(numQuantities==6) {
                C_m[mm + nm*3] +=   P_lm[p] * d2C_lm[c] * mul; // d2Phi_m/dr2
                C_m[mm + nm*4] +=  dP_lm[p] *  dC_lm[c] * mul; // d2Phi_m/drdtheta
                C_m[mm + nm*5] += d2P_lm[p] *   C_lm[c] * mul; // d2Phi_m/dtheta2
            }
        }
    }
    fourierTransformAzimuth(ind, pos.phi, C_m, val, grad, hess);
}

// transform potential derivatives from {ln(r), theta} to {R, z}
static inline void transformDerivsSphToCyl(const coord::PosCyl& pos,
    const coord::GradSph &gradSph, const coord::HessSph &hessSph,
    coord::GradCyl *gradCyl, coord::HessCyl *hessCyl)
{
    // abuse the coordinate transformation framework (Sph -> Cyl), where actually
    // in the source grad/hess we have derivs w.r.t. ln(r) instead of r
    const double r2inv = 1 / (pow_2(pos.R) + pow_2(pos.z));
    coord::PosDerivT<coord::Cyl, coord::Sph> der;
    der.drdR = pos.R * r2inv;
    der.drdz = pos.z * r2inv;
    der.dthetadR =  der.drdz;
    der.dthetadz = -der.drdR;
    if(gradCyl)
        *gradCyl = toGrad(gradSph, der);
    if(hessCyl) {
        coord::PosDeriv2T<coord::Cyl, coord::Sph> der2;
        der2.d2rdR2      = pow_2(der.drdz) - pow_2(der.drdR);
        der2.d2rdRdz     = -2 * der.drdR * der.drdz;
        der2.d2rdz2      = -der2.d2rdR2;
        der2.d2thetadR2  =  der2.d2rdRdz;
        der2.d2thetadRdz = -der2.d2rdR2;
        der2.d2thetadz2  = -der2.d2rdRdz;
        *hessCyl = toHess(gradSph, hessSph, der, der2);
    }
}

// perform scaling transformation for the amplitude of potential and its derivatives:
// on entry, pot, grad and hess contain the value, gradient and hessian of an auxiliary quantity
// G[ln(r),...] = Phi[ln(r),...] * sqrt(r^2 + R0^2);
// on output, they are replaced with the value, gradient and hessian of Phi w.r.t. [ln(r),...];
// grad or hess may be NULL, if they are ultimately not needed.
static inline void transformAmplitude(double r, double Rscale,
    double& pot, coord::GradSph *grad, coord::HessSph *hess)
{
    // additional scaling factor for the amplitude: 1 / sqrt(r^2 + R0^2)
    double amp = 1. / hypot(r, Rscale);
    pot *= amp;
    if(!grad)
        return;
    // unscale the amplitude of derivatives, i.e. transform from
    // d [scaledPhi(scaledCoords) * amp] / d[scaledCoords] to d[scaledPhi] / d[scaledCoords]
    double damp = -r*r*amp;  // d amp[ln(r)] / d[ln(r)] / amp^2
    grad->dr = (grad->dr + pot * damp) * amp;
    grad->dtheta *= amp;
    grad->dphi   *= amp;
    if(hess) {
        hess->dr2 = (hess->dr2 + (2 * grad->dr + (2 - pow_2(r * amp)) * pot ) * damp) * amp;
        hess->drdtheta = (hess->drdtheta + grad->dtheta * damp) * amp;
        hess->drdphi   = (hess->drdphi   + grad->dphi   * damp) * amp;
        hess->dtheta2 *= amp;
        hess->dphi2   *= amp;
        hess->dthetadphi *= amp;
    }
}

}  // end internal namespace

//------ Spherical-harmonic expansion of density ------//

PtrDensity DensitySphericalHarmonic::create(
    const BaseDensity& dens, int lmax, int mmax, 
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || rmax<=rmin)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of min/max grid radii");
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(dens) ? 0 : std::max<int>(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(dens) ? 0 : std::max<int>(mmax, LMIN_SPHHARM);
    std::vector<std::vector<double> > coefs;
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    computeDensityCoefsSph(dens,
        math::SphHarmIndices(lmax_tmp, mmax_tmp, dens.symmetry()),
        gridRadii, coefs);
    // now resize the coefficients back to the requested order
    for(unsigned int k=0; k<gridSizeR; k++)
        restrictSphHarmCoefs(lmax, mmax, coefs[k]);
    return PtrDensity(new DensitySphericalHarmonic(gridRadii, coefs));
}

DensitySphericalHarmonic::DensitySphericalHarmonic(const std::vector<double> &gridRadii,
    const std::vector< std::vector<double> > &coefs) :
    BaseDensity(), ind(getIndicesFromCoefs(coefs))
{
    unsigned int gridSizeR = gridRadii.size();
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || gridSizeR != coefs.size())
        throw std::invalid_argument("DensitySphericalHarmonic: input arrays are empty");
    for(unsigned int n=0; n<gridSizeR; n++)
        if(coefs[n].size() != ind.size())
            throw std::invalid_argument("DensitySphericalHarmonic: incorrect size of coefficients array");
    if(ind.lmax>LMAX_SPHHARM)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");

    // We check (and correct if necessary) the logarithmic slopes of multipole components
    // at the innermost and outermost grid radii, to ensure correctly behaving extrapolation.
    // slope = (1/rho) d(rho)/d(logr), is usually negative (at least at large radii);
    // put constraints on min inner and max outer slopes:
    const double minDerivInner=-2.8, maxDerivOuter=-2.2;
    const double maxDerivInner=20.0, minDerivOuter=-20.;
    // Note that the inner slope less than -2 leads to a divergent potential at origin,
    // but the enclosed mass is still finite if slope is greater than -3;
    // similarly, outer slope greater than -3 leads to a divergent total mass,
    // but the potential tends to a finite limit as long as the slope is less than -2.
    // Both these 'dangerous' semi-infinite regimes are allowed here, but likely may result
    // in problems elsewhere.
    // The l>0 components must not have steeper/shallower slopes than the l=0 component.
    innerSlope.assign(ind.size(), minDerivInner);
    outerSlope.assign(ind.size(), maxDerivOuter);
    spl.resize(ind.size());
    std::vector<double> gridLogR(gridSizeR), tmparr(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridLogR[k] = log(gridRadii[k]);

    // set up 1d splines in radius for each non-trivial (l,m) coefficient
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        if(ind.lmin(m) > ind.lmax)
            continue;
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            //  determine asymptotic slopes of density profile at large and small r
            double derivInner = log(coefs[1][c] / coefs[0][c]) / log(gridRadii[1] / gridRadii[0]);
            double derivOuter = log(coefs[gridSizeR-1][c] / coefs[gridSizeR-2][c]) /
                log(gridRadii[gridSizeR-1] / gridRadii[gridSizeR-2]);
            if( derivInner > maxDerivInner)
                derivInner = maxDerivInner;
            if( derivOuter < minDerivOuter)
                derivOuter = minDerivOuter;
            if(coefs.front()[c] == 0)
                derivInner = 0;
            if(coefs.back() [c] == 0)
                derivOuter = 0;
            if(!math::isFinite(derivInner) || derivInner < innerSlope[0])
                derivInner = innerSlope[0];  // works even for l==0 since we have set it up in advance
            if(!math::isFinite(derivOuter) || derivOuter > outerSlope[0])
                derivOuter = outerSlope[0];
            innerSlope[c] = derivInner;
            outerSlope[c] = derivOuter;
            for(unsigned int k=0; k<gridSizeR; k++)
                tmparr[k] = coefs[k][c];
            spl[c] = math::CubicSpline(gridLogR, tmparr, 
                derivInner * coefs.front()[c],
                derivOuter * coefs.back() [c]);
        }
    }
}

void DensitySphericalHarmonic::getCoefs(
    std::vector<double> &radii, std::vector< std::vector<double> > &coefs) const
{
    radii.resize(spl[0].xvalues().size());
    for(unsigned int k=0; k<radii.size(); k++)
        radii[k] = exp(spl[0].xvalues()[k]);
    computeDensityCoefsSph(*this, ind, radii, coefs);
}

double DensitySphericalHarmonic::densityCyl(const coord::PosCyl &pos) const
{
    double coefs[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double tmparr[3*(LMAX_SPHHARM+1)];
    double logr = log( pow_2(pos.R) + pow_2(pos.z) ) * 0.5;
    double logrmin = spl[0].xmin(), logrmax = spl[0].xmax();
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            if(logr<logrmin) {
                double val = spl[c](logrmin);
                coefs[c] = val==0 ? 0 : val * exp( (logr-logrmin)*innerSlope[c]);
            } else if(logr>logrmax) {
                double val = spl[c](logrmax);
                coefs[c] = val==0 ? 0 : val * exp( (logr-logrmax)*outerSlope[c]);
            } else
                coefs[c] = spl[c](logr);
        }
    double tau = pos.z / (hypot(pos.R, pos.z) + pos.R);
    return math::sphHarmTransformInverse(ind, coefs, tau, pos.phi, tmparr);
}


//----- declarations of two multipole potential interpolators -----//

class MultipoleInterp1d: public BasePotentialCyl {
public:
    /** construct interpolating splines from the values and derivatives of harmonic coefficients */
    MultipoleInterp1d(
        const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return "MultipoleInterp1d"; };
private:
    /// indexing scheme for sph.-harm. coefficients
    math::SphHarmIndices ind;
    /// interpolation splines in log(r) for each {l,m} sph.-harm. component of potential
    std::vector<math::QuinticSpline> spl;
    /// characteristic radius for amplitude scaling transformation
    double Rscale;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

class MultipoleInterp2d: public BasePotentialCyl {
public:
    /** construct interpolating splines from the values and derivatives of harmonic coefficients */
    MultipoleInterp2d(
        const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return "MultipoleInterp2d"; }
private:
    /// indexing scheme for sph.-harm. coefficients
    math::SphHarmIndices ind;
    /// 2d interpolation splines in meridional plane for each azimuthal harmonic (m) component
    std::vector<math::QuinticSpline2d> spl;
    /// characteristic radius for amplitude scaling transformation
    double Rscale;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

//------ the wrapper class for multipole potential ------//

template<class BaseDensityOrPotential>
static PtrPotential createMultipole(
    const BaseDensityOrPotential& src,
    int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || (rmax!=0 && rmax<=rmin))
        throw std::invalid_argument("Multipole: invalid grid parameters");
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");
    chooseGridRadii(src, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(src) ? 0 : std::max<int>(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, LMIN_SPHHARM);
    std::vector<std::vector<double> > Phi, dPhi;
    computePotentialCoefsSph(src,
        math::SphHarmIndices(lmax_tmp, mmax_tmp, src.symmetry()),
        gridRadii, Phi, dPhi);
    // now resize the coefficients back to the requested order
    for(unsigned int k=0; k<gridSizeR; k++) {
        restrictSphHarmCoefs(lmax, mmax, Phi [k]);
        restrictSphHarmCoefs(lmax, mmax, dPhi[k]);
    }
    return PtrPotential(new Multipole(gridRadii, Phi, dPhi));
}

PtrPotential Multipole::create(
    const BaseDensity& src, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{ return createMultipole(src, lmax, mmax, gridSizeR, rmin, rmax); }

PtrPotential Multipole::create(
    const BasePotential& src, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{ return createMultipole(src, lmax, mmax, gridSizeR, rmin, rmax); }

template<typename ParticleT>
PtrPotential Multipole::create(
    const particles::PointMassArray<ParticleT> &points,
    coord::SymmetryType sym, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax, double smooth)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || (rmax!=0 && rmax<=rmin))
        throw std::invalid_argument("Multipole: invalid grid parameters");
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");
    chooseGridRadii(points, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    if(isSpherical(sym))
        lmax = 0;
    if(isZRotSymmetric(sym))
        mmax = 0;
    std::vector<std::vector<double> > Phi, dPhi;
    computePotentialCoefsSph(points,
        math::SphHarmIndices(lmax, mmax, sym),
        gridRadii, Phi, dPhi, smooth);
    return PtrPotential(new Multipole(gridRadii, Phi, dPhi));
}

// list all template instantiations
template PtrPotential Multipole::create(const particles::PointMassArray<coord::PosCar>&,
    coord::SymmetryType, int, int, unsigned int, double, double, double);
template PtrPotential Multipole::create(const particles::PointMassArray<coord::PosCyl>&,
    coord::SymmetryType, int, int, unsigned int, double, double, double);
template PtrPotential Multipole::create(const particles::PointMassArray<coord::PosSph>&,
    coord::SymmetryType, int, int, unsigned int, double, double, double);
template PtrPotential Multipole::create(const particles::PointMassArray<coord::PosVelCar>&,
    coord::SymmetryType, int, int, unsigned int, double, double, double);
template PtrPotential Multipole::create(const particles::PointMassArray<coord::PosVelCyl>&,
    coord::SymmetryType, int, int, unsigned int, double, double, double);
template PtrPotential Multipole::create(const particles::PointMassArray<coord::PosVelSph>&,
    coord::SymmetryType, int, int, unsigned int, double, double, double);

// now the one and only 'proper' constructor
Multipole::Multipole(
    const std::vector<double> &_gridRadii,
    const std::vector<std::vector<double> > &Phi,
    const std::vector<std::vector<double> > &dPhi) :
    gridRadii(_gridRadii), ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = gridRadii.size();
    bool correct = gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        gridSizeR == Phi.size() && gridSizeR == dPhi.size();
    for(unsigned int k=1; correct && k<gridSizeR; k++)
        correct &= gridRadii[k] > gridRadii[k-1];
    if(!correct)
        throw std::invalid_argument("Multipole: invalid radial grid");
    if(ind.lmax>LMAX_SPHHARM)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");

    // construct the interpolating splines
    impl = ind.lmax<=2 ?   // choose between 1d or 2d splines, depending on the expected efficiency
        PtrPotential(new MultipoleInterp1d(gridRadii, Phi, dPhi)) :
        PtrPotential(new MultipoleInterp2d(gridRadii, Phi, dPhi));

    // determine asymptotic behaviour at small and large radii
    asymptInner = initAsympt(gridRadii[0], gridRadii[1], Phi[0], Phi[1], dPhi[0], true);
    asymptOuter = initAsympt(gridRadii[gridSizeR-1], gridRadii[gridSizeR-2],
        Phi[gridSizeR-1], Phi[gridSizeR-2], dPhi[gridSizeR-1], false);
}

void Multipole::getCoefs(
    std::vector<double> &radii,
    std::vector<std::vector<double> > &Phi,
    std::vector<std::vector<double> > &dPhi) const
{
    radii = gridRadii;
    radii.front() *= 1+SAFETY_FACTOR;  // avoid the possibility of getting outside the of radii where
    radii.back()  *= 1-SAFETY_FACTOR;  // the interpolating splines are defined, due to roundoff errors
    // use the fact that the spherical-harmonic transform is invertible to machine precision:
    // take the values and derivatives of potential at grid nodes and apply forward transform
    // to obtain the coefficients.
    computePotentialCoefsSph(*impl, ind, radii, Phi, dPhi);
    radii = gridRadii;  // restore the original values of radii
}

void Multipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double rsq = pow_2(pos.R) + pow_2(pos.z);
    if(rsq < pow_2(gridRadii.front()) * (1+SAFETY_FACTOR))
        asymptInner->eval(pos, potential, deriv, deriv2);
    else if(rsq > pow_2(gridRadii.back()) * (1-SAFETY_FACTOR))
        asymptOuter->eval(pos, potential, deriv, deriv2);
    else
        impl->eval(pos, potential, deriv, deriv2);
}

// ------- Implementations of multipole potential interpolators ------- //

// ------- PowerLawPotential ------- //

PowerLawMultipole::PowerLawMultipole(double _r0, bool _inner,
    const std::vector<double>& _S,
    const std::vector<double>& _U,
    const std::vector<double>& _W) :
    ind(math::getIndicesFromCoefs(_U)), r0sq(_r0*_r0), inner(_inner), S(_S), U(_U), W(_W) 
{
    // safeguard against errors in slope determination - 
    // ensure that all harmonics with l>0 do not asymptotically overtake the principal one (l=0)
    for(unsigned int c=1; c<S.size(); c++)
        if(U[c]!=0 && ((inner && S[c] < S[0]) || (!inner && S[c] > S[0])) )
            S[c] = S[0];
}
    
void PowerLawMultipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    bool needGrad = grad!=NULL || hess!=NULL;
    bool needHess = hess!=NULL;
    double   Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double  dPhi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double d2Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double rsq   = pow_2(pos.R) + pow_2(pos.z);
    double dlogr = log(rsq / r0sq) * 0.5;
    // simplified treatment in strongly asymptotic regime - retain only l==0 term
    int lmax = (inner && rsq < r0sq*1e-16) || (!inner && rsq > r0sq*1e16) ? 0 : ind.lmax;

    // define {v=l, r0=rmin} for the inner or {v=-l-1, r0=rmax} for the outer extrapolation;
    // Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}}            + W_{l,m} * (r/r0)^v   if s!=v,
    // Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}} * ln(r/r0) + W_{l,m} * (r/r0)^v   if s==v.
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            double s=S[c], u=U[c], w=W[c], v = inner ? l : -l-1;
            double rv  = v!=0 ? exp( dlogr * v ) : 1;                // (r/r0)^v
            double rs  = s!=v ? (s!=0 ? exp( dlogr * s ) : 1) : rv;  // (r/r0)^s
            double urs = u * rs * (s!=v || u==0 ? 1 : dlogr);  // if s==v, multiply by ln(r/r0)
            double wrv = w * rv;
            Phi_lm[c] = urs + wrv;
            if(needGrad)
                dPhi_lm[c] = urs*s + wrv*v + (s!=v ? 0 : u*rs);
            if(needHess)
                d2Phi_lm[c] = urs*s*s + wrv*v*v + (s!=v ? 0 : 2*s*u*rs);
        }
    if(lmax == 0) {  // fast track
        if(potential)
            *potential = Phi_lm[0];
        double rsqinv = 1/rsq, Rr2 = pos.R * rsqinv, zr2 = pos.z * rsqinv;
        if(grad) {
            grad->dR = rsq>0 ? dPhi_lm[0] * Rr2 : S[0]>1 ? 0 : INFINITY;
            grad->dz = rsq>0 ? dPhi_lm[0] * zr2 : S[0]>1 ? 0 : INFINITY; 
            grad->dphi = 0;
        }
        if(hess) {
            double d2 = d2Phi_lm[0] - 2 * dPhi_lm[0];
            hess->dR2 = d2 * pow_2(Rr2) + dPhi_lm[0] * rsqinv;
            hess->dz2 = d2 * pow_2(zr2) + dPhi_lm[0] * rsqinv;
            hess->dRdz= d2 * Rr2 * zr2;
            hess->dRdphi = hess->dzdphi = hess->dphi2 = 0;
        }
        return;
    }
    coord::GradSph gradSph;
    coord::HessSph hessSph;
    sphHarmTransformInverseDeriv(ind, pos, Phi_lm, dPhi_lm, d2Phi_lm, potential,
        needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
    if(needGrad)
        transformDerivsSphToCyl(pos, gradSph, hessSph, grad, hess);
}

// ------- Multipole potential with 1d interpolating splines for each SH harmonic ------- //

MultipoleInterp1d::MultipoleInterp1d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        gridSizeR == Phi.size() && gridSizeR == dPhi.size() &&
        Phi[0].size() == ind.size() && ind.lmax >= 0 && ind.mmax <= ind.lmax);

    // determine the characteristic radius from the condition that Phi(0) = -Mtotal/rscale
    Rscale = radii.back() * Phi.back()[0] / Phi.front()[0];
    if(!(Rscale>0))   // something weird happened, set to a reasonable default value
        Rscale = 1.;
#ifdef VERBOSE_REPORT
    std::cout << "Multipole Rscale="<<Rscale<<"\n";
#endif

    // set up a logarithmic radial grid
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> Phi_lm(gridSizeR), dPhi_lm(gridSizeR);  // temp.arrays

    // set up 1d quintic splines in radius for each non-trivial (l,m) coefficient
    spl.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            for(unsigned int k=0; k<gridSizeR; k++) {
                double amp = sqrt(pow_2(Rscale) + pow_2(radii[k]));   // additional scaling multiplier
                Phi_lm[k]  =  amp *  Phi[k][c];
                // transform d Phi / d r  to  d (Phi * amp(r)) / d ln(r)
                dPhi_lm[k] = (amp * dPhi[k][c] + Phi[k][c] * radii[k] / amp) * radii[k];
            }
            spl[c] = math::QuinticSpline(gridR, Phi_lm, dPhi_lm);
        }
}
    
void MultipoleInterp1d::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    bool needGrad = grad!=NULL || hess!=NULL;
    bool needHess = hess!=NULL;
    double   Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double  dPhi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double d2Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double r = hypot(pos.R, pos.z), logr = log(r);
    // compute spherical-harmonic coefs
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            spl[c].evalDeriv(logr, &Phi_lm[c],
                needGrad?  &dPhi_lm[c] : NULL,
                needHess? &d2Phi_lm[c] : NULL);
        }
    double pot;
    coord::GradSph gradSph;
    coord::HessSph hessSph;
    sphHarmTransformInverseDeriv(ind, pos, Phi_lm, dPhi_lm, d2Phi_lm, &pot,
        needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
    transformAmplitude(r, Rscale, pot,
        needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
    if(potential)
        *potential = pot;
    if(needGrad)
        transformDerivsSphToCyl(pos, gradSph, hessSph, grad, hess);
}

// ------- Multipole potential with 2d interpolating splines for each azimuthal harmonic ------- //

/** Set up non-uniform grid in cos(theta), with denser spacing close to z-axis.
    We want (some of) the nodes of the grid to coincide with the nodes of Gauss-Legendre
    quadrature on the interval -1 <= cos(theta) <= 1, which ensures that the values
    of 2d spline at these angles exactly equals the input values, thereby making
    the forward and reverse Legendre transformation invertible to machine precision.
    So we first compute these nodes for the given order of sph.-harm. expansion lmax,
    and then take only the non-negative half of them for the spline in cos(theta),
    plus one at theta=0.
    To achieve better accuracy in approximating the Legendre polynomials by quintic
    splines, we insert additional nodes in between the original ones.
*/
static std::vector<double> createGridInTheta(unsigned int lmax)
{
    unsigned int numPointsGL = lmax+1;
    std::vector<double> tau(numPointsGL+2), dummy(numPointsGL);
    math::prepareIntegrationTableGL(-1, 1, numPointsGL, &tau[1], &dummy.front());
    // convert GL nodes (cos theta) to tau = cos(theta)/(sin(theta)+1)
    for(unsigned int iGL=1; iGL<=numPointsGL; iGL++)
        tau[iGL] /= 1 + sqrt(1-pow_2(tau[iGL]));
    // add points at the ends of original interval (GL nodes are all interior)
    tau.back() = 1;
    tau.front()= -1;
    // split each interval between two successive GL nodes (or the ends of original interval)
    // into this many grid points (accuracy of Legendre function approximation is better than 1e-6)
    unsigned int oversampleFactor = 3;
    // number of grid points for spline in 0 <= theta) <= pi
    unsigned int gridSizeT = (numPointsGL+1) * oversampleFactor + 1;
    std::vector<double> gridT(gridSizeT);
    for(unsigned int iGL=0; iGL<=numPointsGL; iGL++)
        for(unsigned int iover=0; iover<oversampleFactor; iover++) {
            gridT[/*gridT.size() - 1 -*/ (iGL * oversampleFactor + iover)] =
                (tau[iGL] * (oversampleFactor-iover) + tau[iGL+1] * iover) / oversampleFactor;
        }
    gridT.back() = 1;
    return gridT;
}

MultipoleInterp2d::MultipoleInterp2d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        gridSizeR == Phi.size() && gridSizeR == dPhi.size() &&
        Phi[0].size() == ind.size() && ind.lmax >= 0 && ind.mmax <= ind.lmax);

    // determine the characteristic radius from the condition that Phi(0) = -Mtotal/rscale
    Rscale = radii.back() * Phi.back()[0] / Phi.front()[0];
    if(!(Rscale>0))   // something weird happened, set to a reasonable default value
        Rscale = 1.;
#ifdef VERBOSE_REPORT
    std::cout << "Multipole Rscale="<<Rscale<<"\n";
#endif

    // set up a 2D grid in ln(r) and tau = cos(theta)/(sin(theta)+1):
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> gridT = createGridInTheta(ind.lmax);
    unsigned int gridSizeT = gridT.size();

    // allocate temporary arrays for initialization of 2d splines
    math::Matrix<double> Phi_val(gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dR (gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dT (gridSizeR, gridSizeT);
    std::vector<double>  Plm(ind.lmax+1), dPlm(ind.lmax+1);

    // loop over azimuthal harmonic indices (m)
    spl.resize(2*ind.mmax+1);
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        int absm = math::abs(m);
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;
        // assign Phi_m, dPhi_m/d(ln r) & dPhi_m/d(tau) at each node of 2d grid (r_k, tau_j)
        for(unsigned int j=0; j<gridSizeT; j++) {
            math::sphHarmArray(ind.lmax, absm, gridT[j], &Plm.front(), &dPlm.front());
            for(unsigned int k=0; k<gridSizeR; k++) {            
                double val=0, dR=0, dT=0;
                for(int l=lmin; l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    val += Phi [k][c] *  Plm[l-absm];   // Phi_{l,m}(r)
                    dR  += dPhi[k][c] *  Plm[l-absm];   // d Phi / d r
                    dT  += Phi [k][c] * dPlm[l-absm];   // d Phi / d theta
                }
                double amp = sqrt(pow_2(Rscale) + pow_2(radii[k]));   // additional scaling multiplier
                Phi_val(k, j) = mul *  amp * val;
                // transform d Phi / d r      to  d (Phi * amp(r)) / d ln(r)
                Phi_dR (k, j) = mul * (amp * dR + val * radii[k] / amp) * radii[k];
                // transform d Phi / d theta  to  d (Phi * amp) / d tau
                Phi_dT (k, j) = mul *  amp * dT * -2 / (pow_2(gridT[j]) + 1);
            }
        }
        // establish 2D quintic spline for Phi_m(ln(r), tau)
        spl[m+ind.mmax] = math::QuinticSpline2d(gridR, gridT, Phi_val, Phi_dR, Phi_dT);
    }
}

void MultipoleInterp2d::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    const double
        r         = hypot(pos.R, pos.z),
        logr      = log(r),
        rplusRinv = 1. / (r + pos.R),
        tau       = pos.z * rplusRinv;

    // number of azimuthal harmonics to compute
    const int nm = ind.mmax - ind.mmin() + 1;

    // temporary array for storing coefficients: Phi, two first and three second derivs for each m
    double C_m[(2*LMAX_SPHHARM+1) * 6];

    // value, first and second derivs of scaled potential in scaled coordinates,
    // where 'r' stands for ln(r) and 'theta' - for tau
    double trPot;
    coord::GradSph trGrad;
    coord::HessSph trHess;

    // only compute those quantities that will be needed in output
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;

    // compute azimuthal harmonics
    for(int mm=0; mm<nm; mm++) {
        int m = mm + ind.mmin();
        if(ind.lmin(m) > ind.lmax)
            continue;
        spl[m+ind.mmax].evalDeriv(logr, tau, &C_m[mm],
            numQuantities>=3 ? &C_m[mm+nm  ] : NULL,
            numQuantities>=3 ? &C_m[mm+nm*2] : NULL,
            numQuantities==6 ? &C_m[mm+nm*3] : NULL,
            numQuantities==6 ? &C_m[mm+nm*4] : NULL,
            numQuantities==6 ? &C_m[mm+nm*5] : NULL);
    }

    // Fourier synthesis from azimuthal harmonics to actual quantities, still in scaled coords
    fourierTransformAzimuth(ind, pos.phi, C_m, &trPot,
        numQuantities>=3 ? &trGrad : NULL, numQuantities==6 ? &trHess : NULL);

    // scaling transformation for the amplitude of interpolated potential
    transformAmplitude(r, Rscale, trPot,
        numQuantities>=3 ? &trGrad : NULL, numQuantities==6 ? &trHess : NULL);

    if(potential)
        *potential = trPot;
    if(numQuantities==1)
        return;   // nothing else needed

    // abuse the coordinate transformation framework (Sph -> Cyl), where actually
    // our source Sph coords are not (r, theta, phi), but (ln r, tau, phi)
    const double
        rinv  = 1/r,
        r2inv = pow_2(rinv);
    coord::PosDerivT<coord::Cyl, coord::Sph> der;
    der.drdR = pos.R * r2inv;
    der.drdz = pos.z * r2inv;
    der.dthetadR = -tau * rinv;
    der.dthetadz = rinv - rplusRinv;
    if(grad)
        *grad = toGrad(trGrad, der);
    if(hess) {
        coord::PosDeriv2T<coord::Cyl, coord::Sph> der2;
        der2.d2rdR2  = pow_2(der.drdz) - pow_2(der.drdR);
        der2.d2rdRdz = -2 * der.drdR * der.drdz;
        der2.d2rdz2  = -der2.d2rdR2;
        der2.d2thetadR2  = pos.z * r2inv * rinv;
        der2.d2thetadRdz = pow_2(der.dthetadR) - pow_2(der.drdR) * r * rplusRinv;
        der2.d2thetadz2  = -der2.d2thetadR2 - der.dthetadR * rplusRinv;
        *hess = toHess(trGrad, trHess, der, der2);
    }
}

}; // namespace
