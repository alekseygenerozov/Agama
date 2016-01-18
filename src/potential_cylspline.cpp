#include "potential_cylspline.h"
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
const unsigned int MMAX_AZIMUTHAL_FOURIER = 64;
const double MIN_R = 1e-10;
    
/// max number of function evaluations in multidimensional integration
const unsigned int MAX_NUM_EVAL = 10000;

/// relative accuracy of potential computation (integration tolerance parameter)
const double EPSREL_POTENTIAL_INT = 1e-6;    

} // internal namespace

// ------- 2d interpolation + Fourier expansion of density ------- //

void computeDensityCoefs(const BaseDensity& dens,
    const unsigned int mmax, const std::vector<double> &gridR, const std::vector<double> &gridz,
    std::vector< std::vector<double> > &coefs)
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR<2 || sizez<2)
        throw std::invalid_argument("computeDensityCoefs: incorrect grid size");
    if(mmax > MMAX_AZIMUTHAL_FOURIER)
        throw std::invalid_argument("computeDensityCoefs: mmax is too large");
    if((dens.symmetry() & coord::ST_ZREFLECTION) != coord::ST_ZREFLECTION && gridz[0]==0)
        throw std::invalid_argument("computeDensityCoefs: input density is not symmetric "
            "under z-reflection, the grid in z must cover both positive and negative z");
    int mmin = isYReflSymmetric(dens) ? 0 : -1*mmax;
    math::FourierTransformForward trans(mmax, mmin<0);
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, dens.symmetry());
    unsigned int numHarmonicsComputed = indices.size();
    int numPoints = sizeR * sizez;
    coefs.resize(mmax*2+1);
    for(unsigned int i=0; i<numHarmonicsComputed; i++)
        coefs[indices[i]+mmax].resize(numPoints);
    bool badValueEncountered = false;
    std::string errorMsg;

    // the intended application of this class is for storing and interpolating the density
    // which is expensive to compute - that's why the loop below is parallelized
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numPoints; n++) {
        int indR = n % sizeR;  // index in radial grid
        int indz = n / sizeR;  // index in vertical direction
        double densVal[2*MMAX_AZIMUTHAL_FOURIER+1];
        double dens_m [2*MMAX_AZIMUTHAL_FOURIER+1];
        try{
            for(unsigned int i=0; i<trans.size(); i++)
                densVal[i] = dens.density(coord::PosCyl(gridR[indR], gridz[indz], trans.phi(i)));
            trans.transform(densVal, dens_m);
            for(unsigned int i=0; i<numHarmonicsComputed; i++) {
                int m = indices[i];
                coefs[m+mmax][indz*sizeR+indR] =
                    dens_m[mmin<0 ? m+mmax : m] / (m==0 ? 2*M_PI : M_PI);
            }
            if(!math::isFinite(dens_m[0])) {
                errorMsg = "Invalid density value "
                    "at R=" + utils::convertToString(gridR[indR]) +
                    ", z="  + utils::convertToString(gridz[indz]);
                badValueEncountered = true;
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
            badValueEncountered = true;
        }
    }
    if(badValueEncountered)
        throw std::runtime_error("Error in computeDensityCoefs: "+errorMsg);
}

namespace{  // internal structures

template<typename DensityType>
double density_rho_m(const DensityType& dens, int m, double R, double z);

template<>
inline double density_rho_m(const BaseDensity& dens, int m, double R, double z) {
    return m==0 ? dens.density(coord::PosCyl(R, z, 0)) : 0;
}

template<>
inline double density_rho_m(const DensityAzimuthalHarmonic& dens, int m, double R, double z) {
    return dens.rho_m(m, R, z);
}

template<typename DensityType>
class AzimuthalHarmonicIntegrand: public math::IFunctionNdim {
public:
    AzimuthalHarmonicIntegrand(const DensityType& _dens, int _m, double _R, double _z) :
        dens(_dens), m(_m), R(_R), z(_z) {}

    // evaluate the function at a given (R1,z1) point (scaled)
    virtual void eval(const double Rz[], double values[]) const
    {
        for(unsigned int c=0; c<numValues(); c++)
            values[c] = 0;
        if(Rz[0]>=1 || Rz[1]>=1)  // scaled coords point at 0 or infinity
            return;

        // 0. unscale input coordinates
        const double R1 = Rz[0] / (1-Rz[0]);
        const double z1 = Rz[1] / (1-Rz[1]);
        // jacobian of scaled coord transformation
        const double jac = -2*M_PI * R1 / pow_2( (1-Rz[0]) * (1-Rz[1]) );

        // 1. get the values of density at (R1,z1) and (R1,-z1):
        // here the density evaluation may be a computational bottleneck,
        // so in the typical case of z-reflection symmetry we save on using
        // the same value of density for both positive and negative z1.
        double rhoA = density_rho_m(dens, m, R1, z1);
        double rhoB = (dens.symmetry() & coord::ST_ZREFLECTION) == coord::ST_ZREFLECTION ?
            rhoA : density_rho_m(dens, m, R1,-z1);

        // 2. multiply by  \int_0^\infty dk J_m(k R) J_m(k R1) exp(-k|z-z1|)
        double tA = R*R + R1*R1 + pow_2(z-z1);
        double tB = R*R + R1*R1 + pow_2(z+z1);
        if(R > MIN_R && R1 > MIN_R) {  // normal case
            double sq = 1 / (M_PI * sqrt(R*R1));
            double uA = tA / (2*R*R1);
            double uB = tB / (2*R*R1);
            if(uA < 1+1e-12 || uB < 1+1e-12)
                return;  // close to singularity
            double dQA, dQB;
            double QA = math::legendreQ(m-0.5, uA, &dQA);
            double QB = math::legendreQ(m-0.5, uB, &dQB);
            values[0] = jac * sq * (rhoA * QA + rhoB * QB);
            /*values[1] = jac * sq * (
                rhoA * (dQA/R1 - (QA + uA*dQA)/(2*R)) + 
                rhoB * (dQB/R1 - (QB + uB*dQB)/(2*R)) );
            values[2] = jac * sq * (rhoA * dQA * (z-z1) + rhoB * dQB * (z+z1) ) / (R*R1);*/
        } else if(m==0) {  // degenerate case
            if(tA < 1e-15 || tB < 1e-15)
                return;    // close to singularity
            double sA = 1/sqrt(tA);
            double sB = 1/sqrt(tB);
            values[0] = jac * (rhoA * sA + rhoB * sB);
            //values[2] =-jac * (rhoA * sA / tA * (z-z1) + rhoB * sB / tB * (z+z1) );
        }
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 1 /*3*/; }
private:
    const DensityType& dens;
    const int m;
    const double R, z;
};

template<typename DensityType>
void computePotentialCoefsImpl(const DensityType& dens, 
    const std::vector<int>& indices, const std::vector<double>& gridR, const std::vector<double>& gridz,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhidR, std::vector< std::vector<double> > &dPhidz)
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    int numPoints = sizeR * sizez;
    unsigned int numHarmonicsComputed = indices.size();
    unsigned int mmax = (Phi.size()-1)/2;
    bool badValueEncountered = false;
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {  // combined index variable
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
        try{
            double Rzmin[2]={0.,0.}, Rzmax[2]={1.,1.}; // integration box in scaled coords
            double result[3], error[3];
            int numEval;
            for(unsigned int i=0; i<numHarmonicsComputed; i++) {
                int m = indices[i];
                AzimuthalHarmonicIntegrand<DensityType> fnc(dens, m, gridR[iR], gridz[iz]);
                math::integrateNdim(fnc, Rzmin, Rzmax, 
                    EPSREL_POTENTIAL_INT, MAX_NUM_EVAL, result, error, &numEval);
                if(!math::isFinite(result[0])) {
                    errorMsg = "Invalid potential value "
                    "at R=" + utils::convertToString(gridR[iR]) +
                    ", z="  + utils::convertToString(gridz[iz]);
                    badValueEncountered = true;
                }
                Phi   [m+mmax][ind] = result[0];
                dPhidR[m+mmax][ind] = result[1];
                dPhidz[m+mmax][ind] = result[2];
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
            badValueEncountered = true;
        }
    }
    if(badValueEncountered)
        throw std::runtime_error("Error in CylSplineExp: "+errorMsg);
}

}  // internal namespace

void computePotentialCoefs(const BaseDensity& dens, 
    unsigned int mmax, const std::vector<double>& gridR, const std::vector<double>& gridz,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhidR, std::vector< std::vector<double> > &dPhidz)
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR == 0 || sizez == 0 || mmax > CYLSPLINE_MAX_ANGULAR_HARMONIC)
        throw std::invalid_argument("computePotentialCoefs: invalid grid parameters");
    if(isZRotSymmetric(dens))
        mmax = 0;
    // the number of output coefficients - always a full set even if some of them are zeros
    unsigned int  numHarmonicsOutput = 1 + 2*mmax;
    Phi.   resize(numHarmonicsOutput);
    dPhidR.resize(numHarmonicsOutput);
    dPhidz.resize(numHarmonicsOutput);
    for(unsigned int mm=0; mm<numHarmonicsOutput; mm++) {
        Phi   [mm].assign(sizeR*sizez, 0);
        dPhidR[mm].assign(sizeR*sizez, 0);
        dPhidz[mm].assign(sizeR*sizez, 0);
    }
    // list of non-zero m-indices under the given symmetry - necessary because we don't want
    // to waste time computing coefs that are nearly zero
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, dens.symmetry());

    // for an axisymmetric potential we don't use interpolation,
    // as the Fourier expansion of density trivially has only one harmonic
    if(isZRotSymmetric(dens)) {
        computePotentialCoefsImpl(dens, indices, gridR, gridz, Phi, dPhidR, dPhidz);
        return;
    }
    // if the input density is already a Fourier expansion, use it directly
    if(dens.name() == DensityAzimuthalHarmonic::myName()) {
        computePotentialCoefsImpl(dynamic_cast<const DensityAzimuthalHarmonic&>(dens),
            indices, gridR, gridz, Phi, dPhidR, dPhidz);
        return;
    }
    // if the input density is not rotationally-symmetric, we need to create a temporary
    // DensityAzimuthalHarmonic interpolating object.
    double Rmax = gridR.back() * 10;
    double Rmin = gridR[1] * 0.1;
    double zmax = gridz.back() * 10;
    double zmin = gridz[0]==0 ? gridz[1] * 0.1 :
        gridz[sizez/2]==0 ? gridz[sizez/2+1] * 0.1 : Rmin;
    double delta=0.1;  // relative difference between grid nodes = log(x[n+1]/x[n]) 
    std::vector<double> densGridR = math::createNonuniformGrid(log(Rmax/Rmin)/delta, Rmin, Rmax, true);
    std::vector<double> densGridz = math::createNonuniformGrid(log(zmax/zmin)/delta, zmin, zmax, true);
    if((dens.symmetry() & coord::ST_ZREFLECTION) != coord::ST_ZREFLECTION)
        densGridz = math::mirrorGrid(densGridz);
    // to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle;
    // on the other hand, the number of coefficients computed remains the same as requested
    int densmmax = std::max<int>(mmax, 16);
    std::vector< std::vector<double> > densCoefs;
    computeDensityCoefs(dens, densmmax, densGridR, densGridz, densCoefs);
    DensityAzimuthalHarmonic densInterp(densGridR, densGridz, densCoefs);
    computePotentialCoefsImpl(densInterp, indices, gridR, gridz, Phi, dPhidR, dPhidz);
}
    
// -------- public classes --------- //

DensityAzimuthalHarmonic::DensityAzimuthalHarmonic(
    const std::vector<double>& gridR, const std::vector<double>& gridz_orig,
    const std::vector< std::vector<double> > &coefs)
{
    unsigned int sizeR = gridR.size(), sizez_orig = gridz_orig.size(), sizez = sizez_orig;
    if(sizeR<2 || sizez<2 || 
        coefs.size()%2 == 0 || coefs.size() > 2*MMAX_AZIMUTHAL_FOURIER+1)
        throw std::invalid_argument("DensityAzimuthalHarmonic: incorrect grid size");
    int mysym = coord::ST_AXISYMMETRIC;
    // grid in z may only cover half-space z>=0 if the density is z-reflection symmetric
    std::vector<double> gridz;
    if(gridz_orig[0] == 0) {
        gridz = math::mirrorGrid(gridz_orig);
        sizez = 2*sizez_orig-1;
    } else {  // if the original grid covered both z>0 and z<0, we assume that the symmetry is broken
        gridz = gridz_orig;
        mysym &= ~coord::ST_ZREFLECTION;
    }

    math::Matrix<double> val(sizeR, sizez);
    int mmax = (coefs.size()-1)/2;
    spl.resize(2*mmax+1);
    for(int mm=0; mm<=2*mmax; mm++) {
        if(coefs[mm].size() == 0)
            continue;
        if(coefs[mm].size() != sizeR * sizez_orig)
            throw std::invalid_argument("DensityAzimuthalHarmonic: incorrect coefs array size");
        double sum=0;
        for(unsigned int iR=0; iR<sizeR; iR++)
            for(unsigned int iz=0; iz<sizez_orig; iz++) {
                unsigned int ind = iz*sizeR+iR;
                if((mysym & coord::ST_ZREFLECTION) == coord::ST_ZREFLECTION) {
                    val(iR, sizez_orig-1-iz) = coefs[mm][ind];
                    val(iR, sizez_orig-1+iz) = coefs[mm][ind];
                } else
                    val(iR, iz) = coefs[mm][ind];
                sum += fabs(coefs[mm][ind]);
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
    if(abs(m)>mmax || spl[m+mmax].isEmpty() ||
        R<spl[m+mmax].xmin() || z<spl[m+mmax].ymin() || R>spl[m+mmax].xmax() || z>spl[m+mmax].ymax())
        return 0;
    return spl[m+mmax].value(R, z);
}

double DensityAzimuthalHarmonic::densityCyl(const coord::PosCyl &point) const
{
    int mmax = (spl.size()-1)/2;
    if( point.R<spl[mmax].xmin() || point.z<spl[mmax].ymin() ||
        point.R>spl[mmax].xmax() || point.z>spl[mmax].ymax() )
        return 0;
    double result = 0;
    double trig[2*MMAX_AZIMUTHAL_FOURIER];
    if((sym & coord::ST_ZROTATION) != coord::ST_ZROTATION) {
        bool needSine = (sym & coord::ST_YREFLECTION) != coord::ST_YREFLECTION;
        math::trigMultiAngle(point.phi, mmax, needSine, trig);
    }
    for(int m=-mmax; m<=mmax; m++)
        if(!spl[m+mmax].isEmpty()) {
            double trig_m = m==0 ? 1 : m>0 ? trig[m-1] : trig[mmax-1-m];
            result += spl[m+mmax].value(point.R, point.z) * trig_m;
        }
    return result;
}
    
void DensityAzimuthalHarmonic::getCoefs(std::vector<double> &gridR, std::vector<double>& gridz, 
    std::vector< std::vector<double> > &coefs) const
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    unsigned int mmax = (spl.size()-1)/2;
    assert(mmax>=0 && spl.size() == mmax*2+1 && !spl[mmax].isEmpty() && 
        sizeR>=2 && sizez>=2 && sizez%2==1);
    coefs.resize(2*mmax+1);
    gridR = spl[mmax].xvalues();
    if((sym & coord::ST_ZREFLECTION) == coord::ST_ZREFLECTION) {
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
                    coefs[mm][iz*sizeR+iR] = spl[mm].value(gridR[iR], gridz[iz]);
        }
}

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
            double val1 = besselInt(R, pc.R, Z-pc.z, abs(m));
            if((mysymmetry & coord::ST_TRIAXIAL)==coord::ST_TRIAXIAL)   // add symmetric contribution from -Z
                val1 = (val1 + besselInt(R, pc.R, Z+pc.z, abs(m)))/2.;
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

CylSplineExp::CylSplineExp(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const BaseDensity& srcdensity, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    DirectPotential pot_tmp(srcdensity, Ncoefs_phi);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const BasePotential& potential, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, potential, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(unsigned int Ncoefs_R, unsigned int Ncoefs_z, unsigned int Ncoefs_phi, 
    const particles::PointMassArray<coord::PosCyl>& points, coord::SymmetryType _sym, 
    double radius_min, double radius_max, double z_min, double z_max)
{
    mysymmetry=_sym;
    if(Ncoefs_phi==0)
        mysymmetry=(coord::SymmetryType)(mysymmetry | coord::ST_ZROTATION);
    DirectPotential pot_tmp(points, Ncoefs_phi, mysymmetry);
    initPot(Ncoefs_R, Ncoefs_z, Ncoefs_phi, pot_tmp, radius_min, radius_max, z_min, z_max);
}

CylSplineExp::CylSplineExp(const std::vector<double> &gridR, const std::vector<double>& gridz, 
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

double CylSplineExp::computePhi_m(double R, double z, int m, const BasePotential& potential) const
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

void CylSplineExp::initPot(unsigned int _Ncoefs_R, unsigned int _Ncoefs_z, unsigned int _Ncoefs_phi, 
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

void CylSplineExp::initSplines(const std::vector< std::vector<double> > &coefs)
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

void CylSplineExp::getCoefs(std::vector<double>& gridR,
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

void CylSplineExp::evalCyl(const coord::PosCyl &pos,
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
    /*if(pos.z==0 && (mysymmetry & ST_TRIAXIAL)==ST_TRIAXIAL) { // symmetric about z -> -z
        sGrad.dz=0;
        sHess.dzdphi=0;
        sHess.dRdz=0;
    }*/
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
