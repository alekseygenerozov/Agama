#include "potential_factory.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_ferrers.h"
#include "potential_galpot.h"
#include "potential_perfect_ellipsoid.h"
#include "potential_sphharm.h"
#include "particles_io.h"
#include "math_core.h"
#include "utils.h"
#include "utils_config.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <map>

namespace potential {

/// \name Definitions of all known potential types and parameters
//        -------------------------------------------------------
///@{

/** List of all known potential and density types 
    (borrowed from SMILE, not everything is implemented here).
    Note that this type is not a substitute for the real class hierarchy:
    it is intended only to be used in factory methods such as 
    creating an instance of potential from its name 
    (e.g., passed as a string, or loaded from an ini file).
*/
enum PotentialType {

    //  Generic values that don't correspond to a concrete class
    PT_UNKNOWN,      ///< undefined
    PT_COEFS,        ///< pre-computed coefficients of potential expansion loaded from a coefs file
    PT_NBODY,        ///< N-body snapshot that is used for initializing a potential expansion

    //  Density models without a corresponding potential
//    PT_ELLIPSOIDAL,  ///< a generalization of spherical mass profile with arbitrary axis ratios:  CDensityEllipsoidal
//    PT_MGE,          ///< Multi-Gaussian expansion:  CDensityMGE
//    PT_SERSIC,       ///< Sersic density profile:  CDensitySersic
//    PT_EXPDISK,      ///< exponential (in R) disk with a choice of vertical density profile:  CDensityExpDisk

    //  Generic potential expansions
    PT_BSE,          ///< basis-set expansion for infinite systems:  `BasisSetExp`
    PT_SPLINE,       ///< spline spherical-harmonic expansion:  `SplineExp`
    PT_CYLSPLINE,    ///< expansion in azimuthal angle with two-dimensional meridional-plane interpolating splines:  `CylSplineExp`
    PT_MULTIPOLE,    ///< axisymmetric multipole expansion from GalPot:  `Multipole`

    //  Components of Walter Dehnen's GalPot
    PT_DISK,         ///< separable disk density model:  `DiskDensity`
    PT_SPHEROID,     ///< two-power-law spheroid density model:  `SpheroidDensity`

    //  Density interpolators
    PT_DENS_SPHHARM, ///< DensitySphericalHarmonic
    PT_DENS_CYLGRID, ///< DensityCylGrid

    //  Potentials with possibly infinite mass that can't be used as source density for a potential expansion
    PT_COMPOSITE,    ///< a superposition of multiple potential instances:  `CompositeCyl`
    PT_LOG,          ///< triaxial logaritmic potential:  `Logarithmic`
    PT_HARMONIC,     ///< triaxial simple harmonic oscillator:  `Harmonic`
//    PT_SCALEFREE,    ///< triaxial single power-law density profile:  CPotentialScaleFree
//    PT_SCALEFREESH,  ///< spherical-harmonic approximation to a triaxial power-law density:  CPotentialScaleFreeSH
    PT_NFW,          ///< spherical Navarro-Frenk-White profile:  `NFW`

    //  Analytic finite-mass potential models that can also be used as source density for a potential expansion
//    PT_SPHERICAL,    ///< arbitrary spherical mass model:  CPotentialSpherical
    PT_MIYAMOTONAGAI,///< axisymmetric Miyamoto-Nagai(1975) model:  `MiyamotoNagai`
    PT_DEHNEN,       ///< spherical, axisymmetric or triaxial Dehnen(1993) density model:  `Dehnen`
    PT_FERRERS,      ///< triaxial Ferrers model with finite extent:  `Ferrers`
    PT_PLUMMER,      ///< spherical Plummer model:  `Plummer`
//    PT_ISOCHRONE,    ///< spherical isochrone model:  `Isochrone`
    PT_PERFECTELLIPSOID,  ///< oblate axisymmetric Perfect Ellipsoid of Kuzmin/de Zeeuw :  `OblatePerfectEllipsoid`
};

/// structure that contains parameters for all possible potentials
struct ConfigPotential
{
    PotentialType potentialType;   ///< type of the potential
    PotentialType densityType;     ///< specifies the density model used for initializing a potential expansion
    SymmetryType symmetryType;     ///< degree of symmetry (mainly used to explicitly disregard certain terms in a potential expansion)
    double mass;                   ///< total mass of the model (not applicable to all potential types)
    double scaleRadius;            ///< scale radius of the model (if applicable)
    double scaleRadius2;           ///< second scale radius of the model (if applicable)
    double q, p;                   ///< axis ratio of the model (if applicable)
    double gamma;                  ///< central cusp slope (for Dehnen and scale-free models)
    double sersicIndex;            ///< Sersic index (for Sersic density model)
    unsigned int numCoefsRadial;   ///< number of radial terms in BasisSetExp or grid points in spline potentials
    unsigned int numCoefsAngular;  ///< number of angular terms in spherical-harmonic expansion
    unsigned int numCoefsVertical; ///< number of coefficients in z-direction for CylSplineExp potential
    double alpha;                  ///< shape parameter for BasisSetExp potential
    double splineSmoothFactor;     ///< amount of smoothing in SplineExp initialized from an N-body snapshot
    double splineRMin, splineRMax; ///< if nonzero, specifies the inner- and outermost grid node radii for SplineExp and CylSplineExp
    double splineZMin, splineZMax; ///< if nonzero, gives the grid extent in z direction for CylSplineExp
    std::string fileName;          ///< name of file with coordinates of points, or coefficients of expansion, or any other external data array
    /// default constructor initializes the fields to some reasonable values
    ConfigPotential() :
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(ST_DEFAULT),
        mass(1.), scaleRadius(1.), scaleRadius2(1.), q(1.), p(1.), gamma(1.), sersicIndex(4.),
        numCoefsRadial(20), numCoefsAngular(6), numCoefsVertical(20),
        alpha(0.), splineSmoothFactor(1.), splineRMin(0), splineRMax(0), splineZMin(0), splineZMax(0)
        {};
};

///@}
/// \name Correspondence between enum potential and symmetry types and string names
//        -------------------------------------------------------------------------
///@{

/// lists all 'true' potentials, i.e. those providing a complete density-potential(-force) pair
typedef std::map<PotentialType, const char*> PotentialNameMapType;

/// lists all analytic density profiles 
/// (including those that don't have corresponding potential, but excluding general-purpose expansions)
typedef std::map<PotentialType, const char*> DensityNameMapType;

/// lists available symmetry types
typedef std::map<SymmetryType,  const char*> SymmetryNameMapType;

static PotentialNameMapType PotentialNames;
static DensityNameMapType DensityNames;
static SymmetryNameMapType SymmetryNames;
static bool mapinitialized = false;

/// create a correspondence between names and enum identifiers for potential, density and symmetry types
static void initPotentialAndSymmetryNameMap()
{
    PotentialNames.clear();
    PotentialNames[PT_LOG]       = Logarithmic::myName();
    PotentialNames[PT_HARMONIC]  = Harmonic::myName();
    PotentialNames[PT_NFW]       = NFW::myName();
    PotentialNames[PT_PLUMMER]   = Plummer::myName();
    PotentialNames[PT_MIYAMOTONAGAI] = MiyamotoNagai::myName();
    PotentialNames[PT_DEHNEN]    = Dehnen::myName();
    PotentialNames[PT_FERRERS]   = Ferrers::myName();
    PotentialNames[PT_PERFECTELLIPSOID] = OblatePerfectEllipsoid::myName();
    PotentialNames[PT_BSE]       = BasisSetExp::myName();
    PotentialNames[PT_SPLINE]    = SplineExp::myName();
    PotentialNames[PT_CYLSPLINE] = CylSplineExp::myName();
    PotentialNames[PT_MULTIPOLE] = Multipole::myName();
    PotentialNames[PT_DISK]      = DiskDensity::myName();
//    PotentialNames[PT_SCALEFREE] = CPotentialScaleFree::myName();
//    PotentialNames[PT_SCALEFREESH] = CPotentialScaleFreeSH::myName();
//    PotentialNames[PT_SPHERICAL] = CPotentialSpherical::myName();

    // list of density models available for BSE and Spline approximation
    DensityNames.clear();
//    DensityNames[PT_ELLIPSOIDAL] = CDensityEllipsoidal::myName();
//    DensityNames[PT_MGE] = CDensityMGE::myName();
    DensityNames[PT_COEFS]   = "Coefs";  // denotes that potential expansion coefs are loaded from a text file rather than computed from a density model
    DensityNames[PT_NBODY]   = "Nbody"; // denotes a density model from discrete points in Nbody file
    DensityNames[PT_PLUMMER] = Plummer::myName();
    DensityNames[PT_MIYAMOTONAGAI] = MiyamotoNagai::myName();
    DensityNames[PT_DEHNEN]  = Dehnen::myName();
    DensityNames[PT_FERRERS] = Ferrers::myName();
    DensityNames[PT_PERFECTELLIPSOID] = OblatePerfectEllipsoid::myName();
    DensityNames[PT_DENS_CYLGRID] = DensityCylGrid::myName();
    DensityNames[PT_DENS_SPHHARM] = DensitySphericalHarmonic::myName();
//    DensityNames[PT_ISOCHRONE] = CDensityIsochrone::myName();
//    DensityNames[PT_EXPDISK] = CDensityExpDisk::myName();
//    DensityNames[PT_SERSIC] = CDensitySersic::myName();

    SymmetryNames[ST_NONE]         = "None";
    SymmetryNames[ST_REFLECTION]   = "Reflection";
    SymmetryNames[ST_TRIAXIAL]     = "Triaxial";
    SymmetryNames[ST_AXISYMMETRIC] = "Axisymmetric";
    SymmetryNames[ST_SPHERICAL]    = "Spherical";

    mapinitialized=true;
}

/// return the name of the potential of a given type, or empty string if unavailable
static const char* getPotentialNameByType(PotentialType type)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    PotentialNameMapType::const_iterator iter=PotentialNames.find(type);
    if(iter!=PotentialNames.end()) 
        return iter->second;
    return "";
}

/// return the type of the potential model by its name, or PT_UNKNOWN if unavailable
static PotentialType getPotentialTypeByName(const std::string& PotentialName)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    for(PotentialNameMapType::const_iterator iter=PotentialNames.begin(); 
        iter!=PotentialNames.end(); 
        ++iter)
        if(utils::stringsEqual(PotentialName, iter->second)) 
            return iter->first;
    return PT_UNKNOWN;
}

/// return the type of the density model by its name, or PT_UNKNOWN if unavailable
static PotentialType getDensityTypeByName(const std::string& DensityName)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    for(DensityNameMapType::const_iterator iter=DensityNames.begin(); 
        iter!=DensityNames.end(); 
        ++iter)
        if(utils::stringsEqual(DensityName, iter->second)) 
            return iter->first;
    return PT_UNKNOWN;
}

/// return the type of symmetry by its name, or ST_DEFAULT if unavailable
static SymmetryType getSymmetryTypeByName(const std::string& SymmetryName)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    if(SymmetryName.empty()) 
        return ST_DEFAULT;
    // compare only the first letter (should abandon this simplification 
    // if more than one symmetry types are defined that could start with the same letter)
    for(SymmetryNameMapType::const_iterator iter=SymmetryNames.begin(); 
        iter!=SymmetryNames.end(); 
        ++iter)
        if(tolower(SymmetryName[0]) == tolower(iter->second[0])) 
            return iter->first;
    return ST_DEFAULT;
}

/// return file extension for writing the coefficients of potential of the given type,
/// or empty string if the potential type is not one of the expansion types
static const char* getCoefFileExtension(PotentialType pottype)
{
    switch(pottype) {
        case PT_BSE:        return ".coef_bse";
        case PT_SPLINE:     return ".coef_spl";
        case PT_CYLSPLINE:  return ".coef_cyl";
        case PT_MULTIPOLE:  return ".coef_mul";
        default: return "";
    }
}

/// return file extension for writing the coefficients of expansion of the given potential
const char* getCoefFileExtension(const std::string& potName) {
    return getCoefFileExtension(getPotentialTypeByName(potName)); }

///@}
/// \name Conversion between string key/value maps and structured potential parameters
//        ----------------------------------------------------------------------------
///@{

/** Parse the potential parameters contained in a text array of "key=value" pairs.
    \param[in] params  is the array of string pairs "key" and "value", for instance,
    created from command-line arguments, or read from an INI file;
    \param[in] converter  is the instance of unit converter for translating the dimensional
    parameters (such as mass or scale radius) into internal units (may be a trivial converter);
    \return    the structure containing all parameters of potential
*/
static ConfigPotential parseParams(const utils::KeyValueMap& params, const units::ExternalUnits& conv)
{
    ConfigPotential config;
    config.potentialType = getPotentialTypeByName(params.getString("Type"));
    config.densityType   = getDensityTypeByName  (params.getString("Density"));
    config.symmetryType  = getSymmetryTypeByName (params.getString("Symmetry"));
    config.fileName    = params.getString("File");
    config.mass        = params.getDouble("Mass", config.mass) * conv.massUnit;
    config.q           = params.getDoubleAlt("axisRatioY", "q", config.q);
    config.p           = params.getDoubleAlt("axisRatioZ", "p", config.p);
    config.scaleRadius = params.getDoubleAlt("scaleRadius", "rscale", config.scaleRadius) * conv.lengthUnit;
    config.scaleRadius2= params.getDoubleAlt("scaleRadius2","scaleHeight",config.scaleRadius2) * conv.lengthUnit;
    config.gamma       = params.getDouble   ("Gamma", config.gamma);
    config.sersicIndex = params.getDouble   ("SersicIndex", config.sersicIndex);
    config.numCoefsRadial  = params.getInt("NumCoefsRadial",  config.numCoefsRadial);
    config.numCoefsVertical= params.getInt("NumCoefsVertical",config.numCoefsVertical);
    config.numCoefsAngular = params.getInt("NumCoefsAngular", config.numCoefsAngular);
    config.alpha              = params.getDouble("Alpha", config.alpha);
    config.splineSmoothFactor = params.getDouble("splineSmoothFactor", config.splineSmoothFactor);
    config.splineRMin = params.getDouble("splineRMin", config.splineRMin) * conv.lengthUnit;
    config.splineRMax = params.getDouble("splineRMax", config.splineRMax) * conv.lengthUnit;
    config.splineZMin = params.getDouble("splineZMin", config.splineZMin) * conv.lengthUnit;
    config.splineZMax = params.getDouble("splineZMax", config.splineZMax) * conv.lengthUnit;
    return config;
}

static DiskParam parseDiskParams(const utils::KeyValueMap& params, const units::ExternalUnits& conv)
{
    DiskParam config;
    config.surfaceDensity      = params.getDouble("surfaceDensity") * conv.massUnit / pow_2(conv.lengthUnit);
    config.scaleRadius         = params.getDouble("scaleRadius") * conv.lengthUnit;
    config.scaleHeight         = params.getDouble("scaleHeight") * conv.lengthUnit;
    config.innerCutoffRadius   = params.getDouble("innerCutoffRadius") * conv.lengthUnit;
    config.modulationAmplitude = params.getDouble("modulationAmplitude");
    return config;
};

static SphrParam parseSphrParams(const utils::KeyValueMap& params, const units::ExternalUnits& conv)
{
    SphrParam config;
    config.densityNorm        = params.getDouble("densityNorm") * conv.massUnit / pow_3(conv.lengthUnit);
    config.axisRatio          = params.getDoubleAlt("axisRatio", "axisRatioZ", 1.0);
    config.gamma              = params.getDouble("gamma");
    config.beta               = params.getDouble("beta");
    config.scaleRadius        = params.getDouble("scaleRadius") * conv.lengthUnit;
    config.outerCutoffRadius  = params.getDouble("outerCutoffRadius") * conv.lengthUnit;
    return config;
}

///@}
/// \name Factory routines for constructing various Potential classes from data stored in a stream
//        ----------------------------------------------------------------------------------------
///@{

/// attempt to load coefficients of BasisSetExp or SplineExp stored in a text file
static PtrPotential readPotentialSphHarmExp(
    std::istream& strm, const PotentialType potentialType)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer);
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsRadial = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsAngular = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    double param = utils::convertToDouble(fields[0]);   // meaning of this parameter depends on potential type
    if( (potentialType == PT_BSE && param<0.5) || 
        (potentialType == PT_SPLINE && ncoefsRadial<4) ) 
        ok = false;
    std::vector< std::vector<double> > coefs;
    std::vector< double > radii;
    while(ok && std::getline(strm, buffer))  // time, ignored
    {
        std::getline(strm, buffer);  // comments, ignored
        radii.clear();
        coefs.clear();
        for(unsigned int n=0; ok && n<=ncoefsRadial; n++)
        {
            std::getline(strm, buffer);
            utils::splitString(buffer, "# \t", fields);
            radii.push_back(utils::convertToDouble(fields[0]));
            // for BSE this field is basis function index, for spline the radii should be in increasing order
            if( (potentialType == PT_BSE && radii.back()!=n) || 
                (potentialType == PT_SPLINE && n>0 && radii.back()<=radii[n-1]) ) 
                ok = false;
            coefs.push_back( std::vector<double>() );
            for(int l=0; l<=static_cast<int>(ncoefsAngular); l++)
                for(int m=-l; m<=l; m++)
                {
                    unsigned int fi=1+l*(l+1)+m;
                    coefs.back().push_back( fi<fields.size() ? utils::convertToDouble(fields[fi]) : 0);
                }
        }
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") +
            getPotentialNameByType(potentialType));
    switch(potentialType)
    {
    case PT_BSE: 
        return PtrPotential(new BasisSetExp(/*Alpha*/param, coefs)); 
    case PT_SPLINE:
        return PtrPotential(new SplineExp(radii, coefs)); 
    default:
        throw std::invalid_argument(std::string("Unknown potential type to load: ") +
            getPotentialNameByType(potentialType));
    }
}

static PtrPotential readPotentialMultipole(std::istream& strm)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer);
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsRadial = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsAngular = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();  // ignored
    ok &= std::getline(strm, buffer).good();  // ignored
    ok &= std::getline(strm, buffer).good();  // header, ignored
    std::vector< double > radii;
    std::vector< std::vector<double> > Phi, dPhi;
    for(unsigned int n=0; ok && n<ncoefsRadial; n++) {
        std::getline(strm, buffer);
        utils::splitString(buffer, "# \t", fields);
        radii.push_back(utils::convertToDouble(fields[0]));
        if((n>0 && radii.back() <= radii[n-1]) || fields.size() != ncoefsAngular/2+2) 
            ok = false;
        Phi.push_back( std::vector<double>() );
        for(unsigned int l=1; ok && l<fields.size(); l++)
            Phi.back().push_back(utils::convertToDouble(fields[l]));
    }
    ok &= std::getline(strm, buffer).good();  // header, ignored
    for(unsigned int n=0; ok && n<ncoefsRadial; n++) {
        std::getline(strm, buffer);
        utils::splitString(buffer, "# \t", fields);
        if((n>0 && radii.back() <= radii[n-1]) || fields.size() != ncoefsAngular/2+2) 
            ok = false;
        dPhi.push_back( std::vector<double>() );
        for(unsigned int l=1; ok && l<fields.size(); l++)
            dPhi.back().push_back(utils::convertToDouble(fields[l]));
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") + Multipole::myName());
    return PtrPotential(new Multipole(radii, Phi, dPhi)); 
}

/// attempt to load coefficients of CylSplineExp stored in a text file
static PtrPotential readPotentialCylSpline(std::istream& strm)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer);
    utils::splitString(buffer, "# \t", fields);
    size_t size_R = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    size_t ncoefsAngular = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    size_t size_z = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();  // time, ignored
    ok &= size_R>0 && size_z>0;
    std::vector<double> gridR, gridz;
    std::vector<std::vector<double> > coefs(2*ncoefsAngular+1);
    while(ok && std::getline(strm, buffer) && !strm.eof()) {
        utils::splitString(buffer, "# \t", fields);
        int m = utils::convertToInt(fields[0]);  // m (azimuthal harmonic index)
        if(m < -static_cast<int>(ncoefsAngular) || m > static_cast<int>(ncoefsAngular))
            ok=false;
        std::getline(strm, buffer);  // radii
        if(gridR.size()==0) {  // read values of R only once
            utils::splitString(buffer, "# \t", fields);
            for(size_t i=1; i<fields.size(); i++)
                gridR.push_back(utils::convertToDouble(fields[i]));
            if(gridR.size() != size_R)
                ok=false;
        }
        gridz.clear();
        coefs[m+ncoefsAngular].assign(size_R*size_z,0);
        for(size_t iz=0; ok && iz<size_z; iz++) {
            std::getline(strm, buffer);
            utils::splitString(buffer, "# \t", fields);
            gridz.push_back(utils::convertToDouble(fields[0]));
            if(iz>0 && gridz.back()<=gridz[iz-1]) 
                ok=false;  // the values of z should be in increasing order
            for(size_t iR=0; iR<size_R; iR++) {
                double val=0;
                if(iR+1<fields.size())
                    val = utils::convertToDouble(fields[iR+1]);
                else
                    ok=false;
                coefs[m+ncoefsAngular][iz*size_R+iR]=val;
            }
        }
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") + CylSplineExp::myName());
    return PtrPotential(new CylSplineExp(gridR, gridz, coefs));
}

// Main routine: load potential expansion coefficients from a text file
PtrPotential readPotential(const std::string& fileName)
{
    if(fileName.empty()) {
        throw std::runtime_error("readPotentialCoefs: empty file name");
    }
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) {
        throw std::runtime_error("readPotentialCoefs: cannot read from file "+fileName);
    }
    // check header
    std::string buffer;
    bool ok = std::getline(strm, buffer);
    if(ok && buffer.size()<256) {  // to avoid parsing a binary file as a text
        std::vector<std::string> fields;
        utils::splitString(buffer, "# \t", fields);
        if(fields[0] == "BSEcoefs") {
            return readPotentialSphHarmExp(strm, PT_BSE);
        }
        if(fields[0] == "SHEcoefs") {
            return readPotentialSphHarmExp(strm, PT_SPLINE);
        }
        if(fields[0] == "CylSpline") {
            return readPotentialCylSpline(strm);
        }
        if(fields[0] == "Multipole") {
            return readPotentialMultipole(strm);
        }
    }
    throw std::runtime_error("readPotentialCoefs: cannot find "
        "valid potential coefficients in file "+fileName);
}

///@}
/// \name Routines for storing various Density and Potential classes into a stream
//        ------------------------------------------------------------------------
///@{

static void writePotentialSphHarmExp(std::ostream& strm,
    const BasePotentialSphericalHarmonic& potential)
{
    std::vector<double> indices;
    std::vector< std::vector<double> > coefs;
    size_t ncoefsAngular=0;
    switch(getPotentialTypeByName(potential.name()))
    {
    case PT_BSE: {
        const BasisSetExp& potBSE = dynamic_cast<const BasisSetExp&>(potential);
        indices.resize(potBSE.getNumCoefsRadial()+1);
        for(size_t i=0; i<indices.size(); i++) indices[i]=i*1.0;
        potBSE.getCoefs(coefs);
        assert(coefs.size() == indices.size());
        ncoefsAngular = potBSE.getNumCoefsAngular();
        strm << "BSEcoefs\t#header\n" << 
            potBSE.getNumCoefsRadial() << "\t#n_radial\n" << 
            ncoefsAngular << "\t#n_angular\n" << 
            potBSE.getAlpha() <<"\t#alpha\n0\t#time\n";
        strm << "#index";
        break; 
    }
    case PT_SPLINE: {
        const SplineExp& potSpline = dynamic_cast<const SplineExp&>(potential);
        potSpline.getCoefs(indices, coefs);
        assert(coefs.size() == indices.size());
        assert(indices[0] == 0);  // leftmost radius is 0
        coefs[0].resize(1);       // retain only l=0 term for r=0, the rest is supposed to be zero
        ncoefsAngular = potSpline.getNumCoefsAngular();
        strm << "SHEcoefs\t#header\n" << 
            potSpline.getNumCoefsRadial() << "\t#n_radial\n" << 
            ncoefsAngular << "\t#n_angular\n" <<
            0 <<"\t#unused\n0\t#time\n";
        strm << "#radius";
        break; 
    }
    default:  // shouldn't occur, the potential type was checked in the caller
        throw std::invalid_argument("Unknown type of potential to write");
    }
    for(int l=0; l<=static_cast<int>(ncoefsAngular); l++)
        for(int m=-l; m<=l; m++)
            strm << "\tl="<<l<<",m="<<m;  // header line
    strm << "\n";
    for(size_t n=0; n<indices.size(); n++)
    {
        strm << indices[n];
        // leading coeft should be high-accuracy at least for spline potential
        strm << "\t" << std::setprecision(16) << coefs[n][0] << std::setprecision(8);
        for(size_t i=1; i<coefs[n].size(); i++)
            strm << "\t" << coefs[n][i];
        strm << "\n";
    }
}

static void writePotentialMultipole(std::ostream& strm,
    const Multipole& potential)
{
    std::vector<double> radii;
    std::vector< std::vector<double> > Phi, dPhi;
    potential.getCoefs(radii, Phi, dPhi);
    unsigned int ncoefs=Phi[0].size();
    strm << "Multipole\t#header\n" << 
        radii.size() << "\t#n_radial\n" << 
        (2*ncoefs-2) << "\t#n_angular\n" <<
        0 <<"\t#unused\n0\t#time\n";
    strm << "#radius\\Phi_l";
    for(unsigned int l=0; l<ncoefs; l++)
        strm << "\t" << (l*2);
    strm << "\n";  // header line
    for(unsigned int n=0; n<radii.size(); n++) {
        strm << radii[n];
        // leading coeft should be high-accuracy
        strm << "\t" << std::setprecision(16) << Phi[n][0] << std::setprecision(8);
        for(unsigned int i=1; i<Phi[n].size(); i++)
            strm << "\t" << Phi[n][i];
        strm << "\n";
    }
    strm << "#radius\\dPhi_l\n";
    for(unsigned int n=0; n<radii.size(); n++) {
        strm << radii[n];
        strm << "\t" << std::setprecision(16) << dPhi[n][0] << std::setprecision(8);
        for(unsigned int i=1; i<dPhi[n].size(); i++)
            strm << "\t" << dPhi[n][i];
        strm << "\n";
    }
}

static void writePotentialCylSpline(std::ostream& strm, const CylSplineExp& potential)
{
    std::vector<double> gridR, gridz;
    std::vector<std::vector<double> > coefs;
    potential.getCoefs(gridR, gridz, coefs);
    int mmax = coefs.size()/2;
    strm << "CylSpline\t#header\n" << gridR.size() << "\t#size_R\n" << mmax << "\t#m_max\n" <<
        gridz.size() << "\t#size_z\n0\t#time\n" << std::setprecision(16);
    for(int m=0; m<static_cast<int>(coefs.size()); m++) 
        if(coefs[m].size()>0) {
            strm << (m-mmax) << "\t#m\n#z\\R";
            for(size_t iR=0; iR<gridR.size(); iR++)
                strm << "\t" << gridR[iR];
            strm << "\n";
            for(size_t iz=0; iz<gridz.size(); iz++) {
                strm << gridz[iz];
                for(size_t iR=0; iR<gridR.size(); iR++)
                    strm << "\t" << coefs[m][iz*gridR.size()+iR];
                strm << "\n";
            }
        }
}

static void writeDensitySphericalHarmonic(std::ostream& strm, const DensitySphericalHarmonic& density)
{
    std::vector<double> radii;
    std::vector<std::vector<double> > coefs;
    density.getCoefs(radii, coefs);
    const double lmax = coefs.size()-1;

    // use values of polar angle (theta) that correspond to Gauss-Legendre nodes in cos(theta)
    // for the given order of spherical-harmonic expansion lmax
    std::vector<double> theta(lmax+2), weights(lmax+1);
    math::prepareIntegrationTableGL(-1, 1, lmax+1, &theta.front(), &weights.front());
    for(int i=lmax/2; i<=lmax; i++)  // use only upper half of points
        theta[i] = acos(theta[i]);
    strm << "#r\\theta:";
    for(int t=lmax/2; t<=lmax; t++)
        strm << '\t' << theta[t];
    strm << '\n';
    for(unsigned int i=0; i<radii.size(); i++) {
        strm << radii[i];
        for(int t=lmax/2; t<=lmax; t++)
            strm << '\t' << density.density(coord::PosSph(radii[i], theta[t], 0));
        strm << '\n';
    }
}

static void writeDensityCylGrid(std::ostream& strm, const DensityCylGrid& density)
{
    std::vector<double> gridR, gridz;
    math::Matrix<double> densVal;
    density.getCoefs(gridR, gridz, densVal);
    strm << "#R\\z";
    for(unsigned int iz=0; iz<gridz.size(); iz++)
        strm << '\t' << gridz[iz];
    strm << '\n';
    for(unsigned int iR=0; iR<gridR.size(); iR++) {
        strm << gridR[iR];
        for(unsigned int iz=0; iz<gridz.size(); iz++)
            strm << '\t' << densVal(iR, iz);
        strm << '\n';
    }
}


bool writeDensity(const std::string& fileName, const BaseDensity& dens)
{
    if(fileName.empty())
        return false;
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm)
        return false;
    PotentialType type = getPotentialTypeByName(dens.name());
    if(type == PT_UNKNOWN)
        type = getDensityTypeByName(dens.name());
    switch(type) {
    case PT_BSE:
    case PT_SPLINE:
        writePotentialSphHarmExp(strm, dynamic_cast<const BasePotentialSphericalHarmonic&>(dens));
        break;
    case PT_CYLSPLINE:
        writePotentialCylSpline(strm, dynamic_cast<const CylSplineExp&>(dens));
        break;
    case PT_MULTIPOLE:
        writePotentialMultipole(strm, dynamic_cast<const Multipole&>(dens));
        break;
    case PT_DENS_CYLGRID:
        writeDensityCylGrid(strm, dynamic_cast<const DensityCylGrid&>(dens));
        break;
    case PT_DENS_SPHHARM:
        writeDensitySphericalHarmonic(strm, dynamic_cast<const DensitySphericalHarmonic&>(dens));
        break;
    default:
        strm << "Unsupported type: " << dens.name() << "\n";
        return false;
    }
    return strm.good();
}

///@}
/// \name Legacy interface for loading GalPot parameters from a text file (deprecated)
//        ----------------------------------------------------------------------------
///@{

static void swallowRestofLine(std::ifstream& from) {
    char c;
    do {
        from.get(c);
    } while( from.good() && c !='\n');
}

PtrPotential readGalaxyPotential(const std::string& filename, const units::ExternalUnits& conv) 
{
    std::ifstream strm(filename.c_str());
    if(!strm) 
        throw std::runtime_error("Cannot open file "+std::string(filename));
    std::vector<DiskParam> diskpars;
    std::vector<SphrParam> sphrpars;
    bool ok=true;
    int num;
    strm>>num;
    swallowRestofLine(strm);
    if(num<0 || num>10 || !strm) ok=false;
    for(int i=0; i<num && ok; i++) {
        DiskParam dp;
        strm>>dp.surfaceDensity >> dp.scaleRadius >> dp.scaleHeight >> dp.innerCutoffRadius >> dp.modulationAmplitude;
        swallowRestofLine(strm);
        dp.surfaceDensity *= conv.massUnit/pow_2(conv.lengthUnit);
        dp.scaleRadius *= conv.lengthUnit;
        dp.scaleHeight *= conv.lengthUnit;
        dp.innerCutoffRadius *= conv.lengthUnit;
        if(strm) diskpars.push_back(dp);
        else ok=false;
    }
    strm>>num;
    swallowRestofLine(strm);
    ok=ok && strm;
    for(int i=0; i<num && ok; i++) {
        SphrParam sp;
        strm>>sp.densityNorm >> sp.axisRatio >> sp.gamma >> sp.beta >> sp.scaleRadius >> sp.outerCutoffRadius;
        swallowRestofLine(strm);
        sp.densityNorm *= conv.massUnit/pow_3(conv.lengthUnit);
        sp.scaleRadius *= conv.lengthUnit;
        sp.outerCutoffRadius *= conv.lengthUnit;
        if(strm) sphrpars.push_back(sp);
        else ok=false;
    }
    return createGalaxyPotential(diskpars, sphrpars);
}

///@}
/// \name Factory routines for creating instances of Density and Potential classes
//        ------------------------------------------------------------------------
///@{

/// create potential expansion of a given type from a set of point masses
template<typename ParticleT>
PtrPotential createPotentialFromPoints(const ConfigPotential& config,
    const particles::PointMassArray<ParticleT>& points)
{
    switch(config.potentialType) {
    case PT_SPLINE:
        return PtrPotential(new SplineExp(
            config.numCoefsRadial, config.numCoefsAngular, 
            points, config.symmetryType, config.splineSmoothFactor, 
            config.splineRMin, config.splineRMax));
    case PT_CYLSPLINE:
        return PtrPotential(new CylSplineExp(
            config.numCoefsRadial, config.numCoefsVertical,
            config.numCoefsAngular, points, config.symmetryType, 
            config.splineRMin, config.splineRMax,
            config.splineZMin, config.splineZMax));
    case PT_BSE:
        return PtrPotential(new BasisSetExp(
            config.alpha, config.numCoefsRadial, 
            config.numCoefsAngular, points, config.symmetryType));
    default:
        throw std::invalid_argument(std::string("Unknown potential type in createPotentialFromPoints: ")
            + getPotentialNameByType(config.potentialType));
    }
}

template<typename ParticleT>
PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<ParticleT>& points)
{
    return createPotentialFromPoints(parseParams(params, converter), points);
}
// instantiations
template PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<coord::PosCar>& points);
template PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<coord::PosVelCar>& points);
template PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<coord::PosCyl>& points);
template PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<coord::PosVelCyl>& points);
template PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<coord::PosSph>& points);
template PtrPotential createPotentialFromPoints(const utils::KeyValueMap& params,
    const units::ExternalUnits& converter, const particles::PointMassArray<coord::PosVelSph>& points);

/** Create a density model according to the parameters. 
    This only deals with finite-mass models, including some of the Potential descendants.
    This function is used within `createPotential()` to construct 
    temporary density models for initializing a potential expansion.
    \param[in] config  contains the parameters (density type, mass, shape, etc.)
    \return    the instance of a class derived from BaseDensity
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular density model
*/
static PtrDensity createAnalyticDensity(const ConfigPotential& config)
{
    switch(config.densityType) 
    {
    case PT_DEHNEN: 
        return PtrDensity(new Dehnen(
            config.mass, config.scaleRadius, config.q, config.p, config.gamma));
    case PT_PLUMMER:
        if(config.q==1 && config.p==1)
            return PtrDensity(new Plummer(config.mass, config.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Plummer is not supported");
    case PT_PERFECTELLIPSOID:
        if(config.q==1 && config.p<1)
            return PtrDensity(new OblatePerfectEllipsoid(
                config.mass, config.scaleRadius, config.scaleRadius*config.p));
        else
            throw std::invalid_argument("May only create oblate axisymmetric Perfect Ellipsoid model");
    case PT_FERRERS:
        return PtrDensity(new Ferrers(config.mass, config.scaleRadius, config.q, config.p));
    case PT_MIYAMOTONAGAI:
        return PtrDensity(new MiyamotoNagai(config.mass, config.scaleRadius, config.scaleRadius2));
    default:
        throw std::invalid_argument("Unknown density type");
    }
}

/** Create an instance of analytic potential model according to the parameters passed. 
    \param[in] config  specifies the potential parameters
    \return    the instance of potential
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular potential model
*/
static PtrPotential createAnalyticPotential(const ConfigPotential& config)
{
    switch(config.potentialType)
    {
    case PT_LOG:  // NB: it's not really 'mass' here but 'sigma'
        return PtrPotential(new Logarithmic(config.mass, config.scaleRadius, config.q, config.p));
    case PT_HARMONIC:  // NB: it's not really 'mass' here but 'Omega'
        return PtrPotential(new Harmonic(config.mass, config.q, config.p));
    case PT_MIYAMOTONAGAI:
        return PtrPotential(new MiyamotoNagai(config.mass, config.scaleRadius, config.scaleRadius2));
    case PT_DEHNEN:
        return PtrPotential(new Dehnen(
            config.mass, config.scaleRadius, config.q, config.p, config.gamma));
    case PT_FERRERS:
        return PtrPotential(new Ferrers(config.mass, config.scaleRadius, config.q, config.p)); 
    case PT_PLUMMER:
        if(config.q==1 && config.p==1)
            return PtrPotential(new Plummer(config.mass, config.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Plummer is not supported");
    case PT_NFW:
        if(config.q==1 && config.p==1)
            return PtrPotential(new NFW(config.mass, config.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Navarro-Frenk-White is not supported");
    case PT_PERFECTELLIPSOID:
        if(config.q==1 && config.p<1)
            return PtrPotential(new OblatePerfectEllipsoid(
                config.mass, config.scaleRadius, config.scaleRadius*config.p)); 
        else
            throw std::invalid_argument("May only create oblate axisymmetric Perfect Ellipsoid model");
    default:
        throw std::invalid_argument("Unknown potential type");
    }
}

/** Create an instance of potential expansion class according to the parameters passed in config */
static PtrPotential createPotentialExpansion(const ConfigPotential& config)
{
    switch(config.potentialType) {
    case PT_BSE:
        return PtrPotential(new BasisSetExp(
            config.alpha, config.numCoefsRadial, config.numCoefsAngular,
            *createAnalyticDensity(config)));
    case PT_SPLINE: {
        return PtrPotential(new SplineExp(
            config.numCoefsRadial, config.numCoefsAngular,
            *createAnalyticDensity(config), config.splineRMin, config.splineRMax));
    }
    case PT_CYLSPLINE: {
        if( config.densityType == PT_DEHNEN || 
            config.densityType == PT_FERRERS ||
            config.densityType == PT_MIYAMOTONAGAI ) 
        {   // use potential for initialization, without intermediate DirectPotential step
            return PtrPotential(new CylSplineExp(
                config.numCoefsRadial, config.numCoefsVertical, config.numCoefsAngular,
                *createAnalyticPotential(config),
                config.splineRMin, config.splineRMax, config.splineZMin, config.splineZMax));
        } else {
            return PtrPotential(new CylSplineExp(
                config.numCoefsRadial, config.numCoefsVertical, config.numCoefsAngular, 
                *createAnalyticDensity(config), 
                config.splineRMin, config.splineRMax, config.splineZMin, config.splineZMax));
        }
    }
    default: throw std::invalid_argument("Unknown potential expansion type");
    }
}

/// determines whether the potential is of an expansion type
static bool isPotentialExpansion(PotentialType type)
{
    return type == PT_SPLINE || type == PT_BSE || type == PT_CYLSPLINE || type == PT_MULTIPOLE;
}

/** Universal routine for creating any elementary (non-composite) potential,
    either an analytic one or a potential expansion constructed from a density model
    or loaded from a text file.
*/
static PtrPotential createAnyPotential(const ConfigPotential& params,
    const units::ExternalUnits& converter)
{
    if( params.potentialType == PT_UNKNOWN && !params.fileName.empty() )
        // assume that the file contains coefficients of some potential expansion
        return readPotential(params.fileName);  // (will raise an exception if that wasn't the case)
    else if(isPotentialExpansion(params.potentialType)) {
        if( params.densityType == PT_NBODY && !params.fileName.empty()) {
            // create potential expansion from an N-body snapshot file
            particles::PointMassArrayCar points;
            particles::readSnapshot(params.fileName, converter, points);
            if(points.size()==0)
                throw std::runtime_error("Error loading N-body snapshot from " + params.fileName);
            PtrPotential poten = createPotentialFromPoints(params, points);
            // store coefficients in a text file, 
            // later may load this file instead for faster initialization
            writePotential( (params.fileName + 
                getCoefFileExtension(params.potentialType)), *poten);
            return poten;
        } else if(params.densityType == PT_COEFS && !params.fileName.empty())
            // read coefs and all other parameters from a text file
            return readPotential(params.fileName);
        else
            return createPotentialExpansion(params);
    } else  // elementary potential, or an error
        return createAnalyticPotential(params);
}

// create elementary density
PtrDensity createDensity(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    std::string type = kvmap.getString("type");
    if(utils::stringsEqual(type, DiskDensity::myName()))
        return PtrDensity(new DiskDensity(parseDiskParams(kvmap, converter)));
    if(utils::stringsEqual(type, SpheroidDensity::myName()))
        return PtrDensity(new SpheroidDensity(parseSphrParams(kvmap, converter)));
    return createAnalyticDensity(parseParams(kvmap, converter));
}

// create a potential from several components
PtrPotential createPotential(
    const std::vector<utils::KeyValueMap>& kvmap,
    const units::ExternalUnits& converter)
{
    if(kvmap.size() == 0)
        throw std::runtime_error("Empty list of potential components");
    std::vector<PtrPotential> components;

    // first we isolate all components that are part of GalPot
    std::vector<DiskParam> diskParams;
    std::vector<SphrParam> sphrParams;
    std::vector<ConfigPotential> params;
    for(unsigned int i=0; i<kvmap.size(); i++) {
        std::string type = kvmap[i].getString("type");
        if(utils::stringsEqual(type, DiskDensity::myName())) {
            diskParams.push_back(parseDiskParams(kvmap[i], converter));
        } else if(utils::stringsEqual(type, SpheroidDensity::myName())) {
            sphrParams.push_back(parseSphrParams(kvmap[i], converter));
        } else
            params.push_back(parseParams(kvmap[i], converter));
    }
    // create an array of GalPot components if needed
    if(diskParams.size()>0 || sphrParams.size()>0)
        components = createGalaxyPotentialComponents(diskParams, sphrParams);
    // add other components if they exist
    for(unsigned int i=0; i<params.size(); i++) {
        components.push_back(createAnyPotential(params[i], converter));
    }

    assert(components.size()>0);
    if(components.size() == 1)
        return components[0];
    else
        return PtrPotential(new CompositeCyl(components));
}

// create a potential from one component (which may still turn into a composite potential
// if it happened to be one of GalPot things)
PtrPotential createPotential(
    const utils::KeyValueMap& params,
    const units::ExternalUnits& converter)
{
    return createPotential(std::vector<utils::KeyValueMap>(1, params), converter);
}

// create a potential from INI file
PtrPotential createPotential(
    const std::string& iniFileName, const units::ExternalUnits& converter)
{
    utils::ConfigFile ini(iniFileName);
    std::vector<std::string> sectionNames = ini.listSections();
    std::vector<utils::KeyValueMap> components;
    for(unsigned int i=0; i<sectionNames.size(); i++)
        if(utils::stringsEqual(sectionNames[i].substr(0,9), "Potential"))
            components.push_back(ini.findSection(sectionNames[i]));
    if(components.size() == 0)
        throw std::runtime_error("INI file does not contain any [Potential] section");
    return createPotential(components, converter);
}

///@}
}; // namespace
