/** \file    coord.h 
    \brief   General-purpose coordinate types and routines
    \author  Eugene Vasiliev
    \date    2015

This module provides the general framework for working with different coordinate systems.

It is heavily templated but this shouldn't intimidate the end user, because 
the most important data structures and routines have dedicated non-templated aliases.

The fundamental data types are the following:

- coordinate systems (the simplest ones have no parameters at all);
- positions and position-velocity pairs in different coordinate systems;
- an abstract class for a scalar function defined in a particular coordinate system;
- gradients and hessians of scalar functions in different coordinate systems;
- coefficients of coordinate transformations between different systems:
  derivatives of destination coords by source coords (i.e. the jacobian matrix) and 
  second derivatives of destination coords by source coords.

The fundamental routines operating on these structures are the following:

- conversion of position and position-velocity from one coordinate system to another;
- computation of coefficients of coordinate transformation (first/second derivatives);
- transformation of gradients and hessians;
- the "all-mighty function" that uses the above primitives to perform the following task:
  suppose we have a class that computes the value, gradient and hessian of a scalar function 
  in a particular coordinate system ("evaluation CS"), and we need these quantities 
  in a different system ("output CS").
  The routine transforms the input coordinates from outputCS to evalCS, along with their 
  derivatives; computes the value, gradient and hessian in evalCS, transforms them back 
  to outputCS. A modification of this routine uses another intermediate CS for the situation 
  when a direct transformation is not implemented.
  The main application of this routine is the computation of potentials and forces 
  in different coordinate systems.
*/
#pragma once
#include "math_base.h"

/** Classes and routines for representing position/velocity points, 
    gradients and hessians of scalar functions (e.g., gravitational potential), 
    and transformations between coordinate systems.
*/
namespace coord {

/// \name   Primitive data types: coordinate systems
///@{

/// trivial coordinate systems don't have any parameters, 
/// their class names are simply used as tags in the rest of the code

/// cartesian coordinate system (galactocentric)
struct Car{
    static const char* name() { return "Cartesian"; }
};

/// cylindrical coordinate system (galactocentric)
struct Cyl{
    static const char* name() { return "Cylindrical"; }
};

/// spherical coordinate system (galactocentric)
struct Sph{
    static const char* name() { return "Spherical"; }
};

//  less trivial:
/** prolate spheroidal coordinate system, defined by a single parameter 
    delta>0 (squared interfocal distance).
    The traditionally used two parameters alpha and gamma (e.g., de Zeeuw 1985) 
    are not independent, so we define  delta = gamma - alpha.
*/
struct ProlSph{
    const double delta;      ///< = gamma - alpha > 0
    ProlSph(double _delta);  ///< delta is _squared_ interfocal distance
    static const char* name() { return "Prolate spheroidal"; }
};

///@}
/// \name   Primitive data types: symmetry in 3d space
///@{
    
/** defines the symmetry properties of a function in three-dimensional space */
enum SymmetryType{ 
    ST_NONE         = 0, ///< no symmetry whatsoever
    // basic symmetries:
    ST_XREFLECTION  = 1, ///< change of sign in x (flip about yz plane)
    ST_YREFLECTION  = 2, ///< change of sign in y
    ST_ZREFLECTION  = 4, ///< change of sign in z
    ST_REFLECTION   = 8, ///< reflection about origin (change of sign of all coordinates simultaneously)
    ST_ZROTATION    =16, ///< rotation about z axis
    ST_ROTATION     =32, ///< rotation about arbitrary axis
    // composite symmetries:
    /// triaxial - reflection about principal planes (change of sign of any coordinate):
    /// note that while the combination of reflection symmetries about all three principal planes
    /// implies the reflection symmetry about origin (mirroring), the converse is not true, 
    /// that's why these are separate concepts; if all three plane-reflection symmetries are present,
    /// then mirror-reflection is implied, and this all is encoded in the ST_TRIAXIAL value
    ST_TRIAXIAL     = ST_XREFLECTION | ST_YREFLECTION | ST_ZREFLECTION | ST_REFLECTION, 
    ST_AXISYMMETRIC = ST_TRIAXIAL | ST_ZROTATION,    ///< axial symmetry combined with plane symmetry
    ST_SPHERICAL    = ST_AXISYMMETRIC | ST_ROTATION, ///< spherical symmetry
};

///@}
/// \name   Primitive data types: position in different coordinate systems
///@{

/// position in arbitrary coordinates:
/// the data types are defined as templates with the template parameter
/// being any of the coordinate system names defined above
template<typename coordSysT> struct PosT;

/// position in cartesian coordinates
template<> struct PosT<Car>{
    double x, y, z;   ///< three cartesian coordinates
    PosT<Car>() {};
    PosT<Car>(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {};
};
/// an alias to templated type specialization of position in cartesian coordinates
typedef struct PosT<Car> PosCar;

/// position in cylindrical coordinates
template<> struct PosT<Cyl>{
    double R;   ///< cylindrical radius = sqrt(x^2+y^2)
    double z;   ///< z coordinate
    double phi; ///< azimuthal angle in x-y plane [0:2pi)
    PosT<Cyl>() {};
    PosT<Cyl>(double _R, double _z, double _phi) : R(_R), z(_z), phi(_phi) {};
};
typedef struct PosT<Cyl> PosCyl;

/// position in spherical coordinates
template<> struct PosT<Sph>{
    double r;     ///< spherical radius
    double theta; ///< polar angle [0:pi) - 0 means along z axis in positive direction, pi is along z in negative direction, pi/2 is in x-y plane
    double phi;   ///< azimuthal angle in x-y plane [0:2pi)
    PosT<Sph>() {};
    PosT<Sph>(double _r, double _theta, double _phi) : r(_r), theta(_theta), phi(_phi) {};
};
typedef struct PosT<Sph> PosSph;

/** position in prolate spheroidal coordinates.
    We use a somewhat different definition from de Zeeuw 1985, namely: 
    the value of `nu` keeps track of the sign of z, so that the conversion between cylindrical
    and prolate spheroidal coordinates is invertible. */
template<> struct PosT<ProlSph>{
    double lambda;  ///< lies in the range [delta:infinity)
    double nu;      ///< lies in the range [-delta:delta]; negative for z<0
    double phi;     ///< usual azimuthal angle
    const ProlSph& coordsys;  ///< a point means nothing without specifying its coordinate system
    PosT<ProlSph>(double _lambda, double _nu, double _phi, const ProlSph& _coordsys):
        lambda(_lambda), nu(_nu), phi(_phi), coordsys(_coordsys) {};
};
typedef struct PosT<ProlSph> PosProlSph;

///@}
/// \name   Primitive data types: velocity in different coordinate systems
///@{

/// velocity in arbitrary coordinates
template<typename coordSysT> struct VelT;

/// velocity in cartesian coordinates
template<> struct VelT<Car> {
    double vx, vy, vz;   ///< components of velocity along three cartesian axes
    VelT<Car>() {};
    VelT<Car>(double _vx, double _vy, double _vz) : vx(_vx), vy(_vy), vz(_vz) {};
};
/// an alias to templated type specialization of velocity for cartesian coordinates
typedef struct VelT<Car> VelCar;

/// velocity in cylindrical coordinates 
/// (this is not the same as time derivative of position in these coordinates!)
template<> struct VelT<Cyl> {
    double vR, vz, vphi;
    VelT<Cyl>() {};
    VelT<Cyl>(double _vR, double _vz, double _vphi) : vR(_vR), vz(_vz), vphi(_vphi) {};
};
typedef struct VelT<Cyl> VelCyl;

/// velocity in spherical coordinates
/// (this is not the same as time derivative of position in these coordinates!)
template<> struct VelT<Sph> {
    double vr, vtheta, vphi;
    VelT<Sph>() {};
    VelT<Sph>(double _vr, double _vtheta, double _vphi) : vr(_vr), vtheta(_vtheta), vphi(_vphi) {};
};
typedef struct VelT<Sph> VelSph;

///@}
/// \name   Primitive data types: second moments of velocity in different coordinate systems
///@{

/// second moment of velocity in arbitrary coordinates
template<typename coordSysT> struct Vel2T;

/// velocity in cartesian coordinates
template<> struct Vel2T<Car> {
    double vx2, vy2, vz2, vxvy, vxvz, vyvz;
};
typedef struct Vel2T<Car> Vel2Car;

/// second moment of velocity in cylindrical coordinates 
template<> struct Vel2T<Cyl> {
    double vR2, vz2, vphi2, vRvz, vRvphi, vzvphi;
};
typedef struct Vel2T<Cyl> Vel2Cyl;

/// second moment of velocity in spherical coordinates
template<> struct Vel2T<Sph> {
    double vr2, vtheta2, vphi2, vrvtheta, vrvphi, vthetavphi;
};
typedef struct Vel2T<Sph> Vel2Sph;

///@}
/// \name   Primitive data types: position-velocity pairs in different coordinate systems
///@{

/// combined position and velocity in arbitrary coordinates
template<typename coordSysT> struct PosVelT;

/// combined position and velocity in cartesian coordinates
template<> struct PosVelT<Car>: public PosCar, public VelCar {
    PosVelT<Car>() {};
    /// initialize from position and velocity
    PosVelT<Car>(const PosCar& pos, const VelCar& vel) : PosCar(pos), VelCar(vel) {}
    /// initialize from explicitly given numbers
    PosVelT<Car>(double _x, double _y, double _z, double _vx, double _vy, double _vz) :
        PosCar(_x, _y, _z), VelCar(_vx, _vy, _vz) {}
    /// initialize from an array of 6 floats (i.e., from a serialized array)
    PosVelT<Car>(const double p[]) :
        PosCar(p[0], p[1], p[2]), VelCar(p[3], p[4], p[5]) {}
    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=x; out[1]=y; out[2]=z; out[3]=vx; out[4]=vy; out[5]=vz; }
};
/// an alias to templated type specialization of position and velocity for cartesian coordinates
typedef struct PosVelT<Car> PosVelCar;

/// combined position and velocity in cylindrical coordinates
template<> struct PosVelT<Cyl>: public PosCyl, public VelCyl {
    PosVelT<Cyl>() {};
    /// initialize from position and velocity
    PosVelT<Cyl>(const PosCyl& pos, const VelCyl& vel) : PosCyl(pos), VelCyl(vel) {}
    /// initialize from explicitly given numbers
    PosVelT<Cyl>(double _R, double _z, double _phi, double _vR, double _vz, double _vphi) :
        PosCyl(_R, _z, _phi), VelCyl(_vR, _vz, _vphi) {};
    /// initialize from an array of 6 floats (i.e., from a serialized array)
    PosVelT<Cyl>(const double p[]) :
        PosCyl(p[0], p[1], p[2]), VelCyl(p[3], p[4], p[5]) {};
    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=R; out[1]=z; out[2]=phi; out[3]=vR; out[4]=vz; out[5]=vphi; }
};
typedef struct PosVelT<Cyl> PosVelCyl;

/// combined position and velocity in spherical coordinates
template<> struct PosVelT<Sph>: public PosSph, public VelSph {
    PosVelT<Sph>() {};
    /// initialize from position and velocity
    PosVelT<Sph>(const PosSph& pos, const VelSph& vel) : PosSph(pos), VelSph(vel) {}
    /// initialize from explicitly given numbers
    PosVelT<Sph>(double _r, double _theta, double _phi, double _vr, double _vtheta, double _vphi) :
        PosSph(_r, _theta, _phi), VelSph(_vr, _vtheta, _vphi) {};
    /// initialize from an array of 6 floats (i.e., from a serialized array)
    PosVelT<Sph>(const double p[]) :
        PosSph(p[0], p[1], p[2]), VelSph(p[3], p[4], p[5]) {};
    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=r; out[1]=theta; out[2]=phi; out[3]=vr; out[4]=vtheta; out[5]=vphi; }
};
typedef struct PosVelT<Sph> PosVelSph;

/// position and velocity in prolate spheroidal coordinates
template<> struct PosVelT<ProlSph>: public PosProlSph{
    double lambdadot, nudot, phidot;  ///< time derivatives of position variables
    PosVelT<ProlSph>(const PosProlSph& pos, double _lambdadot, double _nudot, double _phidot):
        PosProlSph(pos), lambdadot(_lambdadot), nudot(_nudot), phidot(_phidot) {};
    void unpack_to(double *out) const {
        out[0]=lambda; out[1]=nu; out[2]=phi; out[3]=lambdadot; out[4]=nudot; out[5]=phidot; }
};
typedef struct PosVelT<ProlSph> PosVelProlSph;
    
///@}
/// \name   Primitive data types: gradient of a scalar function in different coordinate systems
///@{

/// components of a gradient in a given coordinate system
template<typename coordSysT> struct GradT;

/// gradient of scalar function in cartesian coordinates
template<> struct GradT<Car>{
    double dx, dy, dz;
};
/// an alias to templated type specialization of gradient for cartesian coordinates
typedef struct GradT<Car> GradCar;

/// gradient of scalar function in cylindrical coordinates
template<> struct GradT<Cyl>{
    double dR, dz, dphi;
};
typedef struct GradT<Cyl> GradCyl;

/// gradient of scalar function in spherical coordinates
template<> struct GradT<Sph>{
    double dr, dtheta, dphi;
};
typedef struct GradT<Sph> GradSph;

/// gradient of scalar function in prolate spheroidal coordinates
template<> struct GradT<ProlSph>{
    double dlambda, dnu, dphi;
};
typedef struct GradT<ProlSph> GradProlSph;

///@}
/// \name   Primitive data types: hessian of a scalar function in different coordinate systems
///@{

/// components of a hessian of a scalar function (matrix of its second derivatives)
template<typename coordSysT> struct HessT;

/// Hessian of scalar function F in cartesian coordinates: d2F/dx^2, d2F/dxdy, etc
template<> struct HessT<Car>{
    double dx2, dy2, dz2, dxdy, dydz, dxdz;
};
typedef struct HessT<Car> HessCar;

/// Hessian of scalar function in cylindrical coordinates
template<> struct HessT<Cyl>{
    double dR2, dz2, dphi2, dRdz, dzdphi, dRdphi;
};
typedef struct HessT<Cyl> HessCyl;

/// Hessian of scalar function in spherical coordinates
template<> struct HessT<Sph>{
    double dr2, dtheta2, dphi2, drdtheta, dthetadphi, drdphi;
};
typedef struct HessT<Sph> HessSph;

/// Hessian of scalar function in prolate spheroidal coordinates
template<> struct HessT<ProlSph>{
    double dlambda2, dnu2, dlambdadnu;  ///< note: derivatives by phi are assumed to be zero
};
typedef struct HessT<ProlSph> HessProlSph;

///@}
/// \name   Abstract interface classes for scalar functions
///@{

/** Prototype of a scalar function which is computed in a particular coordinate system */
template<typename coordSysT>
class IScalarFunction {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    /** Evaluate any combination of value, gradient and hessian of the function at a given point.
        Each of these quantities is computed and stored in the output pointer if it was not NULL. */
    virtual void evalScalar(const PosT<coordSysT>& x,
        double* value=0,
        GradT<coordSysT>* deriv=0,
        HessT<coordSysT>* deriv2=0) const=0;
};

///@}
/// \name   Data types containing conversion coefficients between different coordinate systems
///@{

/** derivatives of coordinate transformation from source to destination 
    coordinate systems (srcCS=>destCS): derivatives of destination variables 
    w.r.t.source variables, aka Jacobian */
template<typename srcCS, typename destCS> struct PosDerivT;

/** instantiations of the general template for derivatives of coordinate transformations
    are separate structures for each pair of coordinate systems */
template<> struct PosDerivT<Car, Cyl> {
    double dRdx, dRdy, dphidx, dphidy;
};
template<> struct PosDerivT<Car, Sph> {
    double drdx, drdy, drdz, dthetadx, dthetady, dthetadz, dphidx, dphidy;
};
template<> struct PosDerivT<Cyl, Car> {
    double dxdR, dxdphi, dydR, dydphi;
};
template<> struct PosDerivT<Cyl, Sph> {
    double drdR, drdz, dthetadR, dthetadz;
};
template<> struct PosDerivT<Sph, Car> {
    double dxdr, dxdtheta, dxdphi, dydr, dydtheta, dydphi, dzdr, dzdtheta;
};
template<> struct PosDerivT<Sph, Cyl> {
    double dRdr, dRdtheta, dzdr, dzdtheta;
};
template<> struct PosDerivT<Cyl, ProlSph> {
    double dlambdadR, dlambdadz, dnudR, dnudz;
};
template<> struct PosDerivT<ProlSph, Cyl> {
    double dRdlambda, dzdlambda, dRdnu, dzdnu;
};
    

/** second derivatives of coordinate transformation from source to destination 
    coordinate systems (srcCS=>destCS): d^2(dest_coord)/d(source_coord1)d(source_coord2) */
template<typename srcCS, typename destCS> struct PosDeriv2T;

/** instantiations of the general template for second derivatives of coordinate transformations */
template<> struct PosDeriv2T<Cyl, Car> {
    double d2xdRdphi, d2xdphi2, d2ydRdphi, d2ydphi2;
};
template<> struct PosDeriv2T<Sph, Car> {
    double d2xdrdtheta, d2xdrdphi, d2xdtheta2, d2xdthetadphi, d2xdphi2,
           d2ydrdtheta, d2ydrdphi, d2ydtheta2, d2ydthetadphi, d2ydphi2,
           d2zdrdtheta, d2zdtheta2;
};
template<> struct PosDeriv2T<Car, Cyl> {
    double d2Rdx2, d2Rdxdy, d2Rdy2, d2phidx2, d2phidxdy, d2phidy2;
};
template<> struct PosDeriv2T<Sph, Cyl> {
    double d2Rdrdtheta, d2Rdtheta2, d2zdrdtheta, d2zdtheta2;
};
template<> struct PosDeriv2T<Car, Sph> {
    double d2rdx2, d2rdxdy, d2rdxdz, d2rdy2, d2rdydz, d2rdz2,
        d2thetadx2, d2thetadxdy, d2thetadxdz, d2thetady2, d2thetadydz, d2thetadz2,
        d2phidx2, d2phidxdy, d2phidy2;
};
template<> struct PosDeriv2T<Cyl, Sph> {
    double d2rdR2, d2rdRdz, d2rdz2, d2thetadR2, d2thetadRdz, d2thetadz2;
};
template<> struct PosDeriv2T<Cyl, ProlSph> {
    double d2lambdadR2, d2lambdadRdz, d2lambdadz2, d2nudR2, d2nudRdz, d2nudz2;
};
template<> struct PosDeriv2T<ProlSph, Cyl> {
    double d2Rdlambda2, d2Rdlambdadnu, d2Rdnu2, d2zdlambda2, d2zdlambdadnu, d2zdnu2;
};
    
///@}
/// \name   Routines for conversion between position/velocity in different coordinate systems
///@{

/** universal templated conversion function for positions:
    template parameters srcCS and destCS may be any of the coordinate system names.
    This template function shouldn't be used directly, because the return type depends 
    on the template and hence cannot be automatically inferred by the compiler. 
    Instead, named functions for each target coordinate system are defined below. */
template<typename srcCS, typename destCS>
PosT<destCS> toPos(const PosT<srcCS>& from);

/** templated conversion taking the parameters of coordinate system into account */
template<typename srcCS, typename destCS>
PosT<destCS> toPos(const PosT<srcCS>& from, const destCS& coordsys);

/** templated conversion functions for positions 
    with names reflecting the target coordinate system. */
template<typename srcCS>
inline PosCar toPosCar(const PosT<srcCS>& from) { return toPos<srcCS, Car>(from); }
template<typename srcCS>
inline PosCyl toPosCyl(const PosT<srcCS>& from) { return toPos<srcCS, Cyl>(from); }
template<typename srcCS>
inline PosSph toPosSph(const PosT<srcCS>& from) { return toPos<srcCS, Sph>(from); }

/** universal templated conversion function for coordinates and velocities:
    template parameters srcCS and destCS may be any of the coordinate system names */
template<typename srcCS, typename destCS>
PosVelT<destCS> toPosVel(const PosVelT<srcCS>& from);

/** templated conversion functions for coordinates and velocities
    with names reflecting the target coordinate system. */
template<typename srcCS>
inline PosVelCar toPosVelCar(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Car>(from); }
template<typename srcCS>
inline PosVelCyl toPosVelCyl(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Cyl>(from); }
template<typename srcCS>
inline PosVelSph toPosVelSph(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Sph>(from); }

/** templated conversion taking the parameters of coordinate system into account */
template<typename srcCS, typename destCS>
PosVelT<destCS> toPosVel(const PosVelT<srcCS>& from, const destCS& coordsys);
    
/** trivial conversions */
template<> inline PosCar toPos<Car,Car>(const PosCar& p) { return p;}
template<> inline PosCyl toPos<Cyl,Cyl>(const PosCyl& p) { return p;}
template<> inline PosSph toPos<Sph,Sph>(const PosSph& p) { return p;}
template<> inline PosVelCar toPosVel<Car,Car>(const PosVelCar& p) { return p;}
template<> inline PosVelCyl toPosVel<Cyl,Cyl>(const PosVelCyl& p) { return p;}
template<> inline PosVelSph toPosVel<Sph,Sph>(const PosVelSph& p) { return p;}

///@}
/// \name   Routines for conversion between position in different coordinate systems with derivatives
///@{

/** universal templated function for coordinate conversion that provides derivatives of transformation.
    Template parameters srcCS and destCS may be any of the coordinate system names;
    \param[in]  from specifies the point in srcCS coordinate system;
    \param[out] deriv will contain derivatives of the transformation 
                (destination coords over source coords);
    \param[out] deriv2 if not NULL, will contain second derivatives of the coordinate transformation;
    \return     point in destCS coordinate system. */
template<typename srcCS, typename destCS>
PosT<destCS> toPosDeriv(const PosT<srcCS>& from, 
    PosDerivT<srcCS, destCS>* deriv, PosDeriv2T<srcCS, destCS>* deriv2=0);

/** templated conversion with derivatives, taking the parameters of coordinate system into account */
template<typename srcCS, typename destCS>
PosT<destCS> toPosDeriv(const PosT<srcCS>& from, const destCS& coordsys,
    PosDerivT<srcCS, destCS>* deriv, PosDeriv2T<srcCS, destCS>* deriv2=0);

///@}
/// \name   Routines for conversion of gradients and hessians between coordinate systems
///@{

/** templated function for transforming a gradient to a different coordinate system */
template<typename srcCS, typename destCS>
GradT<destCS> toGrad(const GradT<srcCS>& src, const PosDerivT<destCS, srcCS>& deriv);

/** templated function for transforming a hessian to a different coordinate system */
template<typename srcCS, typename destCS>
HessT<destCS> toHess(const GradT<srcCS>& srcGrad, const HessT<srcCS>& srcHess, 
    const PosDerivT<destCS, srcCS>& deriv, const PosDeriv2T<destCS, srcCS>& deriv2);

/** All-mighty routine for evaluating the value of a scalar function and its derivatives 
    in a different coordinate system (evalCS), and converting them to the target 
    coordinate system (outputCS). */
template<typename evalCS, typename outputCS>
void evalAndConvert(const IScalarFunction<evalCS>& F,
    const PosT<outputCS>& pos, double* value=0, GradT<outputCS>* deriv=0, HessT<outputCS>* deriv2=0)
{
    bool needDeriv = deriv!=0 || deriv2!=0;
    bool needDeriv2= deriv2!=0;
    GradT<evalCS> evalGrad;
    HessT<evalCS> evalHess;
    PosDerivT <outputCS, evalCS> coordDeriv;
    PosDeriv2T<outputCS, evalCS> coordDeriv2;
    const PosT<evalCS> evalPos = needDeriv ? 
        toPosDeriv<outputCS, evalCS>(pos, &coordDeriv, needDeriv2 ? &coordDeriv2 : 0) :
        toPos<outputCS, evalCS>(pos);
    // compute the function in transformed coordinates
    F.evalScalar(evalPos, value, needDeriv ? &evalGrad : 0, needDeriv2 ? &evalHess : 0);
    if(deriv)  // ... and convert gradient/hessian back to output coords if necessary.
        *deriv  = toGrad<evalCS, outputCS> (evalGrad, coordDeriv);
    if(deriv2)
        *deriv2 = toHess<evalCS, outputCS> (evalGrad, evalHess, coordDeriv, coordDeriv2);
};

/** The same routine for conversion of gradient and hessian 
    in the case that the computation requires the parameters of coordinate system evalCS */
template<typename evalCS, typename outputCS>
void evalAndConvert(const IScalarFunction<evalCS>& F,
    const PosT<outputCS>& pos, const evalCS& coordsys,
    double* value=0, GradT<outputCS>* deriv=0, HessT<outputCS>* deriv2=0)
{
    bool needDeriv = deriv!=0 || deriv2!=0;
    bool needDeriv2= deriv2!=0;
    GradT<evalCS> evalGrad;
    HessT<evalCS> evalHess;
    PosDerivT <outputCS, evalCS> coordDeriv;
    PosDeriv2T<outputCS, evalCS> coordDeriv2;
    const PosT<evalCS> evalPos = needDeriv ? 
        toPosDeriv<outputCS, evalCS>(pos, coordsys, &coordDeriv, needDeriv2 ? &coordDeriv2 : 0) :
        toPos<outputCS, evalCS>(pos, coordsys);
    // compute the function in transformed coordinates
    F.evalScalar(evalPos, value, needDeriv ? &evalGrad : 0, needDeriv2 ? &evalHess : 0);
    if(deriv)  // ... and convert gradient/hessian back to output coords if necessary.
        *deriv  = toGrad<evalCS, outputCS> (evalGrad, coordDeriv);
    if(deriv2)
        *deriv2 = toHess<evalCS, outputCS> (evalGrad, evalHess, coordDeriv, coordDeriv2);
};

/// trivial instantiation of the above function for the case that conversion is not necessary
template<typename CS> 
void evalAndConvert(const IScalarFunction<CS>& F, const PosT<CS>& pos, 
    double* value, GradT<CS>* deriv, HessT<CS>* deriv2)
{  F.evalScalar(pos, value, deriv, deriv2); }

/** An even mightier routine for evaluating the value of a scalar function,
    its gradient and hessian, in a different coordinate system (evalCS), 
    and converting them to the target coordinate system (outputCS)
    through an intermediate coordinate system (intermedCS), 
    for the situation when a direct transformation is not available. */
template<typename evalCS, typename intermedCS, typename outputCS>
void evalAndConvertTwoStep(const IScalarFunction<evalCS>& F,
    const PosT<outputCS>& pos, double* value=0, GradT<outputCS>* deriv=0, HessT<outputCS>* deriv2=0)
{
    bool needDeriv = deriv!=0 || deriv2!=0;
    bool needDeriv2= deriv2!=0;
    GradT<evalCS> evalGrad;
    HessT<evalCS> evalHess;
    GradT<intermedCS> intermedGrad;
    HessT<intermedCS> intermedHess;
    PosDerivT <outputCS, intermedCS> coordDerivOI;
    PosDeriv2T<outputCS, intermedCS> coordDeriv2OI;
    PosDerivT <intermedCS, evalCS> coordDerivIE;
    PosDeriv2T<intermedCS, evalCS> coordDeriv2IE;
    const PosT<intermedCS> intermedPos = needDeriv ? 
        toPosDeriv<outputCS, intermedCS>(pos, &coordDerivOI, needDeriv2 ? &coordDeriv2OI : 0) :
        toPos<outputCS, intermedCS>(pos);
    const PosT<evalCS> evalPos = needDeriv ? 
        toPosDeriv<intermedCS, evalCS>(intermedPos, &coordDerivIE, needDeriv2 ? &coordDeriv2IE : 0) :
        toPos<intermedCS, evalCS>(intermedPos);
    // compute the function in transformed coordinates
    F.evalScalar(evalPos, value, needDeriv ? &evalGrad : 0, needDeriv2 ? &evalHess : 0);
    if(needDeriv)  // may be needed for either grad or hess (or both)
        intermedGrad = toGrad<evalCS, intermedCS> (evalGrad, coordDerivIE);
    if(deriv)
        *deriv  = toGrad<intermedCS, outputCS> (intermedGrad, coordDerivOI);
    if(deriv2) {
        intermedHess = toHess<evalCS, intermedCS> (evalGrad, evalHess, coordDerivIE, coordDeriv2IE);
        *deriv2 = toHess<intermedCS, outputCS> (intermedGrad, intermedHess, coordDerivOI, coordDeriv2OI);
    }
};

/** The same routine for the case that evalCS requires the parameters of coordinate system */
template<typename evalCS, typename intermedCS, typename outputCS>
void evalAndConvertTwoStep(const IScalarFunction<evalCS>& F,
    const PosT<outputCS>& pos, const evalCS& coordsys,
    double* value=0, GradT<outputCS>* deriv=0, HessT<outputCS>* deriv2=0)
{
    bool needDeriv = deriv!=0 || deriv2!=0;
    bool needDeriv2= deriv2!=0;
    GradT<evalCS> evalGrad;
    HessT<evalCS> evalHess;
    GradT<intermedCS> intermedGrad;
    HessT<intermedCS> intermedHess;
    PosDerivT <outputCS, intermedCS> coordDerivOI;
    PosDeriv2T<outputCS, intermedCS> coordDeriv2OI;
    PosDerivT <intermedCS, evalCS> coordDerivIE;
    PosDeriv2T<intermedCS, evalCS> coordDeriv2IE;
    const PosT<intermedCS> intermedPos = needDeriv ? 
        toPosDeriv<outputCS, intermedCS>(pos, &coordDerivOI, needDeriv2 ? &coordDeriv2OI : 0) :
        toPos<outputCS, intermedCS>(pos);
    const PosT<evalCS> evalPos = needDeriv ? 
        toPosDeriv<intermedCS, evalCS>(intermedPos, coordsys, &coordDerivIE, needDeriv2 ? &coordDeriv2IE : 0) :
        toPos<intermedCS, evalCS>(intermedPos, coordsys);
    // compute the function in transformed coordinates
    F.evalScalar(evalPos, value, needDeriv ? &evalGrad : 0, needDeriv2 ? &evalHess : 0);
    if(needDeriv)  // may be needed for either grad or hess (or both)
        intermedGrad = toGrad<evalCS, intermedCS> (evalGrad, coordDerivIE);
    if(deriv)
        *deriv  = toGrad<intermedCS, outputCS> (intermedGrad, coordDerivOI);
    if(deriv2) {
        intermedHess = toHess<evalCS, intermedCS> (evalGrad, evalHess, coordDerivIE, coordDeriv2IE);
        *deriv2 = toHess<intermedCS, outputCS> (intermedGrad, intermedHess, coordDerivOI, coordDeriv2OI);
    }
};

/** Specialized conversion routine for spherically-symmetric functions.
    Convert the derivatives of a simple function that only depends on the spherical radius 
    into gradients and hessians in a target coordinate system (outputCS). */
template<typename outputCS>
void evalAndConvertSph(const math::IFunction& F,
    const PosT<outputCS>& pos, double* value=0, GradT<outputCS>* deriv=0, HessT<outputCS>* deriv2=0);

///@}

/// convenience functions to extract the value of angular momentum and its z-component
template<typename coordT> double Ltotal(const PosVelT<coordT>& p);
template<typename coordT> double Lz(const PosVelT<coordT>& p);

}  // namespace coord
