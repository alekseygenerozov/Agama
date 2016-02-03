/** \file    actions_base.h
    \brief   Base classes for actions, angles, and action/angle finders
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once
#include "coord.h"

/** Classes and routines for transformations between position/velocity and action/angle phase spaces */
namespace actions {

/** Actions in arbitrary potential */
struct Actions {
    double Jr;       ///< radial action or its analog, [0..infinity)
    double Jz;       ///< vertical action or its analog, [0..infinity)
    double Jphi;     ///< azimuthal action (equal to the z-component of angular momentum in
                     ///< axisymmetric case, can have any value)
    Actions() {};
    Actions(double _Jr, double _Jz, double _Jphi) : Jr(_Jr), Jz(_Jz), Jphi(_Jphi) {};
};

/** Angles in arbitrary potential */
struct Angles {
    double thetar;   ///< phase angle of radial motion
    double thetaz;   ///< phase angle of vertical motion
    double thetaphi; ///< phase angle of azimuthal motion
    Angles() {};
    Angles(double tr, double tz, double tphi) : thetar(tr), thetaz(tz), thetaphi(tphi) {};
};

/** A combination of both actions and angles */
struct ActionAngles: Actions, Angles {
    ActionAngles() {};
    ActionAngles(const Actions& acts, const Angles& angs) : Actions(acts), Angles(angs) {};
};

/** Frequencies of motion (Omega = dH/dJ) */
struct Frequencies {
    double Omegar;    ///< frequency of radial motion, dH/dJr
    double Omegaz;    ///< frequency of vertical motion, dH/dJz
    double Omegaphi;  ///< frequency of azimuthal motion, dH/dJphi
    Frequencies() {};
    Frequencies(double omr, double omz, double omphi) : Omegar(omr), Omegaz(omz), Omegaphi(omphi) {};
};

/** Derivatives of position/velocity variables w.r.t actions:
    each of three member fields stores the derivative of 6 pos/vel elements by the given action */
struct DerivAct {
    coord::PosVelCyl dbyJr, dbyJz, dbyJphi;
};

/** Derivatives of position/velocity variables w.r.t angles:
    each of three member fields stores the derivative of 6 pos/vel elements by the given angle */
struct DerivAng {
    coord::PosVelCyl dbythetar, dbythetaz, dbythetaphi;
};


/** Base class for action finders, which convert position/velocity pair to action/angle pair */
class BaseActionFinder{
public:
    BaseActionFinder() {};
    virtual ~BaseActionFinder() {};

    /** Evaluate actions for a given position/velocity point in cylindrical coordinates */
    virtual Actions actions(const coord::PosVelCyl& point) const = 0;

    /** Evaluate actions and angles for a given position/velocity point in cylindrical coordinates;
        if the output argument freq!=NULL, also store the frequencies */
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=0) const = 0;

private:
    /// disable copy constructor and assignment operator
    BaseActionFinder(const BaseActionFinder&);
    BaseActionFinder& operator= (const BaseActionFinder&);
};

/** Base class for action/angle mappers, which convert action/angle variables to position/velocity point */
class BaseActionMapper{
public:
    BaseActionMapper() {};
    virtual ~BaseActionMapper() {};

    /** Map a point in action/angle space to a position/velocity in physical space;
        if the output argument freq!=NULL, also store the frequencies */
    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=0) const = 0;
private:
    /// disable copy constructor and assignment operator
    BaseActionMapper(const BaseActionMapper&);
    BaseActionMapper& operator= (const BaseActionMapper&);
};

/** Base class for toy maps used in torus machinery, which provide conversion from action/angle
    to position/velocity variables, and also provide the derivatives of this transformation */
class BaseToyMap: public BaseActionMapper{
public:
    /// return the number of parameters of toy potential
    virtual unsigned int numParams() const = 0;

    /** Convert from action/angles to position/velocity, optionally computing the derivatives;
        if any of the output arguments is NULL, it is not computed.
        \param[in]  actAng are the action/angles;
        \param[out] freq   are the frequencies;
        \param[out] derivAct are the derivatives of pos/vel w.r.t three actions;
        \param[out] derivAng are the derivatives of pos/vel w.r.t three actions;
        \param[out] derivParam are the derivatives of pos/vel w.r.t the parameters of toy potential:
                    if not NULL, must point to an array of length `numParams()`;
        \return     pos/vel coordinates.
    */
    virtual coord::PosVelCyl mapDeriv(
        const ActionAngles& actAng,
        Frequencies* freq=0,
        DerivAct* derivAct=0,
        DerivAng* derivAng=0,
        coord::PosVelCyl* derivParam=0) const = 0;

    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=0) const
    { return mapDeriv(actAng, freq); }
};

/** Base class for canonical maps in action/angle space, which transform from one set of a/a
    variables to another one */
class BaseCanonicalMap{
    BaseCanonicalMap() {};
    virtual ~BaseCanonicalMap() {};

    virtual unsigned int numParams() const = 0;

    /** Map a point in action/angle space to a point in another action/angle space */
    virtual ActionAngles map(const ActionAngles& actAng) const = 0;
private:
    /// disable copy constructor and assignment operator
    BaseCanonicalMap(const BaseCanonicalMap&);
    BaseCanonicalMap& operator= (const BaseCanonicalMap&);
};

}  // namespace action