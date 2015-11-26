/** \file    galaxymodel.h
    \brief   A complete galaxy model
    \date    2015
    \author  Eugene Vasiliev, Payel Das
*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"
#include "particles_base.h"

/// Complete specification of a galaxy model
namespace galaxymodel{

/** Data-only structure defining a galaxy model: 
    a combination of potential, action finder, and distribution function.
    Its purpose is to temporarily bind together the three common ingredients that are passed
    to various functions; however, as it only keeps references and not shared pointers, 
    it should not generally be used for a long-term storage.
*/
struct GalaxyModel{
public:
    const potential::BasePotential&     potential;  ///< gravitational potential
    const actions::BaseActionFinder&    actFinder;  ///< action finder for the given potential
    const df::BaseDistributionFunction& distrFunc;  ///< distribution function expressed in terms of actions

    /** Create an instance of the galaxy model from the three ingredients */
    GalaxyModel(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df) :
    potential(pot), actFinder(af), distrFunc(df) {}
};

/** Compute density, first-order, and second-order moments of velocity 
    in polar cyclindrical coordinates; if some of them are not needed,
    pass NULL as the corresponding argument, and it will not be computed.
    \param[in]  model  is the galaxy model (potential + DF + action finder);
    \param[in]  point  is the position at which the quantities should be computed;
    \param[in]  reqRelError is the required relative error in the integral;
    \param[in]  maxNumEval  is the maximum number of evaluations in integral;
    \param[out] density  will contain the integral of DF over all velocities;
    \param[out] velocityFirstMoment  will contain the mean streaming velocity;
    \param[out] velocitySecondMoment will contain the tensor of mean squared velocity components;
    \param[out] densityErr  will contain the error estimate of density;
    \param[out] velocityFirstMomentErr  will contain the error estimate of 1st velocity moment;
    \param[out] velocitySecondMomentErr will contain the error estimate of 2nd velocity moment;
*/
void computeMoments(const GalaxyModel& model,
    const coord::PosCyl& point, const double reqRelError, const int maxNumEval,
    double* density, coord::VelCyl* velocityFirstMoment, coord::Vel2Cyl* velocitySecondMoment,
    double* densityErr, coord::VelCyl* velocityFirstMomentErr, coord::Vel2Cyl* velocitySecondMomentErr);


/** Compute the value of 'projected distribution function' at the given point
    specified by two coordinates in the sky plane and line-of-sight velocity.
    \param[in]  model  is the galaxy model;
    \param[in]  R is the cylindrical radius;
    \param[in]  vz is the line-of-sight velocity;
    \param[in]  vz_error is the assumed velocity error (assumed Gaussian):
    if nonzero, then the DF is additionally convolved with the error function;
    \param[in]  reqRelError is the required relative error in the integral;
    \param[in]  maxNumEval  is the maximum number of evaluations in integral;
    \param[out] error  if not NULL, will contain the error estimate;
    \param[out] numEval if not NULL, will contain the actual number of evaluations;
    \return  the value of projected DF.
*/
double computeProjectedDF(const GalaxyModel& model,
    const double R, const double vz, const double vz_error=0,
    const double reqRelError=1e-4, const int maxNumEval=1e3,
    double* error=0, int* numEval=0);


/** Compute the projected moments of distribution function:
    surface density and line-of-sight velocity dispersion at a given projected radius.
    \param[in]  model  is the galaxy model;
    \param[in]  R is the cylindrical radius;
    \param[in]  reqRelError is the required relative error in the integral;
    \param[in]  maxNumEval  is the maximum number of evaluations in integral;
    \param[out] surfaceDensity will contain the computed surface density;
    \param[out] losvdisp will contain the line-of-sight velocity dispersion (dimension: v^2);
    \param[out] surfaceDensityErr if not NULL, will contain the error estimate for density;
    \param[out] losvdispErr if not NULL, will contain the error estimate for velocity dispersion;
    \param[out] numEval if not NULL, will contain the actual number of evaluations.
*/
void computeProjectedMoments(const GalaxyModel& model, const double R,
    const double reqRelError, const int maxNumEval,
    double& surfaceDensity, double& losvdisp,
    double* surfaceDensityErr=0, double* losvdispErr=0, int* numEval=0);


/** Generate N-body samples of the distribution function 
    by sampling in action/angle space:
    sample actions directly from DF and angles uniformly from [0:2pi]^3,
    then use torus machinery to convert from action/angles to position/velocity.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \param[out] points will contain array of particles (position/velocity/mass)
    sampled from the distribution function;
    \param[out] actions (optional) will be filled with values of actions
    corresponding to each point; if not needed may pass NULL as this argument.
*/
void generateActionSamples(const GalaxyModel& model,
    const unsigned int numPoints, particles::PointMassArrayCar &points,
    std::vector<actions::Actions>* actions=0);


/** Generate N-body samples of the distribution function 
    by sampling in position/velocity space:
    use action finder to compute the actions corresponding to the given point,
    and evaluate the value of DF at the given actions.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \param[out] points will contain array of particles (position/velocity/mass)
    sampled from the distribution function;
*/
void generatePosVelSamples(const GalaxyModel& model,
    const unsigned int numPoints, particles::PointMassArrayCar &points);


/** Sample the density profile by discrete points.
    \param[in]  dens  is the density model;
    \param[in]  numPoints  is the required number of sampling points;
    \param[out] points is the array to be filled with the sampled coordinates.
*/
void generateDensitySamples(const potential::BaseDensity& dens,
    const unsigned int numPoints, particles::PointMassArray<coord::PosCyl>& points);

}  // namespace