#!/usr/bin/python

"""
Demonstrates the usage of user-defined Python routines for density and distribution function.
Note that whenever such objects are used, this turns off OpenMP parallelization in the C++ library,
because the Python callback functions cannot be used in a multi-threaded context.
"""
import agama, math

# user-defined density profile
mass = 3.0
radius = 1.5
def MyPlummer(x, y, z):
    return  3 / (4*math.pi) * mass * radius**2 * \
        (x**2 + y**2 + z**2 + radius**2) ** -2.5

# original density/potential model using a C++ object
pot_orig = agama.Potential(type="Plummer", mass=mass, scaleradius=radius)
# potential approximation computed from the user-supplied density profile
pot_appr = agama.Potential(type="Multipole", density=MyPlummer, splineRmin=0.01, splineRmax=100, numCoefsRadial=20)
pot0_orig = pot_orig(0,0,0)
pot0_appr = pot_appr(0,0,0)
print "Phi_appr(0)=%.8g  (true value=%.8g)" % (pot0_appr, pot0_orig)
print "rho_appr(1)=%.8g  (true value=%.8g,  user value=%.8g)" % \
    (pot_appr.density(1,0,0), pot_orig.density(1,0,0), MyPlummer(1,0,0))

# user-defined distribution function
J0 = 2.0
beta = 5.0
def MyDF(Jr, Jz, Jphi):
    return (Jr + Jz + abs(Jphi) + J0) ** -beta

# original DF using a C++ object
df_orig = agama.DistributionFunction(type="DoublePowerLaw", J0=J0, alpha=0, beta=beta, norm=2*math.pi**3)
# to compute the total mass, we create an proxy instance of DistributionFunction class
# with a composite DF consisting of a single component (the user-supplied function),
# which provides the totalMass() method that the Python user function itself does not have
mass_orig = df_orig.totalMass()
mass_my = agama.DistributionFunction(MyDF).totalMass()
print "DF mass=%.8g  (orig value=%.8g)" % (mass_my, mass_orig)

# GalaxyModel objects constructed from the C++ DF and from the Python function
# (without the need of a proxy object; in fact, GalaxyModel.df is itself a proxy object)
gm_orig = agama.GalaxyModel(df=df_orig, pot=pot_orig)
gm_my   = agama.GalaxyModel(df=MyDF, pot=pot_appr)
# note that different potentials were used for gm_orig and gm_my, so the results will slightly disagree

# DF moments computed from the C++ object and from the Python function
dens_orig, veldisp_orig = gm_orig.moments(point=(1,0,0))
print "original DF at r=1: density=%.8g, sigma_r=%.8g, sigma_t=%.8g" % \
    (dens_orig, veldisp_orig[0], veldisp_orig[1])
dens_my, veldisp_my = gm_my.moments(point=(1,0,0))
print "user-def DF at r=1: density=%.8g, sigma_r=%.8g, sigma_t=%.8g" % \
    (dens_my, veldisp_my[0], veldisp_my[1])
# gm_my.df.total_mass() will give the same result as above

if abs(pot0_orig-pot0_appr)<1e-6 and abs(mass_orig-mass_my)<1e-6 and abs(dens_orig-dens_my)<1e-6:
    print "\033[1;32mALL TESTS PASSED\033[0m"
