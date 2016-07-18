#!/usr/bin/python

"""
Create a simple spherical self-consistent model determined by its distribution function in terms of actions
"""
import agama, numpy, matplotlib.pyplot as plt

# the distribution function defining the model
df = agama.DistributionFunction(type="DoublePowerLaw", J0=1, beta=6, norm=10)
mass = df.totalMass()
scaleradius = 5   # educated guess

# initial guess for the density profile
dens = agama.Density(type='Plummer', mass=mass, scaleradius=scaleradius)
print "DF mass=", mass, ' Density mass=', dens.totalMass()  # should be identical

# define the self-consistent model consisting of a single component
params = dict(rminSph=0.1, rmaxSph=100., sizeRadialSph=25, lmaxAngularSph=0)
comp = agama.Component(df=df, dens=dens, disklike=False, **params)
scm = agama.SelfConsistentModel(**params)
scm.components=[comp]

# prepare visualization
r=numpy.logspace(-1.5,1.5)
xyz=numpy.vstack((r,r*0,r*0)).T
plt.plot(r, dens.density(xyz), label='Init density')

# perform several iterations of self-consistent modelling procedure
for i in range(5):
    scm.iterate()
    print 'Iteration', i, ' Phi(0)=', scm.pot(0,0,0), ' Mass=', scm.pot.totalMass()
    plt.plot(r, scm.pot.density(xyz), label='Iteration #'+str(i))

# save the final density/potential profile
scm.pot.export("simple_scm.coef_mul")

# show the results
plt.legend(loc='lower left')
plt.xlabel("r")
plt.ylabel(r'$\rho$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-7, 2e-3)
plt.xlim(0.03, 30)
plt.show()
