#!/usr/bin/python

'''
    This example demonstrates how to simultaneously constrain the parameters of
    an action-based distribution function and the potential, given the observations
    (i.e. an array of stars/particles sampled from the DF in the given potential).

    First we create the array of samples from the known potential and DF;
    then we pretend that we don't know their parameters, and search for them
    using first a deterministic minimization algorithm (Nelder-Mead), and then 
    a series of Markov Chain Monte Carlo passes (using EMCEE ensemble sampler).
    After each pass we display the entire chain (to visually determine when
    when it reaches the equilibrium after the initial burn-in period),
    and the triangle-plot showing the covariation of parameters.

    This example uses the data from the "Spherical/triaxial" working group
    of Gaia Challenge III.
'''
import py_wrapper, numpy
from matplotlib import pyplot
from scipy.optimize import minimize

# compute log-likelihood of DF with given params against an array of actions
def df_likelihood(dfparams, actions):
    dpl = py_wrapper.DistributionFunction(**dfparams)
    norm = dpl.total_mass()
    sumlog = numpy.sum( numpy.log(dpl(actions)/norm) )
    if numpy.isnan(sumlog): sumlog = -numpy.inf
    return sumlog, norm

# convert from parameter space to DF and potential params: 
# note that we apply some non-trivial scaling to make the life easier for the minimizer.
# Parameters are, in order of appearance:
#   -- DF params --
# alpha (DF slope at small J)
# beta  (DF slope at large J)
# log(J0)  (boundary between small and large J)
# ar/at  (ration between coefficients in the linear combination of actions - radial vs any of the other two, for the small J region)
# br/bt  (ration between coefficients in the linear combination of actions - radial vs any of the other two, for the large J region)
#   -- potential params --
# log(M1)  (enclosed mass within a fixed radius R1)
# log(Rscale)  (scale radius for the two-power-law density profile)
# |gamma|  (inner slope of the density profile; take abs value to avoid problems when the best-fit value is near zero)
def unscale_params(args):
    M1     = numpy.exp(args[5])
    Rscale = numpy.exp(args[6])
    gamma  = abs(args[7])
    R1     = 0.5 # fixed radius of 0.5 kpc
    Mtotal = M1 * (1 + Rscale/R1)**(3-gamma)
    RhoScale = Mtotal * (3-gamma) / (4*3.1416*Rscale**3)
    return dict(
        # DF params
        type  = 'DoublePowerLaw',
        alpha = abs(args[0]),
        beta  = args[1],
        j0    = numpy.exp(args[2]),  # will always stay positive
        ar    = 3./(2+args[3])*args[3],
        az    = 3./(2+args[3]),
        aphi  = 3./(2+args[3]),  # ensure that sum of ar*Jr+az*Jz+aphi*Jphi doesn't depend on args[3]
        br    = 3./(2+args[4])*args[4],
        bz    = 3./(2+args[4]),
        bphi  = 3./(2+args[4]),
        jcore = 0.,
        norm  = 1.), dict( \
        # potential params
        type  = 'TwoPowerLawSpheroid',
        mass  = Mtotal,
        gamma = gamma,
        beta  = 3,
        densityNorm = RhoScale,
        scaleRadius = Rscale)

def model_likelihood(args, particles):
    dfparams, potparams = unscale_params(args)
    print "J0=%6.5g, alpha=%6.5g, ar=%6.5g, br=%6.5g, beta=%6.5g; mass=%6.5g, gamma=%4g, Rscale=%6.5g: " \
        % (dfparams['j0'], dfparams['alpha'], dfparams['ar'], dfparams['br'], dfparams['beta'], \
        potparams['mass'], potparams['gamma'], potparams['scaleRadius']),
    if  dfparams['j0']<1e-1 or dfparams['j0']>1e4 or \
        dfparams['alpha']<0 or dfparams['alpha']>=3 or \
        dfparams['beta']<=3 or dfparams['beta']>10 or \
        dfparams['ar']<=0 or dfparams['ar']>=3.0 or \
        dfparams['br']<=0 or dfparams['br']>=3.0 or \
        potparams['gamma']<0 or potparams['gamma']>=2 or \
        potparams['mass']>1e14 or potparams['mass']<1e4 or \
        potparams['scaleRadius']>1e2 or potparams['scaleRadius']<1e-2:
        print 'Out of range'
        return -numpy.inf
    try:
        pot = py_wrapper.Potential(**potparams)
        af  = py_wrapper.ActionFinder(pot)
        actions = af(particles[:,:6])
        loglikelihood, norm = df_likelihood(dfparams, actions)
        print "LogL=%8g" % loglikelihood
        return loglikelihood
    except ValueError as err:
        print "Exception ", err
        return -numpy.inf

# function to minimize using the deterministic algorithm
def model_search_fnc(args, particles):
    ll = -model_likelihood(args, particles)
    if numpy.isinf(ll) or numpy.isnan(ll): ll=100*len(particles)
    return ll

def prepare_samples(dfp, pp, N):
    pot = py_wrapper.Potential(**pp)
    df  = py_wrapper.DistributionFunction(**dfp)
    samp= py_wrapper.sample(pot=pot, df=df, N=N)
    #numpy.savetxt("test.txt",numpy.hstack((samp[0], samp[1].reshape(-1,1))))
    return samp[0]

def get_dm_density(pp):
    pot   = py_wrapper.Potential(**pp)
    radii = numpy.logspace(-2., 1.).reshape(-1,1)
    xyz   = numpy.hstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii)))
    dens  = pot.density(xyz)
    return radii,dens

# plot DM density profile for an ensemble of parameters, together with the 'true' profile
def plot_dm_density(params, trueparams):
    for i in range(len(params[:,0])):
        radii,dens = get_dm_density(unscale_params(params[i,:])[1])
        pyplot.plot(radii, dens, color='k', alpha=0.4)
    radii,dens = get_dm_density(trueparams)
    pyplot.plot(radii, dens, color='r', lw=2)
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.show()

# plot velocity dispersions of tracer population (determined by their DF),
# together with the binned original data points
def plot_vel_disp(params, points):
    # plot velocities of original data points (divided by 3)
    radii = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2) ** 0.5
    velsq = (points[:,3]**2 + points[:,4]**2 + points[:,5]**2)
    velradsq = ((points[:,0]*points[:,3] + points[:,1]*points[:,4] + points[:,2]*points[:,5]) / radii) ** 2
    veltansq = (velsq - velradsq) * 0.5  # per each of the two tangential directions
    sorted_radii = numpy.sort(radii)
    hist_boundaries = numpy.hstack((sorted_radii[0:50:25], sorted_radii[50:-25:50], sorted_radii[-1]))
    sumnum,_ = numpy.histogram(radii, bins=hist_boundaries, weights=numpy.ones_like(radii))
    sumvelradsq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=velradsq)
    sumveltansq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=veltansq)
    pyplot.plot(radii, (velsq)**0.5 / 3, '.', color='grey', label=r'$v_{1d}$', ms=3.)

    # plot velocity dispersion of model DF
    radii = numpy.logspace(-2., 2., 25).reshape(-1,1)
    xyz   = numpy.hstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii)))
    for i in range(len(params[:,0])):
        dfparams, potparams = unscale_params(params[i,:])
        pot = py_wrapper.Potential(**potparams)
        df  = py_wrapper.DistributionFunction(**dfparams)
        mom = py_wrapper.moments(df=df, pot=pot, point=xyz, dens=False, vel=False, vel2=True)
        pyplot.plot(radii, mom[:,0]**0.5, color='r')
        pyplot.plot(radii, mom[:,1]**0.5, color='g')
        pyplot.plot(radii, mom[:,2]**0.5, color='b')

    # overplot the binned velocity dispersion of original data points
    sigmar = (sumvelradsq/sumnum)**0.5
    sigmat = (sumveltansq/sumnum)**0.5
    # emulate steps plot
    pyplot.plot(numpy.hstack(zip(hist_boundaries[:-1], hist_boundaries[1:])), numpy.hstack(zip(sigmar, sigmar)), \
        color='r', lw=2, label=r'$\sigma_r$')
    pyplot.plot(numpy.hstack(zip(hist_boundaries[:-1], hist_boundaries[1:])), numpy.hstack(zip(sigmat, sigmat)), \
        color='b', lw=2, label=r'$\sigma_t$')
    pyplot.xscale('log')
    pyplot.yscale('linear')
    pyplot.legend(loc='upper right')
    pyplot.show()

#### Main program starts here
py_wrapper.set_units(mass=1, length=1, velocity=1)
#particles = numpy.loadtxt("../temp/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_1000_0.dat")
particles = numpy.loadtxt("../temp/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_1000_0.dat")
true_potential_params = dict(type='TwoPowerLawSpheroid', gamma=0, beta=3, densityNorm=400e6, scaleRadius=1)

# do a deterministic search to find (the initial guess for) the best-fit parameters
# of distribution function describing these particles
initparams = numpy.array([2.0, 4.0, 1.0, 1.0, numpy.log(10.), numpy.log(1e9), numpy.log(2.), 0.5])
initdisp   = numpy.array([0.1, 0.1, 0.1, 1.0, 0.1, 0.2 , 0.1, 0.1])

# perform several iterations of search, to avoid getting stuck in a local minimum, 
# until the log-likelihood ceases to improve
prevloglike = -1e5
while True:
    print 'Starting deterministic search'
    result = minimize(model_search_fnc, initparams, args=(particles,), method='Nelder-Mead',
        options=dict(maxiter=500, maxfev=500, disp=True, eps=1e-4))
    print 'result=', result.x, 'LogL=', result.fun, result.message
    initparams = result.x
    #plot_dm_density(initparams.reshape(1,-1), true_potential_params)
    #plot_vel_disp(initparams.reshape(1,-1), particles)
    initloglike= -result.fun
    if initloglike - prevloglike < 1.0: break
    else: print 'Improved log-likelihood by', initloglike - prevloglike
    prevloglike = initloglike

truevalues = result.x   # pretend that the minimum found by deterministic search is the right one

# explore the parameter space around the best-fit values using the MCMC method
import emcee, triangle
ndim = len(initparams)
nwalkers = 32
nsteps   = 100  # number of MC steps
# initial coverage of parameter space
initwalkers = [initparams + initdisp*numpy.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, model_likelihood, args=(particles,), threads=2)
while True:  # run several passes until the user gets bored
    print 'Starting MCMC'
    sampler.run_mcmc(initwalkers, nsteps)

    initwalkers = sampler.chain[:,-1,:]  # restart the next pass from the last values in the Markov chain
    plot_vel_disp(initwalkers[::4,:], particles)
    plot_dm_density(initwalkers, true_potential_params)

    # show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps)
    fig,axes = pyplot.subplots(ndim+1, 1, sharex=True)
    for i in range(ndim):
        axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.4)
    # last panel shows the evolution of log-likelihood for the ensemble of walkers
    axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.4)
    maxloglike = numpy.max(sampler.lnprobability.T)
    if initloglike>maxloglike: maxloglike = initloglike
    axes[-1].set_ylim(maxloglike-10, maxloglike)   # restrict the range of log-likelihood to 10 units down from its maximum
    fig.tight_layout(h_pad=0.)
    pyplot.show()

    # show the posterior distribution of parameters
    nstat   = nsteps/2   # number of last MC steps to show
    samples = sampler.chain[:, -nstat:, :]. reshape((-1, ndim))
    triangle.corner(samples, \
        labels=[r'$\alpha$', r'$\beta$', r'ln($J_0$)', r'$a_r/a_{z,\phi}$', r'$b_r/b_{z,\phi}$', \
        r'ln(M(<0.5 kpc)', r'ln($R_{scale}$)', r'$\gamma$'], \
        quantiles=[0.16, 0.5, 0.84], truths=truevalues)
    pyplot.show()
    print "Acceptance fraction: ", numpy.mean(sampler.acceptance_fraction)  # should be in the range 0.2-0.5
    print "Autocorrelation time: ", sampler.acor  # should be considerably shorter than the total number of steps
