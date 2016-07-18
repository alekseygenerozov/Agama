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
'''
import agama, numpy
from scipy.optimize import minimize

# compute log-likelihood of DF with given params against an array of actions
def df_likelihood(dfparams, actions):
    dpl = agama.DistributionFunction(**dfparams)
    norm = dpl.totalMass()
    sumlog = numpy.sum( numpy.log(dpl(actions)/norm) )
    if numpy.isnan(sumlog): sumlog = -numpy.inf
    return sumlog, norm

# convert from parameter space to DF and potential params: 
# note that we apply some non-trivial scaling to make the life easier for the minimizer
def unscale_params(args):
    return dict(
        # DF params
        type  = 'DoublePowerLaw',
        alpha = args[0],
        beta  = args[1],
        j0    = numpy.exp(args[2]),  # will always stay positive
        ar    = 3./(2+args[3])*args[3],
        az    = 3./(2+args[3]),
        aphi  = 3./(2+args[3]),  # ensure that sum of ar*Jr+az*Jz+aphi*Jphi doesn't depend on args[3]
        br    = 1.,
        bz    = 1.,
        bphi  = 1.,
        jcore = 0.,
        norm  = 1.), dict( \
        # potential params
        type  = 'Dehnen', #'TwoPowerLawSpheroid',
        densityNorm = numpy.exp(args[4]),
        mass  = numpy.exp(args[4]),
        gamma = args[5],
        #beta  = 4.,
        scaleRadius = numpy.exp(args[6]))

def model_likelihood(args, particles):
    dfparams, potparams = unscale_params(args)
    print "J0=%6.5g, alpha=%6.5g, ar=%6.5g, az=%6.5g, beta=%6.5g; mass=%6.5g, gamma=%4g, Rscale=%6.5g: " \
        % (dfparams['j0'], dfparams['alpha'], dfparams['ar'], dfparams['az'], dfparams['beta'], \
        potparams['mass'], potparams['gamma'], potparams['scaleRadius']),
    if dfparams['j0']<1e-10 or dfparams['j0']>1e10 or dfparams['alpha']<0 or dfparams['alpha']>=3 or \
        dfparams['beta']<=3 or dfparams['beta']>12 or dfparams['ar']<=0 or dfparams['ar']>=3.0 or \
        potparams['gamma']<0 or potparams['gamma']>=2:
        print 'Out of range'
        return -numpy.inf
    try:
        pot = agama.Potential(**potparams)
        #actf= agama.ActionFinder(pot)
        actions = agama.actions(point=particles[:,:6], pot=pot, ifd=1e-3)
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

def prepare_samples(dfp, pp):
    #dfp = dict(type='DoublePowerLaw', norm=1, ar=1.5, az=0.75, aphi=0.75, br=1, bz=1, bphi=1, alpha=1, beta=4, j0=1000)
    #pp  = dict(type='Dehnen', mass=1e+08, gamma=0.5, scaleRadius=500)
    pot = agama.Potential(**pp)
    df  = agama.DistributionFunction(**dfp)
    samp= agama.sample(pot=pot, df=df, N=1000)
    #numpy.savetxt("test.txt",numpy.hstack((samp[0], samp[1].reshape(-1,1))))
    return samp[0]

#### Main program starts here
agama.set_units(mass=1, length=1e-3, velocity=1)
#particles = numpy.loadtxt("test.txt")

# create the samples with known parameters of DF and potential
truevalues = [1., 4., numpy.log(1000.), 2., numpy.log(1e8), 0.5, numpy.log(500.)]
particles  = prepare_samples(*unscale_params(truevalues))

# do a deterministic search to find (the initial guess for) the best-fit parameters
# of distribution function describing these particles
initparams = numpy.array([1.5, 6.0, 1.0, 1.0, 25.0, 1.0, 10.0])
initdisp   = numpy.array([0.1, 0.1, 0.1, 0.1, 0.2 , 0.1, 0.1])
# perform several iterations of search, to avoid getting stuck in a local minimum
for iters in range(3):
    result = minimize(model_search_fnc, initparams, args=(particles,), method='Nelder-Mead',
        options=dict(maxiter=500, maxfev=500, disp=True, eps=1e-4))
    print 'result=', result.x, 'LogL=', result.fun, result.message
    initparams = result.x
initloglike= -result.fun

# explore the parameter space around the best-fit values using the MCMC method
import emcee, matplotlib.pyplot as plt, triangle
print 'Starting MCMC'
ndim = len(initparams)
nwalkers = 32
nsteps   = 100  # number of MC steps
# initial coverage of parameter space
initwalkers = [initparams + initdisp*numpy.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, model_likelihood, args=(particles,), threads=2)
while True:  # run several passes until the user gets bored
    sampler.run_mcmc(initwalkers, nsteps)

    # show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps)
    fig,axes = plt.subplots(ndim+1, 1, sharex=True)
    for i in range(ndim):
        axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.4)
    axes[-1].plot(sampler.lnprobability.T, color='k', alpha=0.4)
    axes[-1].set_ylim(initloglike, numpy.max(sampler.lnprobability.T))
    fig.tight_layout(h_pad=0.)
    plt.show()

    # show the posterior distribution of parameters
    nstat   = nsteps/2   # number of last MC steps to show
    samples = sampler.chain[:, -nstat:, :]. reshape((-1, ndim))
    triangle.corner(samples, \
        labels=[r'$\alpha$', r'$\beta$', r'ln($J_0$)', r'$a_r/a_{z,\phi}$', r'ln(M)', r'$\gamma$', r'ln($R_{scale}$)'], 
        quantiles=[0.16, 0.5, 0.84], truths=truevalues)
    plt.show()
    print "Acceptance fraction: ", numpy.mean(sampler.acceptance_fraction)  # should be in the range 0.2-0.5
    print "Autocorrelation time: ", sampler.acor  # should be considerably shorter than the total number of steps
    initwalkers = sampler.chain[:,-1,:]  # restart the next pass from the last values in the Markov chain
