#!/usr/bin/python

'''
    This example demonstrates how to find best-fit parameters of an action-based
    distribution function that matches the given N-body snapshot.

    The N-body model itself corresponds to a spherically-symmetric isotropic
    Hernquist profile, and we fit it with a double-power-law distribution function
    of Posti et al.2015. We use the exact potential (i.e., do not compute it
    from the N-body model itself, nor try to vary its parameters, although
    both options are possible), and compute actions for all particles only once.
    Then we scan the parameter space of DF, finding the maximum of the likelihood
    function with a multidimensional minimization algorithm.
    This takes a few hundred iterations to converge.

    This Python script is almost equivalent to the C++ test program test_df_fit.cpp,
    up to the difference in implementation of Nelder-Mead minimization algorithm.

    Additionally, we use the MCMC implementation EMCEE to explore the confidence
    intervals of model parameters around their best-fit values
'''
import agama, numpy
from scipy.optimize import minimize

# compute log-likelihood of DF with given params against an array of points
def model_likelihood(params, points):
    print "J0=%6.5g, alpha=%6.5g, ar=%6.5g, az=%6.5g, beta=%6.5g: " \
        % (params['j0'], params['alpha'], params['ar'], params['az'], params['beta']),
    try:
        dpl = agama.DistributionFunction(params)
        norm = dpl.total_mass()
        sumlog = numpy.sum( numpy.log(dpl(points)/norm) )
        print "LogL=%8g, norm=%6.5g" % (sumlog, norm)
        return sumlog
    except ValueError as err:
        print "Exception ", err
        return -1000.*len(points)

# convert from parameter space to DF params: note that we apply
# some non-trivial scaling to make the life easier for the minimizer
def dfparams(args):
    return dict(
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
        norm  = 1.)

# function to minimize
def model_search_fnc(args, actions):
    return -model_likelihood(dfparams(args), actions)

# function to maximize
def model_search_emcee(args, actions):
    return model_likelihood(dfparams(args), actions)

def main():
    pot = agama.Potential(type="Dehnen", mass=1, scaleRadius=1.)
    actf= agama.ActionFinder(pot)
    particles = numpy.loadtxt("../temp/hernquist.dat", skiprows=1)
    actions = actf(particles[:,:6])

    # do a parameter search to find best-fit distribution function describing these particles
    initparams = numpy.array([2.0, 4.0, 0.0, 1.0])
    result = minimize(model_search_fnc, initparams, args=(actions,), method='Nelder-Mead', 
        options=dict(maxiter=1000, maxfev=500, disp=True))
    print 'result=',result.x, result.message

    # explore the parameter space around the best-fit values using the MCMC chain
    import emcee, matplotlib.pyplot as plt, triangle
    print 'Starting MCMC'
    ndim = len(initparams)
    nwalkers = 32
    # initial coverage of parameter space
    initwalkers = [initparams + 0.1*numpy.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model_search_emcee, args=(actions,))
    sampler.run_mcmc(initwalkers, 300)

    # show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps)
    fig,axes = plt.subplots(ndim, 1, sharex=True)
    for i in range(ndim):
        axes[i].plot(sampler.chain[:,:,i].T, color='k', alpha=0.4)
    fig.tight_layout(h_pad=0.)
    plt.show()

    # show the posterior distribution of parameters
    num_burnin = 100   # number of initial steps to discard
    samples = sampler.chain[:, num_burnin:, :]. reshape((-1, ndim))
    triangle.corner(samples, \
        labels=[r'$\alpha$', r'$\beta$', r'ln($J_0$)', r'$a_r/a_{z,\phi}$'], 
        quantiles=[0.16, 0.5, 0.84], truths=result.x)
    plt.show()
    print "Acceptance fraction: ", numpy.mean(sampler.acceptance_fraction)  # should be in the range 0.2-0.5
    print "Autocorrelation time: ", sampler.acor  # should be considerably shorter than the total number of steps

main()