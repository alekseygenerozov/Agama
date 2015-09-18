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
import sys, py_wrapper, numpy
import emcee, triangle
from scipy.optimize import minimize
from matplotlib import pyplot

def unscale_params(args):
    '''
    Convert from parameter space to DF and potential params: 
    note that we apply some non-trivial scaling to make the life easier for the minimizer.
    Parameters are, in order of appearance:
      -- DF params --
    alpha (DF slope at small J)
    beta  (DF slope at large J)
    log(J0)  (boundary between small and large J)
    ar/at  (ration between coefficients in the linear combination of actions - radial vs any of the other two, for the small J region)
    br/bt  (ration between coefficients in the linear combination of actions - radial vs any of the other two, for the large J region)
      -- potential params --
    log(M1)  (enclosed mass within a fixed radius R1)
    log(Rscale)  (scale radius for the two-power-law density profile)
    |gamma|  (inner slope of the density profile; take abs value to avoid problems when the best-fit value is near zero)
    '''
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

def df_likelihood(dfparams, actions):
    '''
    Compute log-likelihood of DF with given params against an array of actions
    '''
    dpl = py_wrapper.DistributionFunction(**dfparams)
    norm = dpl.total_mass()
    sumlog = numpy.sum( numpy.log(dpl(actions)/norm) )
    if numpy.isnan(sumlog): sumlog = -numpy.inf
    return sumlog, norm

def model_likelihood(params, particles):
    '''
    Compute the likelihood of model (df+potential specified by scaled params)
    against the data (array of Nx6 position/velocity coordinates of tracer particles).
    This is the function to be maximized; if parameters are nonsense it returns -infinity
    '''
    dfparams, potparams = unscale_params(params)
    print "J0=%6.5g, alpha=%6.5g, ar=%6.5g, br=%6.5g, beta=%6.5g; mass=%6.5g, gamma=%4g, Rscale=%6.5g: " \
        % (dfparams['j0'], dfparams['alpha'], dfparams['ar'], dfparams['br'], dfparams['beta'], \
        potparams['mass'], potparams['gamma'], potparams['scaleRadius']),
    # check that the parameters are within allowed range
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

def deterministic_search_fnc(params, particles):
    '''
    function to minimize using the deterministic algorithm
    '''
    ll = -model_likelihood(params, particles)
    if numpy.isinf(ll) or numpy.isnan(ll): ll=100*len(particles)
    return ll

def deterministic_search(initparams, args, user_fnc=None):
    '''
    do a deterministic search to find the best-fit parameters of potential and distribution function.
    perform several iterations of search, to avoid getting stuck in a local minimum,
    until the log-likelihood ceases to improve
    '''
    prevloglike = -1e5
    while True:
        print 'Starting deterministic search'
        result = minimize(deterministic_search_fnc, initparams, args=args, method='Nelder-Mead',
            options=dict(maxiter=500, maxfev=500, disp=True))
        print 'result=', result.x, 'LogL=', result.fun, result.message
        initparams = result.x
        if not user_fnc is None:
            try:
                user_fnc(initparams)
            except:
                print "Exception in user function:", sys.exc_info()
        initloglike= -result.fun
        if initloglike - prevloglike < 1.0:
            return initparams,initloglike
        else:
            print 'Improved log-likelihood by', initloglike - prevloglike
        prevloglike = initloglike

def monte_carlo_search(initwalkers, args, user_fnc=None):
    '''
    Explore the parameter space around the best-fit values using the MCMC method
    initwalkers is the Nwalkers * Nparams array of initial parameters
    '''
    nsteps  = 100  # number of MC steps per pass
    nwalkers, nparams = initwalkers.shape
    sampler = emcee.EnsembleSampler(nwalkers, nparams, model_likelihood, args=args, threads=2)
    prevmaxloglike = None
    while True:  # run several passes until convergence
        print 'Starting MCMC'
        sampler.run_mcmc(initwalkers, nsteps)
        initwalkers = sampler.chain[:,-1,:]    # restart the next pass from the latest values in the Markov chain
        print "Acceptance fraction: ", numpy.mean(sampler.acceptance_fraction)  # should be in the range 0.2-0.5
        print "Autocorrelation time: ", sampler.acor  # should be considerably shorter than the total number of steps
        if not user_fnc is None:
            try:
                user_fnc(sampler.chain, sampler.lnprobability)
            except:
                print "Exception in user function:", sys.exc_info()
        # check for convergence
        maxloglike = numpy.max(sampler.lnprobability)
        avgloglike = numpy.average(sampler.lnprobability[:,-nsteps:])  # avg.log-likelihood during the pass
        avgparams  = [numpy.average(sampler.chain[:,-nsteps:,i]) for i in range(nparams)]
        if not prevmaxloglike is None:
            if maxloglike-prevmaxloglike < 1.0 and \
                abs(avgloglike-prevavgloglike) < 1.0 : pass
            print "Max(prev) log-likelihood= %7g (%7g), avg(prev) log-likelihood= %7g (%7g)" \
                % (maxloglike, prevmaxloglike, avgloglike, prevavgloglike)
            for i in range(nparams):
                print "Parameter %d avg(prev)= %7g (%7g)" % (i, avgparams[i], prevavgparams[i])
        prevmaxloglike = maxloglike
        prevavgloglike = avgloglike
        prevavgparams  = avgparams


###################################  DATA ANALYSIS  ########################################

def compute_df_moments(dfparams, potparams):
    '''
    Compute moments of distribution function (density and velocity dispersion);
    DF and potential are specified by scaled parameters (params)
    Return: a tuple of four arrays: radii, density, radial velocity dispersion and tangential v.d.
    '''
    radii = numpy.logspace(-2., 2., 25).reshape(-1,1)
    xyz   = numpy.hstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii)))
    pot = py_wrapper.Potential(**potparams)
    df  = py_wrapper.DistributionFunction(**dfparams)
    dens, mom = py_wrapper.moments(df=df, pot=pot, point=xyz, dens=True, vel=False, vel2=True)
    return radii, dens, mom[:,0]**0.5, mom[:,1]**0.5

def compute_dm_density(potparams):
    '''
    Compute density profile corresponding to the potential specified by potparams
    Return: a tuple of two arrays: radii, density
    '''
    pot   = py_wrapper.Potential(**potparams)
    radii = numpy.logspace(-2., 1.).reshape(-1,1)
    xyz   = numpy.hstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii)))
    dens  = pot.density(xyz)
    return radii,dens

def compute_orig_moments(points):
    '''
    Compute the moments (density and velocity dispersions in radial and tangential direction)
    of the original data points (tracer particles), binned in radius.
    Return: tuple of four arrays: radii, density, radial velocity dispersion and tangential v.d.,
    where the array of radii is one element longer than the other three arrays, and denotes the bin boundaries,
    and the other arrays contain the values in each bin.
    '''
    radii = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2) ** 0.5
    velsq = (points[:,3]**2 + points[:,4]**2 + points[:,5]**2)
    velradsq = ((points[:,0]*points[:,3] + points[:,1]*points[:,4] + points[:,2]*points[:,5]) / radii) ** 2
    veltansq = (velsq - velradsq) * 0.5  # per each of the two tangential directions
    sorted_radii = numpy.sort(radii)
    # select bin boundaries so that each bin contains 50 data points, or 25 points at the edges of radial interval
    hist_boundaries = numpy.hstack((sorted_radii[0:50:25], sorted_radii[50:-25:50], sorted_radii[-1]))
    sumnum,_ = numpy.histogram(radii, bins=hist_boundaries, weights=numpy.ones_like(radii))
    sumvelradsq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=velradsq)
    sumveltansq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=veltansq)
    binvol = 4*3.1416/3 * (hist_boundaries[1:]**3 - hist_boundaries[:-1]**3)
    density= sumnum/len(points[:,0]) / binvol
    sigmar = (sumvelradsq/sumnum)**0.5
    sigmat = (sumveltansq/sumnum)**0.5
    return hist_boundaries, density, sigmar, sigmat

########################  PLOTTING  ###########################

# plot DM density profile for an ensemble of parameters, together with the 'true' profile
def plot_dm_density(params, trueparams):
    for i in range(len(params[:,0])):
        radii,dens = compute_dm_density(unscale_params(params[i,:])[1])
        pyplot.plot(radii, dens, color='k', alpha=0.4)
    radii,dens = get_dm_density(trueparams)
    pyplot.plot(radii, dens, color='r', lw=2)
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.show()

def plot_tracer_density_veldisp(params, particles):
    for i in range(len(params[:,0])):
        if i>=4: break
        radii,dens,sigmar,sigmat = compute_df_moments(*unscale_params(params[i,:]))
        print dens
        pyplot.plot(radii, sigmar, color='r', alpha=0.4)
        pyplot.plot(radii, sigmat, color='b', alpha=0.4)

    bins,dens,sigmar,sigmat = compute_orig_moments(particles)
    # emulate steps plot
    pyplot.plot(numpy.hstack(zip(bins[:-1], bins[1:])), numpy.hstack(zip(sigmar, sigmar)), \
        color='r', lw=2, label=r'$\sigma_r$')
    pyplot.plot(numpy.hstack(zip(bins[:-1], bins[1:])), numpy.hstack(zip(sigmat, sigmat)), \
        color='b', lw=2, label=r'$\sigma_t$')
    pyplot.xscale('log')
    pyplot.yscale('linear')
    pyplot.legend(loc='upper right')
    pyplot.show()

def plot_time_evol(chain, loglike):
    """
    Show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps)
    """
    ndim = chain.shape[2]
    fig,axes = pyplot.subplots(ndim+1, 1, sharex=True)
    for i in range(ndim):
        axes[i].plot(chain[:,:,i].T, color='k', alpha=0.4)
    # last panel shows the evolution of log-likelihood for the ensemble of walkers
    axes[-1].plot(loglike.T, color='k', alpha=0.4)
    maxloglike = numpy.max(loglike)
    axes[-1].set_ylim(maxloglike-3*ndim, maxloglike)   # restrict the range of log-likelihood arount its maximum
    fig.tight_layout(h_pad=0.)
    pyplot.show()

    """
    # show the posterior distribution of parameters
    nstat   = nsteps/2   # number of last MC steps to show
    samples = sampler.chain[:, -nstat:, :]. reshape((-1, ndim))
    triangle.corner(samples, \
        labels=[r'$\alpha$', r'$\beta$', r'ln($J_0$)', r'$a_r/a_{z,\phi}$', r'$b_r/b_{z,\phi}$', \
        r'ln(M(<0.5 kpc)', r'ln($R_{scale}$)', r'$\gamma$'], \
        quantiles=[0.16, 0.5, 0.84], truths=truevalues)
    pyplot.show()
    """

################  INTERMEDIATE RESULTS  ###############
def user_fnc_mcmc(chain, loglike):
    # store the latest location of all walkers (together with their likelihood)
    numpy.savetxt('bestfit.dat', numpy.hstack((chain[:,-1,:], loglike[:,-1].reshape(-1,1))))
    plot_time_evol(chain, loglike)
    plot_tracer_density_veldisp(chain[:,-1,:], particles)

################  MAIN PROGRAM  ##################

py_wrapper.set_units(mass=1, length=1, velocity=1)
#particles = numpy.loadtxt("../temp/gs100_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_1000_0.dat")
particles = numpy.loadtxt("../temp/gs100_bs050_rcrs100_rarcinf_core_0400mpc3_df_1000_0.dat")
true_potential_params = dict(type='TwoPowerLawSpheroid', gamma=0, beta=3, densityNorm=400e6, scaleRadius=1)
true_tracer_density_params = dict(scaleRadius=1., gamma=1., beta=5.)

try:
    params = numpy.loadtxt('bestfit.dat')
    if params.ndim==2:  # ensemble of realizations
        params = params[:,:-1]  # last column is log-likelihood, discard it
    else:  # single line
        params = params[:-1]  # last item is log-likelihood
except:  # load failed, start afresh
    # initial guess for parameters (scaled)
    params = numpy.array([2.0, 4.0, 1.0, 1.0, numpy.log(10.), numpy.log(1e9), numpy.log(2.), 0.5])

    # first locate the purported best-fit parameters using deterministic algorithm
    params,loglike = deterministic_search(params, (particles,))
    numpy.savetxt('bestfit.dat', numpy.hstack((params,loglike)).reshape(1,-1))

if params.ndim==1:
    # initial coverage of parameter space (dispersion around the current best-fit values)
    nwalkers   = 32
    params = numpy.array([params + (numpy.random.randn(len(params))*0.1 if i>0 else 0) for i in range(nwalkers)])

# explore the landscape of parameters around their best-fit values using MCMC
monte_carlo_search(params, (particles,), user_fnc_mcmc)
