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
import sys, os, re, agama, numpy, scipy, matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
import emcee, triangle

###################$ GLOBAL CONSTANTS $###############

nsteps_deterministic  = 500   # number of steps per pass in deterministic minimizer
nsteps_mcmc           = 200   # number of MC steps per pass
nwalkers_mcmc         = 32    # number of independent MC walkers
nthreads_mcmc         = 4     # number of parallel threads in MCMCMCMCMC
initial_disp_mcmc     = 0.01  # initial dispersion of parameters carried by walkers around their best-fit values
full_phase_space_info = False # whether to use full phase-space information (3 positions and 3 velocities)
                              # or only the projected data (cylindrical radius and line-of-sight velocity)
use_resampling        = True  # whether to create resampled array of input particles by filling in the missing components,
                              # or marginalize over them using deterministic integration
num_samples           = 1000  # each input data point is represented by this number of samples
                              # which fill in the missing values of coordinate and velocity components
vel_error             = 0.0   # assumed observational error on velocity components (add noise to velocity if it is non-zero)

#################### DEFINITION OF MODEL PARAMETERS #######################

class ModelParams:
    '''
    Class that represents the parameters space (used by the model-search routines)
    and converts from this space to the actual parameters of distribution function and potential,
    applying some non-trivial scaling to make the life easier for the minimizer.
    Parameters are, in order of appearance:
      -- potential params --
    lg(rho_0)  (density normalization at Rscale, in units of Msun/Kpc^3)
    lg(Rscale) (scale radius for the two-power-law density profile, in units of Kpc)
    |gamma|    (inner slope of the density profile; take abs value to avoid problems when the best-fit value is near zero)
    beta       (outer slope of the density profile)
      -- DF params --
    |alpha|    (DF slope at small J; take abs value to avoid problems when the best-fit value is near zero)
    beta       (DF slope at large J)
    ar         (coefficient for the radial action in the linear combination of actions
               for the small J region; the sum of coefficients for all three actions is taken to be unity)
    br         (same for the large J region)
    lg(J0)     (boundary between small and large J, in units of Kpc*km/s)
    '''
    def __init__(self, filename):
        '''
        Initialize starting values of scaled parameters and and define upper/lower limits on them;
        also obtain the true values by analyzing the input file name
        '''
        self.init_values = numpy.array( [numpy.log10(5e8),  numpy.log10(2.0), .5, 4.0, 1.5, 6.0, 0.4, 0.4, numpy.log10(10.)])
        self.min_values  = numpy.array( [numpy.log10(1e6),  numpy.log10(0.1), 0., 2.5, 0.0, 3.2, 0.1, 0.1, numpy.log10(0.1)])
        self.max_values  = numpy.array( [numpy.log10(1e10), numpy.log10(100), 2., 5.0, 2.8, 12., 0.9, 0.9, numpy.log10(1e3)])
        self.parse_true_values(filename)
        self.labels      = (r'$\log(\rho_0)$', r'$\log(R_{scale})$', r'$\gamma$', r'$\beta$', \
            r'$\alpha_{DF}$', r'$\beta_{DF}$', r'$a_r$', r'$b_r$', r'$\log(J_0)$')
        self.num_pot_params = 4  # potential params come first in the list
        self.num_df_params  = 5  # DF params come last in the list

    def parse_true_values(self,filename):
        m = re.match(r'gs(\d+)_bs(\d+)_rcrs([a-z\d]+)_rarc([a-z\d]+)_([a-z]+)_(\d+)mpc3', filename)
        n = re.match(r'data_([ch])_rh(\d+)_rs(\d+)_gs(\d+)', filename)
        if m:
            self.true_values = numpy.array( [ \
                numpy.log10(float(m.group(6))*1e6), numpy.log10(1.0), 1.0 if m.group(5)=='cusp' else 0.0, 3.0, \
                numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan] )
            self.true_tracer_params = dict(scaleRadius=float(m.group(3))*0.01, \
                gamma=float(m.group(1))*0.01, beta=float(m.group(2))*0.1)
        elif n:
            if n.group(1)=='c':
                self.true_values = numpy.array( [ \
                    numpy.log10(3.021516e7), numpy.log10(float(n.group(2))), 0.0, 4.0, \
                    numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan ] )
            else:
                self.true_values = numpy.array( [ \
                    numpy.log10(2.387329e7), numpy.log10(float(n.group(2))), 1.0, 4.0, \
                    numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan ] )
            self.true_tracer_params = dict(scaleRadius=1.75 if n.group(3)=='175' else 0.5, \
                 gamma=float(n.group(4))*0.1, beta=5)
        else:
            print "Can't determine true parameters!"
            self.true_values = self.init_values
            self.true_tracer_params    = dict(scaleRadius=numpy.nan, gamma=numpy.nan, beta=numpy.nan)
        if 'err' in filename:
            vel_error = 2
            print "Assumed error of",vel_error,"km/s in velocity"
        else:
            vel_error = 0
        self.true_potential_params = dict(type='TwoPowerLawSpheroid', \
            densityNorm=10**self.true_values[0], scaleRadius=10**self.true_values[1], \
            gamma=self.true_values[2], beta=self.true_values[3])

    def unscale(self, params):
        return \
            dict(     # potential params
                type  = 'TwoPowerLawSpheroid',
                densityNorm = 10**params[0],
                scaleRadius = 10**params[1],
                gamma = abs(params[2]),
                beta  = params[3] ), \
            dict(     # DF params
                type  = 'DoublePowerLaw',
                alpha = abs(params[4]),
                beta  = params[5],
                ar    = params[6],
                az    = (1-params[6])/2,
                aphi  = (1-params[6])/2,
                br    = params[7],
                bz    = (1-params[7])/2,
                bphi  = (1-params[7])/2,
                j0    = 10**params[8],  # will always stay positive
                jcore = 0.,
                norm  = 1. )

    def prior(self,params):
        '''
        Return prior log-probability of the scaled parameters,
        or -infinity if they are outside the allowed range
        '''
        return 0 if numpy.all( params >= self.min_values ) and numpy.all( params <= self.max_values ) else -numpy.inf


######################## SPHERICAL MODEL DEPROJECTION #########################

def bin_indices(length):
    '''
    Devise a scheme for binning the particles in projected radius,
    so that the inner- and outermost bins contain 10-15 particles,
    and the intermediate ones - Nptbin particles.
    '''
    Nptbin = numpy.maximum(length**0.5 * 2, 20)
    return numpy.hstack((0, 10, 25, \
        numpy.linspace(50, length-50, (length-100)/Nptbin).astype(int), \
        length-25, length-11, length-1))

class SphericalModel:
    '''
    Construct a spherically symmetric model using the information about
    projected particle positions and line-of-sight velocities
    '''
    def __init__(self, particles):
        # sort particles in projected radius
        particles_sorted = particles[ numpy.argsort((particles[:,0]**2 + particles[:,1]**2) ** 0.5) ]

        # create binning in projected radii
        Radii   = (particles_sorted[:,0]**2 + particles_sorted[:,1]**2) ** 0.5
        indices = bin_indices(len(Radii))

        # compute the cumulative mass profile from input particles (M(R) - mass inside projected radius)
        # assuming equal-mass particles and total mass = 1
        cumulMass = numpy.linspace(1./len(Radii), 1., len(Radii))

        # create a smoothing spline for log(M/(Mtotal-M)) as a function of log(R), 
        # using points from the interval indices[1]..indices[-2], and spline knots at Radii[indices]
        self.spl_mass = agama.SplineApprox( \
            numpy.log(Radii[indices[1:-1]]), \
            numpy.log(Radii[indices[1]:indices[-2]]), \
            numpy.log(cumulMass[indices[1]:indices[-2]] / (1 - cumulMass[indices[1]:indices[-2]])), \
            smooth=2.0 )

        # compute 3d density at the same radial grid
        rho_grid   = numpy.log([self.rho_integr(R) for R in Radii[indices]])
        good_elems = numpy.where(numpy.isfinite(rho_grid))[0]
        if(len(good_elems)<len(rho_grid)):
            print "Invalid density encountered at r=", \
                Radii[indices[numpy.where(numpy.logical_not(numpy.isfinite(rho_grid)))]]

        # initialize an interpolating spline for 3d density (log-log scaled)
        self.spl_rho = scipy.interpolate.InterpolatedUnivariateSpline( \
            numpy.log(Radii[indices[good_elems]]), rho_grid[good_elems])

        # store the derivatives of the spline at endpoints (for extrapolation)
        self.logrmin = numpy.log(Radii[indices[ 0]])
        self.derrmin = self.spl_rho(self.logrmin,1)
        self.logrmax = numpy.log(Radii[indices[-1]])
        self.derrmax = self.spl_rho(self.logrmax,1)
        print 'Density slope: inner=',self.derrmin,', outer=',self.derrmax
        if self.derrmin>0: self.derrmin=0
        if self.derrmax>-3: self.derrmax=-3

        # cumulative kinetic energy (up to a constant factor) $\int_0^R \Sigma(R') \sigma_{los}^2(R') 2\pi R' dR'$
        cumulEkin  = numpy.cumsum(particles_sorted[:,5]**2) / len(Radii)
        self.total_Ekin = cumulEkin[-1]
        self.spl_Ekin = agama.SplineApprox( \
            numpy.log(Radii[indices]), \
            numpy.log(Radii[indices[0]:indices[-1]]), \
            numpy.log(cumulEkin[indices[0]:indices[-1]] / \
            (self.total_Ekin - cumulEkin[indices[0]:indices[-1]])), \
            smooth=2.0 )

    def cumul_mass(self, R):
        ''' Return M(<R) '''
        return 1 / (1 + numpy.exp(-self.spl_mass(numpy.log(R))))

    def surface_density(self, R):
        ''' Return surface density: Sigma(R) = 1 / (2 pi R)  d[ M(<R) ] / dR '''
        lnR = numpy.log(R)
        val = numpy.exp(self.spl_mass(lnR))
        return self.spl_mass(lnR, 1) * val / ( 2*3.1416 * R**2 * (1+val)**2 )

    def sigma_los(self, R):
        ''' Return line-of-sight velocity dispersion:
            sigma_los^2(R) = 1 / (2 pi R Sigma(R))  d[ cumulEkin(R) ] / dR '''
        lnR = numpy.log(R)
        val = numpy.exp(self.spl_Ekin(lnR))
        return ( self.total_Ekin * self.spl_Ekin(lnR, 1) * val / \
            ( 2*3.1416 * R**2 * (1 + val)**2 * self.surface_density(R) ) )**0.5

    def integrand(self, t, r, spl):
        ''' integrand in the density or pressure deprojection formula, depending on the spline passed as argument '''
        R = (r**2+(t/(1-t))**2) ** 0.5
        lnR = numpy.log(R)
        val = numpy.exp(spl(lnR))
        der = spl(lnR, 1)
        der2= spl(lnR, 2)
        dSdR= val / ( 2*3.1416 * R**3 * (1+val)**2 ) * (der2 - 2*der + der**2*(1-val)/(1+val) )
        return -1/3.1416 * dSdR / (1-t) / (r**2*(1-t)**2 + t**2)**0.5

    def rho_integr(self, r):
        ''' Return 3d density rho(r) computed by integration of the deprojection equation '''
        return scipy.integrate.quad(self.integrand, 0, 1, (r, self.spl_mass), epsrel=1e-3)[0]

    def rho(self, r):
        ''' Return 3d density rho(r) approximated by a spline '''
        logr   = numpy.log(r)
        result = self.spl_rho(numpy.maximum(self.logrmin, numpy.minimum(self.logrmax, logr))) \
            + self.derrmin * numpy.minimum(logr-self.logrmin, 0) \
            + self.derrmax * numpy.maximum(logr-self.logrmax, 0)
        return numpy.exp(result)

    def sigma_iso_integr(self, r):
        ''' Return isotropic velocity dispersion computed by integration of the deprojection equation '''
        return (scipy.integrate.quad(self.integrand, 0, 1, (r, self.spl_Ekin), epsrel=1e-3)[0] * \
            self.total_Ekin / self.rho_integr(r) )**0.5


##################### RESAMPLING OF ORIGINAL DATA TO FILL MISSING VALUES #################

def sample_z_position(R, sph_model):
    '''
    Sample the missing z-component of particle coordinates
    from the density distribution given by the spherical model.
    input argument 'R' contains the array of projected radii,
    and the output will contain the z-values assigned to them,
    and the weights of individual samples.
    '''
    print 'Assigning missing z-component of position'
    rho_max = sph_model.rho(R)*1.1
    R0      = numpy.maximum(2., R)
    result  = numpy.zeros_like(R)
    weights = sph_model.surface_density(R)
    indices = numpy.where(result==0)[0]   # index array initially containing all input points
    while len(indices)>0:   # rejection sampling
        t = numpy.random.uniform(-1, 1, size=len(indices))
        z = R0[indices] * t/(1-t*t)**0.5
        rho = sph_model.rho( (R[indices]**2 + z**2)**0.5 )
        rho_bar = rho / (1-t*t)**1.5 / rho_max[indices]
        overflows = numpy.where(rho_bar>1)[0]
        if(len(overflows)>0):
            print 'Overflows:', len(overflows)
            #numpy.hstack((R[overflows].reshape(-1.1), t[overflows].reshape(-1.1), rho_bar[overflows].reshape(-1.1)))
        assigned = numpy.where(numpy.random.uniform(size=len(indices)) < rho_bar)[0]
        result [indices[assigned]]  = z[assigned]
        weights[indices[assigned]] /= rho[assigned]
        indices = numpy.where(result==0)[0]  # find out the unassigned elements
    return result,weights

def sample_particle_realizations(particles):
    '''
    Split each input particle into num_samp samples, perturbing its velocity or
    assigning values for missing components of position/velocity.
    '''
    # duplicate the elements of the original particle array (each particle is expanded into num_samp identical samples)
    samples  = numpy.repeat(particles, num_samples, axis=0)
    nsamples = samples.shape[0]
    weights  = numpy.ones_like(nsamples)

    print 'Resample',particles.shape[0],'input particles into',nsamples,'internal samples'
    # compute maximum magnitude of l.o.s. velocity used in assigning missing velocity components for resampled particles
    vmax = numpy.amax(abs(particles[:,5]))
    zmax = numpy.sort(abs(particles[:,2]))[int(particles.shape[0]*0.98)]

    sph_model = SphericalModel(particles)
    numpy.random.seed(0)  # make it repeatable from run to run
    samples[:,2], weights = sample_z_position((samples[:,0]**2+samples[:,1]**2)**0.5, sph_model)
    vtrans_mag = vmax * numpy.random.uniform(0, 1, size=nsamples)**0.5
    vtrans_ang = numpy.random.uniform(0, 6.2832, size=nsamples)
    samples[:,3] = vtrans_mag * numpy.cos(vtrans_ang)
    samples[:,4] = vtrans_mag * numpy.sin(vtrans_ang)
    weights *= 3.1416*vmax**2 / num_samples   # volume of {z,vx,vy}-space per sample

    if vel_error != 0:    # add noise to the velocity components
        samples  += numpy.random.standard_normal(samples.shape) * \
            numpy.tile(numpy.array([0, 0, 0, vel_error, vel_error, vel_error]), nsamples).reshape(-1,6)

    return samples, weights


######################## MODEL-SEARCHER ####################################

def deterministic_search_fnc(params, obj):
    '''
    function to minimize using the deterministic algorithm (needs to be declared outside the class)
    '''
    loglike = obj.model_likelihood(params)
    if not numpy.isfinite(loglike):
        loglike = -100*len(obj.particles)   # replace infinity with a very large negative number
    return -loglike

def monte_carlo_search_fnc(params, obj):
    '''
    function to maximize using the monte carlo algorithm (needs to be declared outside the class)
    '''
    return obj.model_likelihood(params)

class ModelSearcher:
    '''
    Class that encompasses the computation of likelihood for the given parameters,
    and implements model-searching algorithms (deterministic and MCMC)
    '''
    def __init__(self, filename):
        self.filename  = filename
        self.model     = ModelParams(filename)
        self.particles = numpy.loadtxt(filename)[:,:6]
        if not full_phase_space_info and use_resampling:
            self.samples, self.weights = sample_particle_realizations(self.particles)
            #numpy.savetxt('samples.txt', numpy.hstack((self.samples, self.weights.reshape(-1,1))), fmt='%.7g')
        try:
            self.values = numpy.loadtxt(self.filename+".best")
            if self.values.ndim==1:
                self.values = self.values[:-1]
            else:
                self.values = self.values[:,:-1]
            print "Loaded from saved file: (nwalkers,nparams)=",self.values.shape
        except:
            self.values = None
            return

    def model_likelihood(self, params):
        '''
        Compute the likelihood of model (df+potential specified by scaled params)
        against the data (array of Nx6 position/velocity coordinates of tracer particles).
        This is the function to be maximized; if parameters are outside the allowed range, it returns -infinity
        '''
        prior = self.model.prior(params)
        potparams, dfparams = self.model.unscale(params)
        logfile = open(self.filename+".log", "a")
        print >>logfile, \
            "dens=%10.4g, Rscale=%6.4g, gamma=%8.4g, beta=%6.4g; " \
            "alpha=%8.4g, beta=%6.4g, ar=%8.4g, br=%7.4g, J0=%7.4g: " \
            % (potparams['densityNorm'], potparams['scaleRadius'], potparams['gamma'], potparams['beta'], \
            dfparams['alpha'], dfparams['beta'], dfparams['ar'], dfparams['br'], dfparams['j0']),
        if prior == -numpy.inf:
            print >>logfile, "Out of range"
            return prior
        try:
            # Compute log-likelihood of DF with given params against an array of actions
            pot     = agama.Potential(**potparams)
            df      = agama.DistributionFunction(**dfparams)
            norm    = df.totalMass()     # total mass of tracers given by their DF
            if full_phase_space_info:
                af      = agama.ActionFinder(pot)
                actions = af(self.particles)  # actions of tracer particles
                df_val  = df(actions) / norm  # values of DF for these actions
            elif use_resampling:  # have full phase space info for resampled input particles (missing components are filled in)
                af      = agama.ActionFinder(pot)
                actions = af(self.samples)    # actions of resampled tracer particles
                # compute values of DF for these actions, multiplied by sample weights
                df_val  = df(actions) / norm * self.weights
                # compute the weighted sum of likelihoods of all samples for a single particle,
                # replacing the improbable samples (with NaN as likelihood) with zeroes
                df_val  = numpy.sum(numpy.nan_to_num(df_val.reshape(-1, num_samples)), axis=1)
                '''
                df_vprj = agama.GalaxyModel(pot, df).projected_df( \
                    numpy.hstack((self.particles[:,0:2], self.particles[:,5].reshape(-1,1))), \
                    vz_error=vel_error) / norm
                numpy.savetxt("df.txt", numpy.hstack((self.particles[:,0:2], self.particles[:,5].reshape(-1,1), \
                    df_val.reshape(-1,1), df_vprj.reshape(-1,1))), fmt="%.7g")
                '''
            else:   # have only x,y,vz values, marginalize over missing components
                df_val = agama.GalaxyModel(pot, df).projected_df( \
                    numpy.hstack((self.particles[:,0:2], self.particles[:,5].reshape(-1,1))), \
                    vz_error=vel_error) / norm

            loglike = numpy.sum( numpy.log( df_val ) )
            if numpy.isnan(loglike): loglike = -numpy.inf
            loglike += prior
            print >>logfile, "LogL=%.8g" % loglike
            return loglike
        except ValueError as err:
            print >>logfile, "Exception ", err
            return -numpy.inf


    def deterministic_search(self, user_fnc=None):
        '''
        do a deterministic search to find the best-fit parameters of potential and distribution function.
        perform several iterations of search, to avoid getting stuck in a local minimum,
        until the log-likelihood ceases to improve
        '''
        if self.values is None:                   # just started
            self.values = self.model.init_values  # get the first guess from the model-scaling object
        elif self.values.ndim == 2:               # entire ensemble of values (after MCMC)
            self.values = self.values[0,:]        # leave only one set of values from the ensemble
        prevloglike = -deterministic_search_fnc(self.values, self)  # initial likelihood

        while True:
            print 'Starting deterministic search'
            result = scipy.optimize.minimize(deterministic_search_fnc, \
                self.values, args=(self,), method='Nelder-Mead', \
                options=dict(maxfev=nsteps_deterministic, disp=True))
            print 'result=', result.x, 'LogL=', result.fun,
            self.values = result.x
            loglike= -result.fun
            # store the latest best-fit parameters and their likelihood
            numpy.savetxt(self.filename+'.best', numpy.hstack((self.values, loglike)).reshape(1,-1), fmt='%.8g')
            if loglike - prevloglike < 1.0:
                print 'Converged'
                return
            else:
                print 'Improved log-likelihood by', loglike - prevloglike
            prevloglike = loglike

    def monte_carlo_search(self, user_fnc=None):
        '''
        Explore the parameter space around the best-fit values using the MCMC method
        initwalkers is the Nwalkers * Nparams array of initial parameters
        '''
        if self.values is None:   # first attempt a deterministic search
            self.deterministic_search()
        if self.values.ndim == 1:
            # initial coverage of parameter space (dispersion around the current best-fit values)
            nparams = len(self.values)
            ensemble = numpy.empty((nwalkers_mcmc, len(self.values)))
            for i in range(nwalkers_mcmc):
                while True:   # ensure that we initialize walkers with feasible values
                    walker = self.values + (numpy.random.randn(nparams)*initial_disp_mcmc if i>0 else 0)
                    prob   = monte_carlo_search_fnc(walker, self)
                    if numpy.isfinite(prob):
                        ensemble[i,:] = walker
                        break
                    print '*',
            self.values = ensemble
        else:
            # check that all walkers have finite likelihood
            prob = numpy.zeros((self.values.shape[0],1))
            for i in range(self.values.shape[0]):
                prob[i,0] = monte_carlo_search_fnc(self.values[i,:], self)
                if not numpy.isfinite(prob[i,0]):
                    print 'Invalid parameters for',i,'-th walker (likelihood is bogus)'
                else: print prob[i,0]
            try:
                oldchain = numpy.loadtxt(self.filename+".chain")
                prob     = oldchain[:, -1].reshape(self.values.shape[0], -1)
                oldchain = oldchain[:,:-1].reshape(self.values.shape[0], -1, self.values.shape[1])
            except:
                oldchain = self.values.reshape(self.values.shape[0],1,self.values.shape[1])
            if not user_fnc is None:
                user_fnc(oldchain, prob)

        nwalkers, nparams = self.values.shape
        sampler = emcee.EnsembleSampler(nwalkers, nparams, monte_carlo_search_fnc, args=(self,), threads=nthreads_mcmc)
        prevmaxloglike = None
        while True:  # run several passes until convergence
            print 'Starting MCMC'
            sampler.run_mcmc(self.values, nsteps_mcmc)
            # restart the next pass from the latest values in the Markov chain
            self.values = sampler.chain[:,-1,:]

            # store the latest best-fit parameters and their likelihood, and the entire chain
            numpy.savetxt(self.filename+'.best', \
                numpy.hstack((self.values, sampler.lnprobability[:,-1].reshape(-1,1))), fmt='%.8g')
            numpy.savetxt(self.filename+".chain", \
                numpy.hstack((sampler.chain.reshape(-1,nparams), sampler.lnprobability.reshape(-1,1))), fmt='%.8g')

            print "Acceptance fraction: ", numpy.mean(sampler.acceptance_fraction)  # should be in the range 0.2-0.5
            print "Autocorrelation time: ", sampler.acor  # should be considerably shorter than the total number of steps
            maxloglike = numpy.max(sampler.lnprobability[:,-nsteps_mcmc:])
            avgloglike = numpy.average(sampler.lnprobability[:,-nsteps_mcmc:])  # avg.log-likelihood during the pass
            avgparams  = [numpy.average(sampler.chain[:,-nsteps_mcmc:,i]) for i in range(nparams)]
            print "Max log-likelihood= %.8g, avg log-likelihood= %.8g" % (maxloglike, avgloglike)
            for i in range(nparams):
                sorted_values = numpy.sort(sampler.chain[:,-nsteps_mcmc:,i], axis=None)
                print "Parameter %20s  avg= %8.5g;  one-sigma range = (%8.5f, %8.5f), true value = %8.5f" \
                    % (self.model.labels[i], avgparams[i], \
                    sorted_values[int(len(sorted_values)*0.16)], sorted_values[int(len(sorted_values)*0.84)], \
                    self.model.true_values[i])
            if not user_fnc is None:
                user_fnc(sampler.chain, sampler.lnprobability)

            # check for convergence
            if not prevmaxloglike is None:
                if maxloglike-prevmaxloglike < 1.0 and abs(avgloglike-prevavgloglike) < 1.0:
                    print "Converged"
                    return
            prevmaxloglike = maxloglike
            prevavgloglike = avgloglike
            prevavgparams  = avgparams


###################################  DATA ANALYSIS  ########################################

def compute_df_moments(potparams, dfparams, rmin, rmax):
    '''
    Compute moments of distribution function (density and velocity dispersion);
    DF and potential are specified by scaled parameters (params)
    Return: a tuple of four arrays: radii, density, radial velocity dispersion and tangential v.d.
    '''
    radii = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), 25).reshape(-1,1)
    xyz   = numpy.hstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii)))
    pot   = agama.Potential(**potparams)
    df    = agama.DistributionFunction(**dfparams)
    norm  = df.totalMass()
    dens, mom = agama.GalaxyModel(pot, df).moments(xyz, dens=True, vel=False, vel2=True)
    return radii.reshape(-1), dens/norm, mom[:,0]**0.5, mom[:,1]**0.5

def compute_df_moments_projected(potparams, dfparams, rmin, rmax):
    '''
    Compute projected moments of distribution function (surface density and l.o.s. velocity dispersion);
    DF and potential are specified by scaled parameters (params)
    Return: a tuple of three arrays: radii, density, velocity dispersion
    '''
    radii = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), 25)
    pot   = agama.Potential(**potparams)
    df    = agama.DistributionFunction(**dfparams)
    norm  = df.totalMass()
    dens, veldisp = agama.GalaxyModel(pot, df).projectedMoments(radii)
    return radii, dens/norm, veldisp**0.5

def compute_dm_density(potparams, rmin, rmax):
    '''
    Compute density profile corresponding to the potential specified by potparams
    Return: a tuple of two arrays: radii, density
    '''
    pot   = agama.Potential(**potparams)
    radii = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax)).reshape(-1,1)
    xyz   = numpy.hstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii)))
    dens  = pot.density(xyz)
    return radii.reshape(-1), dens

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
    # select bin boundaries so that each bin contains Nptbin data points, or less points at the edges of radial interval
    indices = bin_indices(len(radii))
    hist_boundaries = sorted_radii[indices]
    sumnum,_ = numpy.histogram(radii, bins=hist_boundaries, weights=numpy.ones_like(radii))
    sumvelradsq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=velradsq)
    sumveltansq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=veltansq)
    binvol = 4*3.1416/3 * (hist_boundaries[1:]**3 - hist_boundaries[:-1]**3)
    density= sumnum/len(points[:,0]) / binvol
    sigmar = (sumvelradsq/sumnum)**0.5
    sigmat = (sumveltansq/sumnum)**0.5
    return hist_boundaries, density, sigmar, sigmat

def compute_orig_moments_projected(points):
    '''
    Compute the moments (surface density and line-of-sight velocity dispersion)
    of the original data points (tracer particles), binned in projected radius.
    Return: tuple of three arrays: radii, density, l.o.s. velocity dispersion,
    where the array of radii is one element longer than the other three arrays, and denotes the bin boundaries,
    and the other arrays contain the values in each bin.
    '''
    Radii = (points[:,0]**2 + points[:,1]**2) ** 0.5
    velsq = points[:,5]**2
    sorted_radii = numpy.sort(Radii)
    # select bin boundaries so that each bin contains Nptbin data points, or less points at the edges of radial interval
    indices = bin_indices(len(Radii))
    hist_boundaries = sorted_radii[indices]
    sumnum,_ = numpy.histogram(Radii, bins=hist_boundaries, weights=numpy.ones_like(Radii))
    sumvelsq,_ = numpy.histogram(Radii, bins=hist_boundaries, weights=velsq)
    binarea= 3.1416 * (hist_boundaries[1:]**2 - hist_boundaries[:-1]**2)
    density= sumnum/len(points[:,0]) / binarea
    sigma  = (sumvelsq/sumnum)**0.5
    return hist_boundaries, density, sigma

##############  USER FUNCTION FOR PLOTTING INTERMEDIATE RESULTS  ###############

class UserFnc:
    def __init__(self, model, particles, filename):
        self.model = model
        self.filename = filename
        self.ibins, self.idens, self.isigmar, self.isigmat = \
            compute_orig_moments(particles)   # get intrinsic profiles for the original data points
        self.pbins, self.pdens, self.psigma = \
            compute_orig_moments_projected(particles)  # get projected profiles for the original points
        self.deprojected = SphericalModel(particles)

    def __call__(self, chain, loglike):
        # evolution of MCMC chain parameters and likelihood
        self.plot_time_evol(chain, loglike)
        # distribution of posterior parameters and their correlations, for the potential only
        self.plot_distribution(chain[:, -nsteps_mcmc:, :self.model.num_pot_params], "_pot", \
            self.model.labels[:self.model.num_pot_params], self.model.true_values[:self.model.num_pot_params])
        # distribution of posterior parameters and their correlations, for all parameters
        self.plot_distribution(chain[:, -nsteps_mcmc:, :], "_all", self.model.labels, self.model.true_values)
        # density and velocity dispersion profiles (3d and projected)
        self.plot_profiles (chain[:,-1,:])

    def plot_profiles(self, params):
        '''
        plot radial profiles of velocity dispersion of tracer particles, their density,
        and DM density (corresponding to the potential),
        for an ensemble of parameters, together with the 'true' profile.
        '''
        rmin     = self.ibins[0]*0.5
        rmax     = self.ibins[-1]*2   # radial range to plot (where the data points lie)
        tdnorm   = 1e6   # multiplicator for tracer density (to bring it close to the range of DM density for plotting)
        tgamma   = self.model.true_tracer_params['gamma']
        trscale  = self.model.true_tracer_params['scaleRadius']
        fig,axes = pyplot.subplots(1, 2, figsize=(16,10))

        if not params is None:
            for i in range(len(params[:,0])):
                potparams, dfparams = self.model.unscale(params[i,:])
                radii,dens = compute_dm_density(potparams, rmin, rmax)
                axes[1].plot(radii, dens, color='k', alpha=0.5)    # DM density profile
                radii,dens,sigmar,sigmat = compute_df_moments(potparams, dfparams, rmin, rmax)
                axes[0].plot(radii, sigmar, color='r', alpha=0.5)  # velocity dispersions
                axes[0].plot(radii, sigmat, color='b', alpha=0.5)
                axes[1].plot(radii, dens * tdnorm, color='k', alpha=0.5)    # tracer density

        radii,dens = compute_dm_density(self.model.true_potential_params, rmin, rmax)
        axes[1].plot(radii, dens, color='r', lw=3, linestyle='--', label='DM density')

        # binned vel.disp. and density profile from input particles; emulating steps plot
        axes[0].plot(numpy.hstack(zip(self.ibins[:-1], self.ibins[1:])), numpy.hstack(zip(self.isigmar, self.isigmar)), \
            color='r', lw=1, label=r'$\sigma_r$')
        axes[0].plot(numpy.hstack(zip(self.ibins[:-1], self.ibins[1:])), numpy.hstack(zip(self.isigmat, self.isigmat)), \
            color='b', lw=1, label=r'$\sigma_t$')
        axes[1].plot(numpy.hstack(zip(self.ibins[:-1], self.ibins[1:])), numpy.hstack(zip(self.idens, self.idens)) * tdnorm, \
            color='g', lw=1, label=r'Tracer density, binned')

        # analytic density profile
        axes[1].plot(radii, tdnorm * (3-tgamma) / (4*3.1416*trscale**3) * \
            (radii/trscale)**(-tgamma) * (1 + (radii/trscale)**2) ** ((tgamma-5)/2), \
            color='g', lw=3, linestyle='--', label=r'Tracer density, analytic')
        # deprojected density profile
        axes[1].plot(radii, self.deprojected.rho(radii)*tdnorm, \
            linestyle='--', lw=3, color='cyan', label='Tracer density, deprojected')

        axes[0].set_xscale('log')
        axes[0].set_yscale('linear')
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(rmin, rmax)
        axes[0].set_xlabel('$r$')
        axes[0].set_ylabel(r'$\sigma_{r,t}$')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].legend(loc='lower left')
        axes[1].set_xlim(rmin, rmax)
        axes[1].set_xlabel('$r$')
        axes[1].set_ylabel(r'$\rho$')
        fig.tight_layout()
        pyplot.savefig(self.filename+"_profiles.png")

        # projected quantities
        rmin     = self.pbins[0]*0.5
        rmax     = self.pbins[-1]*2   # radial range to plot (where the data points lie)
        fig,axes = pyplot.subplots(1, 2, figsize=(16,10))

        if not params is None:
            for i in range(len(params[:,0])):
                potparams, dfparams = self.model.unscale(params[i,:])
                radii,dens,losvdisp = compute_df_moments_projected(potparams, dfparams, rmin, rmax)
                axes[0].plot(radii, losvdisp, color='k', alpha=0.5)       # l.o.s. velocity dispersion
                axes[1].plot(radii, dens * tdnorm, color='k', alpha=0.5)  # surface density
        else:
            radii = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), 25)

        # binned l.o.s.vel.disp. and surface density profile from input particles; emulating steps plot
        axes[0].plot(numpy.hstack(zip(self.pbins[:-1], self.pbins[1:])), numpy.hstack(zip(self.psigma, self.psigma)), \
            color='g', lw=1, label=r'$\sigma_{los}$,binned')
        axes[1].plot(numpy.hstack(zip(self.pbins[:-1], self.pbins[1:])), numpy.hstack(zip(self.pdens, self.pdens)) * tdnorm, \
            color='g', lw=1, label=r'Surface density,binned')

        # projected density and losvd profile from the smoothed model
        axes[0].plot(radii, self.deprojected.sigma_los(radii), \
            linestyle=':', lw=3, color='r', label=r'$\sigma_{los}$,smoothed')
        axes[1].plot(radii, self.deprojected.surface_density(radii)*tdnorm, \
            linestyle=':', lw=3, color='r', label=r'Surface density,smoothed')

        axes[0].set_xscale('log')
        axes[0].set_yscale('linear')
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(rmin, rmax)
        axes[0].set_xlabel('$r$')
        axes[0].set_ylabel(r'$\sigma_{los}$')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].legend(loc='lower left')
        axes[1].set_xlim(radii[0]*0.5, radii[-1]*2)
        axes[1].set_xlabel('$r$')
        axes[1].set_ylabel(r'$\Sigma$')
        fig.tight_layout()
        pyplot.savefig(self.filename+"_projected.png")

    def plot_time_evol(self, chain, loglike):
        '''
        Show the time evolution of parameters carried by the ensemble of walkers (time=number of MC steps)
        '''
        ndim = chain.shape[2]
        fig,axes = pyplot.subplots(ndim+1, 1, sharex=True, figsize=(20,15))
        for i in range(ndim):
            axes[i].plot(chain[:,:,i].T, color='k', alpha=0.5)
            axes[i].plot(numpy.ones_like(chain[0,:,i]) * self.model.true_values[i], color='r', linestyle='--', lw=2)
            axes[i].set_ylabel(self.model.labels[i])
        # last panel shows the evolution of log-likelihood for the ensemble of walkers
        axes[-1].plot(loglike.T, color='k', alpha=0.5)
        axes[-1].set_ylabel('log(L)')
        maxloglike = numpy.max(loglike)
        axes[-1].set_ylim(maxloglike-3*ndim, maxloglike)   # restrict the range of log-likelihood arount its maximum
        fig.tight_layout(h_pad=0.)
        pyplot.savefig(self.filename+"_chain.png")

    def plot_distribution(self, chain, suffix, labels, truevalues):
        '''
        Show the posterior distribution of parameters
        '''
        triangle.corner(chain.reshape((-1, chain.shape[2])), quantiles=[0.16, 0.5, 0.84], labels=labels, truths=truevalues)
        pyplot.savefig(self.filename+"_posterior"+suffix+".png")


################  MAIN PROGRAM  ##################

agama.set_units(mass=1, length=1, velocity=1)
basefilename = os.getcwd().split('/')[-1]  # get the directory name which is the same as the first part of the filename
model_searcher = ModelSearcher(basefilename+"_1000_0.dat")
user_fnc = UserFnc(model_searcher.model, model_searcher.particles, model_searcher.filename)
model_searcher.monte_carlo_search(user_fnc)
