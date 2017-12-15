#!/usr/bin/python

import numpy as np
from cosmolopy.distance import diff_comoving_volume, luminosity_distance
from cosmolopy.distance import comoving_volume, set_omega_k_0
import scipy
from scipy.stats import lognorm
import time
from scipy.interpolate import UnivariateSpline
import cPickle
import argparse


# StarFormationHistory from Hopkins & Beacom 2006, unit = M_sun/yr/Mpc^3
@np.vectorize
def HopkinsBeacom2006StarFormationRate(z):
    x = np.log10(1.+z)
    if x < 0.30963:
        return np.power(10, 3.28*x-1.82)
    if (x >= 0.30963)and(x < 0.73878):
        return np.power(10, -0.26*x-0.724)
    if x >= 0.73878:
        return np.power(10, -8.0*x+4.99)


# Conversion factor b calculates the energy integral
def calc_conversion_factor(z, Gamma, upper_E=1e7, lower_E=1e4):
    u_lim = upper_E/(1+z)   # upper IceCube range
    l_lim = lower_E/(1+z)   # lower IceCube range

    if Gamma != 2.0:
        exponent = (2-Gamma)
        nenner = 1/exponent*(u_lim**exponent-l_lim**exponent)

    else:
        nenner = (np.log(u_lim)-np.log(l_lim))

    return (1/(1e5)**(Gamma-2))*1/(nenner)


# Use local source Density to deduce total number of sources in the universe
def tot_num_src(redshift_evolution, cosmology, zmax, density):
    integrand = lambda z: redshift_evolution(z) * \
        diff_comoving_volume(z, **cosmology)
    norm = 4 * np.pi * scipy.integrate.quad(integrand, 0, zmax)[0]
    area = 4 * np.pi * scipy.integrate.quad(integrand, 0, 0.01)[0]
    vlocal = comoving_volume(0.01, **cosmology)
    Ntotal = density * vlocal / (area/norm)
    return Ntotal, norm


# Calculate the number of sources in volume and luminosity interval
def calc_dN(LF, logL0, z0, logLbinwidth, zbinwidth,
            N_tot, int_norm, cosmology):
    LF_val = LF(z0, logL0)
    return (4*np.pi*LF_val*diff_comoving_volume(z0, **cosmology)/int_norm) * \
        logLbinwidth*zbinwidth*N_tot


# Physics Settings
def calc_pdf(density=1e-7, L_nu=1e50, sigma=1, gamma=2.19,
             logMu_range=[-10, 6], N_Mu_bins=200,
             z_limits=[0.04, 10.], nzbins=120,
             Lum_limits=[1e45, 1e54], nLbins=120,
             flux_to_mu=10763342917.859608):
    """
    Parameter:
        - density in 1/Mpc^3
        - L_nu in erg/yr
        - sigma in dex
        - gamma
        - flux_to_mu

    Integration Parameters
        - logMu_range = [-10,6]      # expected range in log mu,
        - N_Mu_bins = 200            # number of bins for log nu histogram
        - z_limits = [0.04, 10.]     # Redshift limits
        - nzbins = 120               # number of z bins
        - Lum_limits = [1e45,1e54]   # Luminosity limits
        - nLbins = 120               # number of logLuminosity bins
    """
    # Conversion Factors
    Mpc_to_cm = 3.086e+24
    erg_to_GeV = 624.151
    year2sec = 365*24*3600

    cosmology = {'omega_M_0': 0.308, 'omega_lambda_0': 0.692, 'h': 0.678}
    cosmology = set_omega_k_0(cosmology)  # Flat universe

    ### Define the Redshift and Luminosity Evolution
    redshift_evolution = lambda z: HopkinsBeacom2006StarFormationRate(z)
    LF = lambda z, logL: redshift_evolution(z)*np.log(10)*10**logL * \
        lognorm.pdf(10**logL, np.log(10)*sigma,
                    scale=L_nu*np.exp(-0.5*(np.log(10)*sigma)**2))

    N_tot, int_norm = tot_num_src(redshift_evolution, cosmology,
                                  z_limits[-1], density)
    print "Total number of sources {:.0f} (All-Sky)".format(N_tot)

    # Setup Arrays
    logMu_array = np.linspace(logMu_range[0], logMu_range[1], N_Mu_bins)
    Flux_from_fixed_z = []

    zs = np.linspace(z_limits[0], z_limits[1], nzbins)
    deltaz = (float(z_limits[1])-float(z_limits[0]))/nzbins

    Ls = np.linspace(np.log10(Lum_limits[0]), np.log10(Lum_limits[1]), nLbins)
    deltaL = (np.log10(Lum_limits[1])-np.log10(Lum_limits[0]))/nLbins

    # Integration
    t0 = time.time()
    Count_array = np.zeros(N_Mu_bins)
    muError = []
    tot_bins = nLbins * nzbins
    print('Starting Integration...Going to evaluate {} bins'.format(tot_bins))
    N_sum = 0

    Flux_from_fixed_z.append([])
    print "-"*20

    # Loop over redshift bins
    for z_count, z in enumerate(zs):
        # Conversion Factor for given z
        bz = calc_conversion_factor(z, gamma)
        dlz = luminosity_distance(z, **cosmology)
        tot_flux_from_z = 0.

        # Loop over Luminosity bins
        for l_count, lum in enumerate(Ls):
                run_id = z_count*nLbins+l_count
                if run_id % (tot_bins/10) == 0.:
                    print "{}%".format(100*run_id/tot_bins)
                # Number of Sources in
                dN = calc_dN(LF, lum, z, deltaL, deltaz,
                             N_tot, int_norm, cosmology)
                N_sum += dN

                #Flux to Source Strength
                logmu = np.log10(flux_to_mu * erg_to_GeV*10**lum/year2sec /
                                 (4*np.pi*(Mpc_to_cm*dlz)**2)*bz)

                # Add dN to Histogram
                if logmu < logMu_range[1] and logmu > logMu_range[0]:
                    tot_flux_from_z += dN*10**logmu
                    idx = int((logmu-logMu_range[0])*N_Mu_bins /
                              (logMu_range[1]-logMu_range[0]))
                    Count_array[idx] += dN
                else:
                    muError.append(logmu)

        Flux_from_fixed_z.append(tot_flux_from_z)

    print "Number of Mu out of Range: {}".format(len(muError))
    print "Num Sou {}".format(N_sum)
    t1 = time.time()

    print "-"*20
    print "\n Time needed for {}x{} bins: {}s".format(nzbins,
                                                      nLbins,
                                                      int(t1 - t0))

    return logMu_array, Count_array, zs, Flux_from_fixed_z


class SourceCountDistribution(object):
    def __init__(self, **kwargs):
        seed = kwargs.pop("seed", None)
        self.rng = np.random.RandomState(seed)
        if "file" in kwargs.keys():
            self.load_from_file(kwargs["file"])
        elif "log10mu" in kwargs.keys() and \
             "dNdlog10mu" in kwargs.keys() and \
             "density" in kwargs.keys():
            self.generate_from_histogram(kwargs["log10mu"],
                                         kwargs["dNdlog10mu"],
                                         kwargs["density"])
        else:
            raise NotImplementedError("""You have to provide either,
                'log10mu', 'dNdlog10mu', 'density' or 'file'""")

    def generate_from_histogram(self, log10mu, dNdlog10mu, density):
        mask = np.diff(np.cumsum(dNdlog10mu)) > 0
        self.min_log10mu = np.min(log10mu[1:][mask])
        self.max_log10mu = np.max(log10mu[1:][mask])
        self.N_source_default_density = np.sum(dNdlog10mu)
        self.default_density = density
        self.sf_spline = UnivariateSpline(np.cumsum(dNdlog10mu)[1:][mask] /
                                          np.sum(dNdlog10mu),
                                          log10mu[1:][mask], k=1, s=0)

    def load_from_file(self, file_path):
        with open(file_path, "r") as open_file:
            content = cPickle.load(open_file)
        self.min_log10mu = content["min_log10mu"]
        self.max_log10mu = content["max_log10mu"]
        self.N_source_default_density = content["N_source_default_density"]
        self.default_density = content["default_density"]
        self.sf_spline = content["sf_spline"]

    def save_to_file(self, file_path):
        content = {}
        content["min_log10mu"] = self.min_log10mu
        content["max_log10mu"] = self.max_log10mu
        content["N_source_default_density"] = self.N_source_default_density
        content["default_density"] = self.default_density
        content["sf_spline"] = self.sf_spline

        with open(file_path, "w") as open_file:
            cPickle.dump(content, open_file, protocol=2)

    def random(self, density=None):
        if density is None:
            density = self.default_density
        N_sources = int(density/self.default_density *
                        self.N_source_default_density)
        return self.sf_spline(self.rng.uniform(low=0.0, high=1.0,
                              size=N_sources))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='filename',
                        default='SourceCount.pickle',
                        help='Output filename')
    parser.add_argument('-d', action='store', dest='density', type=float,
                        default=1e-7,
                        help='Local neutrino source density [1/Mpc^3]')
    parser.add_argument("--L", action="store",
                        dest="luminosity", type=float, default=1e50,
                        help="Set luminosity for each source, unit erg/yr")
    parser.add_argument("--index", action="store", dest='index',
                        type=float, default=2.0,
                        help="Astrophysical neutrino spectral index")
    parser.add_argument("--sigma", action="store",
                        dest="sigma", type=float, default=1.0,
                        help="Width of a log normal function in dex")

    # Not yet implemented
    # parser.add_argument("--LF",action="store", dest="LF",default="SC",
    #                     help="Luminosity function, SC for standard candles,
    #                           LG for lognormal, PL for powerlaw")
    # parser.add_argument("--evolution", action="store",
    #                    dest="Evolution", default='HB2006SFR',
    #                    help="Source evolution options:  HB2006SFR (default),
    #                          NoEvolution")

    args = parser.parse_args()

    logMu_array, Count_array, zs, Flux_from_fixed_z = calc_pdf(
        density=args.density,
        L_nu=args.luminosity,
        sigma=args.sigma,
        gamma=args.index,
        flux_to_mu=1,
        logMu_range=[-20, -6],
        N_Mu_bins=2000,
        z_limits=[0.04, 10.],  # lower bound in Firesong 0.0005
        nzbins=1200,
        Lum_limits=[args.luminosity*1e-5, args.luminosity*1e5],
        nLbins=1200)

    dist = SourceCountDistribution(log10mu=logMu_array,
                                   dNdlog10mu=np.array(Count_array),
                                   density=args.density)
    dist.save(args.filename)
