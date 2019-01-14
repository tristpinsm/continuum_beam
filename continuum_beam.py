import healpy
import numpy as np
from ch_util import ephemeris as ephem
from caput.time import unix_to_skyfield_time
from drift.telescope import cylbeam


class ModelVis(object):
    c = 2.99792458e2
    synch_ind = -0.75

    def __init__(self, fname="./lambda_haslam408_nofilt.fits", freq=408., smooth=True,
                 sky_map=None, max_za=90., harm_basis=False):
        # observed frequency
        self.freq = freq
        self.wl = self.c / freq

        self.synch_ind = -0.75

        # load sky map
        if sky_map is not None:
            self.basemap = sky_map
        else:
            self.basemap = healpy.fitsfunc.read_map(fname)
        self.nside = int((self.basemap.shape[0]/12)**0.5)

        # smooth map using CHIME EW primary beam
        if smooth:
            self.smoothmap = healpy.sphtfunc.smoothing(self.basemap, sigma=self._res())
        else:
            self.smoothmap = self.basemap

        # scale map with average synchrotron index
        #self.smoothmap *= (freq / 408.)**self.synch_ind

        # get an observer at CHIME arguments
        self.obs = ephem.chime_observer().skyfield_obs()

        # set fitting constraints
        self.max_za = max_za
        self.harm_basis = harm_basis

        self.xtalk = None
        self.chi2 = None

    def set_baselines(self, ns_baselines, ew_baselines=None):
        self.ns_baselines = ns_baselines
        self.ew_baselines = ew_baselines

    def get_vis(self, times, vis, n, max_za=90.,
                model_beam=None, skip_basis=False):
        # use this to check/visualize model
        if not skip_basis:
            self._gen_basis(times, vis, n, max_za)
        if model_beam is None:
            beam = 1.
        else:
            beam = model_beam
        return np.sum(self._basis * beam, axis=2)

    def get_chi2(self):
        if self.chi2 is not None:
            return self.chi2
        else:
            raise Exception("No chi^2 is available until a fit is performed.")

    def fit_beam(self, times, vis, weight, n, max_za=90., set_beam=None,
                 rcond=None, xtalk_iter=1, chain_len=500, resume=False):

        # Initialize xtalk
        if resume and self.xtalk is not None:
            xtalk = self.xtalk
            self.total_iter += xtalk_iter
        else:
            # for first iteration remove nothing
            xtalk = np.zeros(vis.shape[0])
            self.total_iter = xtalk_iter
            # generate model basis
            self._gen_basis(times, vis, n, max_za)

        xtalk_step = 0.1 * np.mean(np.abs(vis), axis=-1)

        # least squares beam fit, crosstalk iterations
        for i in range(xtalk_iter):
            print("\rCrosstalk iteration {:d}/{:d}...".format(i+1, xtalk_iter)),
            # least squares solution
            if i == 0 and set_beam is not None:
                self.beam_sol = set_beam
            else:
                self._lls_beam_sol(vis, weight, n, xtalk, rcond)
            # update chain step size
            if i > 0:
                xtalk_step = np.std(np.abs(xchain['xtalk']), axis=0)
            # update cross-talk estimate using fit result
            xchain = self.xtalk_chain(times, vis, weight, n, max_za, xtalk, chain_len,
                                      skip_basis=True, step=xtalk_step)
            #resid = vis - self.get_vis(times, vis, n, max_za, self.beam_sol)
            #mad_resid = np.median(np.abs(resid - np.median(resid, axis=1)[:,np.newaxis]), axis=1)
            #xtalk = np.mean(resid * (np.abs(resid) < 3 * mad_resid[:,np.newaxis]), axis=1)
            #xtalk = np.sum(resid * weight, axis=-1) / np.sum(weight, axis=-1)
            #xtalk = np.mean(resid, axis=-1)
            xtalk = np.mean(xchain['xtalk'], axis=0)
        print("\nDone {:d} iterations.".format(self.total_iter))
        self.xtalk = xtalk
        self.xchain = xchain
        # calculate chi^2
        self.chi2 = np.sum(np.abs((np.dot(self._basis, self.beam_sol) - vis))**2 * weight)
        # calculate covariance
        U, S, V = np.linalg.svd(self.M, full_matrices=False)
        self.cov = np.dot(V, np.dot(np.diag(1./S), V.T))
        return self.beam_sol

    def xtalk_chain(self, times, vis, weight, n, max_za=90., xtalk_sample=None,
                    num_steps=1000, step=None, skip_basis=False):
        # starting parameters
        if xtalk_sample is None:
            # TODO: consider starting with vis average
            xtalk_sample = np.zeros(vis.shape[0])
        if not skip_basis:
            self._gen_basis(times, vis, n, max_za)
        model_vis = self.get_vis(times, vis, n, max_za, skip_basis=True,
                                 model_beam=self.beam_sol)
        if step is None:
            step = 0.1 * np.mean(np.abs(vis), axis=-1)

        # likelihood calculation
        def chi2_lnlkhd(x_sample):
            return - np.sum(np.abs(vis - x_sample[:,np.newaxis] - model_vis)**2 * weight)

        # proposal distribution
        def chain_step(sample):
            real_part = np.random.normal(scale=step / np.sqrt(2), size=sample.shape)
            imag_part = 1j * np.random.normal(scale=step / np.sqrt(2), size=sample.shape)
            return sample + real_part + imag_part

        # likelihood for starting params
        last_lnlkhd = chi2_lnlkhd(xtalk_sample)

        # initialize chain
        chain = {'xtalk': np.zeros((num_steps, vis.shape[0]), dtype=np.complex64),
                 'accept': np.zeros(num_steps, dtype=np.int8),
                 'lnlkhd': np.zeros(num_steps)}
        for i in range(num_steps):
            # draw a new cross-talk sample from conditional lkhd
            xtalk_sample_new = chain_step(xtalk_sample)
            new_lnlkhd = chi2_lnlkhd(xtalk_sample_new)
            if (new_lnlkhd >= last_lnlkhd or
                    np.exp(new_lnlkhd - last_lnlkhd) > np.random.uniform()):
                # accept sample
                xtalk_sample = xtalk_sample_new
                last_lnlkhd = new_lnlkhd
                chain['accept'][i] = 1
            chain['lnlkhd'][i] = new_lnlkhd

            chain['xtalk'][i] = xtalk_sample

        return chain

    def compute_lkhd_chain(self, times, vis, weight, n, max_za=90., rcond=None,
                           num_steps=1000, step_tweaks=(0.1, 0.1), skip_basis=False):
        # starting parameters
        xtalk_sample = np.zeros(vis.shape[0])
        if not skip_basis:
            self._gen_basis(times, vis, n, max_za)
        self._lls_beam_sol(vis, weight, n)
        beam_sample = self.beam_sol
        # make a choice for proposal steps
        beam_step = step_tweaks[1] * np.mean(np.abs(beam_sample))
        xtalk_step = step_tweaks[0] * np.mean(np.abs(vis), axis=-1)

        # likelihood calculation
        def chi2_lnlkhd(x_sample, b_sample):
            model_vis = self.get_vis(times, vis, n, max_za, skip_basis=True,
                                     model_beam=b_sample)
            return - np.sum(np.abs(vis - x_sample[:,np.newaxis] - model_vis)**2 * weight)

        # proposal distribution
        def chain_step(sample, sig=1., imag=False):
            base = 1j if imag else 1.
            return sample + base * np.random.normal(scale=sig, size=sample.shape)

        # likelihood for starting params
        last_lnlkhd = chi2_lnlkhd(xtalk_sample, beam_sample)

        # initialize chain
        chain = {'beam': np.zeros((num_steps, n)),
                 'xtalk': np.zeros((num_steps, vis.shape[0]), dtype=np.complex64),
                 'accept': np.zeros((num_steps, 2), dtype=np.int8),
                 'lnlkhd': np.zeros((num_steps, 2))}
        for i in range(num_steps):
            if i % 10 == 0:
                print("\rStep {:d}/{:d}...".format(i, num_steps)),
            # draw a new cross-talk sample from conditional lkhd
            xtalk_sample_new = chain_step(xtalk_sample, xtalk_step / np.sqrt(2))
            xtalk_sample_new = chain_step(xtalk_sample_new, xtalk_step / np.sqrt(2), imag=True)
            new_lnlkhd = chi2_lnlkhd(xtalk_sample_new, beam_sample)
            if (new_lnlkhd >= last_lnlkhd or
                    np.exp(new_lnlkhd - last_lnlkhd) > np.random.uniform()):
                # accept sample
                xtalk_sample = xtalk_sample_new
                last_lnlkhd = new_lnlkhd
                chain['accept'][i,0] = 1
            chain['lnlkhd'][i,0] = new_lnlkhd

            # draw and evaluate beam sample
            beam_sample_new = chain_step(beam_sample, beam_step)
            #self._lls_beam_sol(vis, weight, n, xtalk=xtalk_sample)
            #beam_sample_new = self.beam_sol
            new_lnlkhd = chi2_lnlkhd(xtalk_sample, beam_sample_new)
            if (new_lnlkhd >= last_lnlkhd or
                    np.exp(new_lnlkhd - last_lnlkhd) > np.random.uniform()):
                # accept sample
                beam_sample = beam_sample_new
                last_lnlkhd = new_lnlkhd
                chain['accept'][i,1] = 1
            chain['lnlkhd'][i,1] = new_lnlkhd

            chain['beam'][i] = beam_sample
            chain['xtalk'][i] = xtalk_sample

        self.chain = chain

    def _lls_beam_sol(self, vis, weight, n, xtalk=0, rcond=None):
        # construct least squares equation
        # take the real part since we omit the lower half of the vis matrix
        M = np.zeros((n, n), dtype=np.float64)
        v = np.zeros((n,), dtype=np.float64)
        for t in range(vis.shape[1]):
            M += np.dot(self._basis[:,t,:].T.conj() * weight[:,t],
                        self._basis[:,t,:]).real
            v += np.dot(((vis[:,t] - xtalk) * weight[:,t]).T,
                        self._basis.conj()[:,t,:]).real
        # invert to solve
        if rcond is None:
            self.beam_sol = np.dot(np.linalg.inv(M), v)
        else:
            self.beam_sol = np.dot(np.linalg.pinv(M, rcond=rcond), v)
        # save intermediate products for debugging
        self.M = M
        self.v = v

    def _gen_basis(self, times, vis, n, max_za=90.):

        # make za axis
        max_sinza = np.sin(np.radians(max_za))
        if self.harm_basis:
            sinza = np.linspace(-max_sinza, max_sinza, 2*int(2*np.radians(max_za)/np.pi * n))
        else:
            sinza = np.linspace(-max_sinza, max_sinza, n)
        za = np.arcsin(sinza)
        az = np.zeros_like(za)
        az[:az.shape[0]/2] = 180.
        alt = 90. - np.cos(np.radians(az)) * np.degrees(za)
        self.za = za
        self.sinza = sinza
        self._basis = np.zeros((vis.shape[0], vis.shape[1], n), dtype=np.complex64)
        # make EW grid to integrate over
        phi = np.linspace(-2*self._res(), 2*self._res(), 20)
        z, p = np.meshgrid(za, phi)
        self.z, self.p = z, p
        alt, az = tel2azalt(z, p)
        # model the EW beam as a sinc
        ew_beam = cylbeam.fraunhofer_cylinder(lambda x: np.ones_like(x), 20.)
        ew_beam = ew_beam(p)**2
        self.ew_beam = ew_beam
        # calculate phases within beam
        phases = np.exp(1j * self._fringe_phase(z, p))
        for i, t in enumerate(times):
            sf_t = unix_to_skyfield_time(t)
            pix = np.zeros((alt.shape[0], alt.shape[1]), dtype=int)
            for j in range(alt.shape[0]):
                for k in range(alt.shape[1]):
                    pos = self.obs.at(sf_t).from_altaz(az_degrees=np.degrees(az[j,k]),
                                                       alt_degrees=np.degrees(alt[j,k]))
                    gallat, gallon = pos.galactic_latlon()[:2]
                    pix[j,k] = healpy.ang2pix(self.nside, gallon.degrees, gallat.degrees, lonlat=True)
            if len(phases.shape) > 2:
                basis = np.sum(self.smoothmap[pix] * ew_beam * phases, axis=1)
            else:
                basis = self.smoothmap[pix] * ew_beam * phases
            if self.harm_basis:
                # convolve NS slice with basis functions
                basis = np.dot(basis, np.sin(np.arange(1, n+1)[np.newaxis,:] * (za[:,np.newaxis] + np.pi/2)))
            self._basis[:,i,:] = basis

    def _res(self):
        # match FWHM of sinc for 20m aperture
        return self.wl / 20. / 1.95

    def _fringe_phase(self, za, phi=None):
        phases = self.ns_baselines[np.newaxis,:] * np.sin(za)[...,np.newaxis]
        if phi is not None:
            phases += self.ew_baselines[np.newaxis,...] * (np.sin(phi) * np.cos(za))[...,np.newaxis]
        phases = np.moveaxis(phases, -1, 0)
        return 2 * np.pi / self.wl * phases


def altaz2tel(alt, az, deg=False, reverse=False):
    if deg:
        alt, az = np.radians(alt), np.radians(az)
    az = az % (2 * np.pi)

    # rotate so that origin is in the S
    az += np.pi

    # calculate new coordinates
    theta = np.arctan(- (np.cos(az) * np.cos(alt)) /
                      np.sqrt(np.sin(alt)**2 + np.sin(az)**2 * np.cos(alt)**2))
    phi = np.arctan(- np.sin(az) / np.tan(alt))
    if reverse:
        phi[np.sin(alt) < 0] = np.pi - phi[np.sin(alt) < 0]
        #theta *= -1
    else:
        phi[np.sin(alt) < 0] = np.pi - phi[np.sin(alt) < 0]

    if reverse:
        # azimuth goes clockwise
        phi = (2*np.pi - phi) % (2*np.pi)
        # undo rotation
        #phi = (phi + np.pi) % (2*np.pi)

    # special case at origin
    zero_case = np.logical_and(np.isclose(az, 0.), np.isclose(alt, 0.))
    phi[zero_case] = 0.
    theta[zero_case] = np.pi/2.

    if deg:
        theta, phi = np.degrees(theta), np.degrees(phi)
    return theta, phi


def tel2azalt(theta, phi, deg=False):
    return altaz2tel(theta, phi, reverse=True, deg=deg)
