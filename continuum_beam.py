import healpy
import numpy as np
from ch_util import ephemeris as ephem
from caput.time import unix_to_skyfield_time
#import pyximport; pyximport.install(reload_support=True)
from mat_prod import outer_sum


class ModelVis(object):
    c = 2.99792458e2
    synch_ind = -0.75

    def __init__(self, fname="./lambda_haslam408_nofilt.fits", freq=408.):
        # observed frequency
        self.freq = freq
        self.wl = self.c / freq

        self.synch_ind = -0.75

        # smooth map using CHIME EW primary beam
        self.basemap = healpy.fitsfunc.read_map(fname)
        self.nside = int((self.basemap.shape[0]/12)**0.5)
        self.smoothmap = healpy.sphtfunc.smoothing(self.basemap, sigma=self._res())
        #self.smoothmap = self.basemap

        # scale map with average synchrotron index
        #self.smoothmap *= (freq / 408.)**self.synch_ind

        # get an observer at CHIME arguments
        self.obs = ephem.chime_observer().skyfield_obs()

        self.xtalk = None
        self.chi2 = None

    def set_baselines(self, baselines=None):
        if type(baselines) is np.ndarray:
            self.ns_baselines = baselines
        elif type(baselines) is int or type(baselines) is float:
            # generate them
            pass

    def get_vis(self, times, vis, n, max_za=90., model_beam=None):
        # use this to check/visualize model
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

    def fit_beam(self, times, vis, weight, n, max_za=90., rcond=None,
                 xtalk_iter=1, resume=False, t_stride=1):
        if resume and self.xtalk is not None:
            xtalk = self.xtalk
            self.total_iter += xtalk_iter
        else:
            # for first iteration remove nothing
            xtalk = np.zeros(vis.shape[0])
            self.total_iter = xtalk_iter

        if times.shape[0] % t_stride != 0:
            raise Exception("t_stride must divide time axis exactly.")

        for i in range(xtalk_iter):
            print("\rCrosstalk iteration {:d}/{:d}...".format(i, xtalk_iter)),
            # generate model basis
            self._gen_basis(times, vis, n, max_za)
            # construct least squares equation
            # take the real part since we omit the lower half of the vis matrix
            M = np.zeros((n, n), dtype=np.float64)
            v = np.zeros((n,), dtype=np.float64)
            if t_stride > 1:
                wgt_view = weight.reshape(t_stride * weight.shape[0],
                                          weight.shape[1] / t_stride)
                vis_view = vis.reshape(t_stride * vis.shape[0],
                                       vis.shape[1] / t_stride)
                basis = self._basis.reshape(t_stride * vis.shape[0],
                                            vis.shape[1] / t_stride, n)
                xtalk_view = np.concatenate([xtalk] * t_stride)
            else:
                wgt_view = weight
                vis_view = vis
                xtalk_view = xtalk
                basis = self._basis
            for t in range(vis_view.shape[1]):
                M += np.dot(basis[:,t,:].T.conj() * wgt_view[:,t],
                            basis[:,t,:]).real
                v += np.dot(((vis_view[:,t] - xtalk_view) * wgt_view[:,t]).T,
                            basis.conj()[:,t,:]).real
            # normalize to order unity
            #norm = np.median(np.abs(v))
            #v /= norm
            #M /= norm
            # invert to solve
            if rcond is None:
                self.beam_sol = np.dot(np.linalg.inv(M), v)
            else:
                self.beam_sol = np.dot(np.linalg.pinv(M, rcond=rcond), v)
            # update cross-talk estimate using fit result
            xtalk = np.mean(
                    vis - self.get_vis(times, vis, n, max_za, self.beam_sol),
                    axis=-1
            )
            del xtalk_view, vis_view, wgt_view, basis
        print("\nDone {:d} iterations.".format(self.total_iter))
        # save intermediate products for debugging
        self.M = M
        self.v = v
        self.xtalk = xtalk
        # calculate chi^2
        self.chi2 = np.sum(np.abs((np.dot(self._basis, self.beam_sol) - vis))**2 * weight)
        # calculate covariance
        U, S, V = np.linalg.svd(self.M, full_matrices=False)
        self.cov = np.dot(V, np.dot(np.diag(1./S), V.T))
        return self.beam_sol

    def _gen_basis(self, times, vis, n, max_za=90.):
        # evaluate Haslam map at n declinations and all times
        max_sinza = np.sin(np.radians(max_za))
        sinza = np.linspace(-max_sinza, max_sinza, n)
        za = np.arcsin(sinza)
        az = np.zeros_like(za)
        az[:n/2] = 180.
        alt = 90. - np.cos(np.radians(az)) * np.degrees(za)
        self.za = za
        self.sinza = sinza
        self._basis = np.zeros((vis.shape[0], vis.shape[1], za.shape[0]), dtype=np.complex64)
        phases = np.exp(1j * self._fringe_phase(sinza))
        for i, t in enumerate(times):
            sf_t = unix_to_skyfield_time(t)
            pos = self.obs.at(sf_t).from_altaz(az_degrees=az, alt_degrees=alt)
            gallat, gallon = pos.galactic_latlon()[:2]
            pix = healpy.ang2pix(self.nside, gallon.degrees, gallat.degrees, lonlat=True)
            self._basis[:,i,:] = self.smoothmap[pix] * phases

    def _res(self):
        # match FWHM of sinc for 20m aperture
        return self.wl / 20. / 1.95

    def _fringe_phase(self, sinza):
        return 2 * np.pi * self.ns_baselines[:,np.newaxis] / self.wl * sinza[np.newaxis,:]
