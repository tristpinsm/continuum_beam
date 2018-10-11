import healpy
import numpy as np
from ch_util import ephemeris as ephem
from caput.time import unix_to_skyfield_time

class ModelVis(object):
    c = 2.99792458e2

    def __init__(self, fname="./lambda_haslam408_nofilt.fits", freq=408.):
        # observed frequency
        self.freq = freq
        self.wl = self.c / freq

        # smooth map using CHIME EW primary beam
        self.basemap = healpy.fitsfunc.read_map(fname)
        self.nside = int((self.basemap.shape[0]/12)**0.5)
        self.smoothmap = healpy.sphtfunc.smoothing(self.basemap, sigma=self._res())

        # get an observer at CHIME
        self.obs = ephem.chime_observer().skyfield_obs()

    def set_baselines(self, baselines=None):
        if type(baselines) is np.ndarray:
            self.ns_baselines = baselines
        elif type(baselines) is int or type(baselines) is float:
            # generate them
            pass

    def get_vis(self, times, vis, n, max_za=90.):
        # use this to check/visualize model
        self._gen_basis(times, vis, n, max_za)
        return np.sum(self._basis, axis=2)


    def fit_beam(self, times, vis, weight, n, max_za=90., return_basis=False):
        # generate model basis
        self._gen_basis(times, vis, n, max_za)
        # construct least squares equation
        # take the real part since we omit the lower half of the vis matrix
        M = np.sum(np.matmul(self._basis[:,:,:,np.newaxis], self._basis[:,:,np.newaxis,:]).real
                   * weight[:,:,np.newaxis,np.newaxis], axis=(0,1))
        v = np.sum(((vis * weight)[:,:,np.newaxis] * self._basis).real, axis=(0,1))
        # return solution
        return np.dot(np.linalg.inv(M), v)

    def _gen_basis(self, times, vis, n, max_za=90.):
        # evaluate Haslam map at n declinations and all times
        za = np.linspace(-max_za, max_za, n)
        az = np.zeros_like(za)
        az[:n/2] = 180.
        alt = 90. - np.cos(np.radians(az)) * za
        self.za = za
        self._basis = np.zeros((vis.shape[0], vis.shape[1], za.shape[0]), dtype=np.complex64)
        phases = np.exp(1j * self._fringe_phase(za))
        for i, t in enumerate(times):
            sf_t = unix_to_skyfield_time(t)
            pos = self.obs.at(sf_t).from_altaz(az_degrees=az, alt_degrees=alt)
            gallat, gallon = pos.galactic_latlon()[:2]
            pix = healpy.ang2pix(self.nside, gallon.degrees, gallat.degrees, lonlat=True)
            self._basis[:,i,:] = self.smoothmap[pix] * phases

    def _res(self):
        # match FWHM of sinc for 20m aperture
        return self.wl / 20. / 1.95

    def _fringe_phase(self, za):
        return 2 * np.pi * self.ns_baselines[:,np.newaxis] / self.wl * np.sin(np.radians(za))[np.newaxis,:]