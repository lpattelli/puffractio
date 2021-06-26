# the puffractio module

import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path
import os

from tqdm import tqdm
from scipy.spatial import KDTree
from skimage import draw


class Challenge:
    def __init__(self, p, q=None, f=0.5, exact=True, seed=None):
        self.p = p
        self.q = p if q is None else q
        self.f = f
        self.exact = exact
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.challenge = self.generate()

    def generate(self):
        p, q, f = self.p, self.q, self.f
        if self.exact:
            c = np.full(p*q, False)
            c[:int(p*q*f)] = True
            self.rng.shuffle(c)
        else:
            c = self.rng.choice([True, False], size=p*q, p=[f, 1-f])
        return c.reshape((p,q))

    def upscale(self, c=None, upscale=1):
        if c is None:
            c = self.challenge
        c = np.repeat(np.repeat(c, upscale, axis=0), upscale, axis=1)
        return c

    def flip_pixels(self, c=None, npix=1, seed=None):
        if c is None:
            c = self.challenge
        rng = self.rng if seed is None else np.random.default_rng(seed)
        f = c.copy()
        mask = np.full(c.size, False)
        mask[:npix] = True
        rng.shuffle(mask)
        mask = mask.reshape(c.shape)
        f[mask] = np.logical_not(f[mask])
        return f

    def invert(self):
        return np.logical_not(self.challenge)

    def shift(self, sx, sy):
        self.challenge = np.roll(self.challenge, sx, axis=1) # along "x"
        self.challenge = np.roll(self.challenge, sy, axis=0) # along "y"

    def save(self, fname, path='.', upscale=1, flip=0):
        c = self.flip_pixels(npix=flip)
        c = self.upscale(c, upscale)
        Path(path).mkdir(parents=True, exist_ok=True)
        img = PIL.Image.fromarray(~c).convert("1") # ~c to invert colormap
        img.save(os.path.join(path, fname), optimize=True)

    def show(self, upscale=1, flip=0):
        c = self.flip_pixels(npix=flip)
        c = self.upscale(c, upscale)
        fig, ax = plt.subplots()
        ax.imshow(c, cmap='Greys')


from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_sources_XY import Scalar_field_XY
from scipy.signal import correlate
from scipy.optimize import curve_fit

class Response:
    def __init__(self, challenge, puf, wavelength, pixelsize=1): # default: 1 micrometer
        ups = puf.mask.shape[0] // challenge.challenge.shape[0] # warning: assuming square masks...
        self.challenge = challenge.upscale(upscale=ups)
        self.puf = puf.mask
        self.wavelength = wavelength
        self.pixelsize = pixelsize
        self.absuz = None
        self.init_source()

    def init_source(self):
        N, M = self.challenge.shape
        x0 = np.linspace(-N//2, N//2, N) * self.pixelsize
        y0 = np.linspace(-M//2, M//2, M) * self.pixelsize
        self.u0 = Scalar_source_XY(x=x0, y=y0, wavelength=self.wavelength)
        self.u0.plane_wave()
        self.u0.u *= self.puf * self.challenge

    def propagate(self, x, y, z, verbose=True, scaleupby=None):
        N, M = self.challenge.shape
        x0, y0 = self.u0.x, self.u0.y
        # define center positions of all tiles
        if scaleupby is None:
            scaleupby = 1 # no tiling and rescaling required
        if np.floor(np.log2(scaleupby)) != np.ceil(np.log2(scaleupby)):
            raise ValueError('scaleupby must be a power of 2')
        hxspan, hyspan = (scaleupby-1)*(N//2)*self.pixelsize, (scaleupby-1)*(M//2)*self.pixelsize
        xc, yc = np.linspace(x-hxspan, x+hxspan, scaleupby), np.linspace(y-hyspan, y+hyspan, scaleupby)
        xc, yc = np.meshgrid(xc, yc)
        Ntiles = xc.size
        sh = N//scaleupby, scaleupby, M//scaleupby, scaleupby
        tiles = np.zeros((Ntiles, N//scaleupby, M//scaleupby))
        for t in range(Ntiles):
            xt = - xc[np.unravel_index(t, xc.shape)] - x0[-1]
            yt = - yc[np.unravel_index(t, yc.shape)] - y0[-1]
            temp = self.u0._RS_(z=z, n=1, new_field=False, out_matrix=True,
                                kind='z', verbose=verbose, xout=xt, yout=yt)
            tiles[t,:,:] = np.abs(temp).reshape(sh).mean(-1).mean(1) # taking the abs() value !!!
        absu = np.split(tiles,np.arange(scaleupby,Ntiles,scaleupby))
        absu = np.concatenate(np.concatenate(absu, axis=1), axis=1)
        x = x0*scaleupby + x
        y = y0*scaleupby + y
        self.absuz = Scalar_field_XY(x=x, y=y, wavelength=self.wavelength, info="abs field")
        self.absuz.u = absu
        return self.absuz

    def shrinkby(self, factor):
        nN, nM = self.absuz.u.shape[0]//factor, self.absuz.u.shape[1]//factor
        nx, ny = self.absuz.x.reshape(nN,-1).mean(1), self.absuz.y.reshape(nM,-1).mean(1)
        sh = nN, factor, nM, factor
        absus = Scalar_field_XY(x=nx, y=ny, wavelength=self.wavelength, info="abs field (shrunk)")
        absus.u = self.absuz.u.reshape(sh).mean(-1).mean(1)
        return absus

    def registerxy(self, f0, fitwidth=20, thresh=0.8, plotfit=False, guess=None):
        f1 = np.abs(self.absuz.u)**2 # reference pattern
        xcorr = correlate((f0-np.mean(f0))/np.std(f0), (f1-np.mean(f1))/np.std(f1), 'full')
        nxc = xcorr/f0.size
        x = np.arange(-nxc.shape[0]//2,nxc.shape[0]//2)+1 # for 'full' correlation size
        if guess is None:
            dx, dy = np.unravel_index(np.argmax(xcorr), xcorr.shape)
        else:
            dx, dy = np.argmin(np.abs(x-guess[1])), np.argmin(np.abs(x-guess[0]))
        xrange = np.arange(dx-fitwidth, dx+fitwidth)
        yrange = np.arange(dy-fitwidth, dy+fitwidth)
        X, Y = np.meshgrid(xrange, yrange)
        ydata = nxc[X,Y] # select window
        X, Y = np.meshgrid(x[xrange],x[yrange]) # overwrite to actual units
        p0 = [1, x[dx], x[dy], fitwidth/2, fitwidth/2, 0] # initial guesses
        popt, _ = curve_fit(self._gauss2d, (X,Y), ydata.ravel(), p0=p0)
        if plotfit:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(nxc, extent=(x.min(), x.max(), x.min(), x.max()), origin='lower')
            ax[0].axline((0, x.min()), (0, x.max()), color='w', linewidth=1)
            ax[0].axline((x.min(), 0), (x.max(), 0), color='w', linewidth=1)
            ax[0].set_title('full cross-correlation map')
            ax[1].imshow(ydata, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower')
            ax[1].contour(X, Y, self._gauss2d((X,Y), *popt).reshape(X.shape), 8, colors='w')
            ax[1].set_xlabel("dy") # rows and colums
            ax[1].set_ylabel("dx") # are swapped
            ax[1].set_title("cross-correlation peak fit")
        ps = self.pixelsize * np.round(np.mean(np.diff(self.absuz.x)) / self.pixelsize)
        regx = ps*popt[2] if np.abs(popt[2]) > thresh else 0 # a shift below 1 pixel is
        regy = ps*popt[1] if np.abs(popt[1]) > thresh else 0 # probably not significant
        return regx, regy

    def _gauss2d(self, xytuple, A, x0, y0, sigmax, sigmay, theta):
        ''' internal model to fit the cross-correlation peak '''
        (x, y) = xytuple
        a = (np.cos(theta)**2)/(2*sigmax**2) + (np.sin(theta)**2)/(2*sigmay**2)
        b = -(np.sin(2*theta))/(4*sigmax**2) + (np.sin(2*theta))/(4*sigmay**2)
        c = (np.sin(theta)**2)/(2*sigmax**2) + (np.cos(theta)**2)/(2*sigmay**2)
        g = A*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
        return g.ravel()


class PUFmask:
    def __init__(self, Ngrid, Npart, rpart, rexcl, ppos=None, seed=None):
        self.Ngrid = Ngrid
        self.Npart = Npart
        self.rpart = rpart
        self.rexcl = rexcl
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        if ppos is not None and ppos.shape[0] == Npart:
            self.ppos = ppos
        else:
            self.ppos = self.RSA_PBC(p=ppos)
        self.mask = self.generate()
        self.f, self.fexcl, self.fpix = self.area_fraction()

    def area_fraction(self):
        f = lambda r: self.Npart * np.pi * r**2 / self.Ngrid**2
        return f(self.rpart), f(self.rexcl), np.sum(self.mask)/self.mask.size

    def RSA_PBC(self, N=None, r=None, p=None, seed=None):
        if N is None:
            N = self.Npart
        if r is None:
            r = self.rexcl
        rng = self.rng if seed is None else np.random.default_rng(seed)
        r /= self.Ngrid # go to normalized units
        p = rng.random((1, 2)) if (p is None) else p/self.Ngrid
        with tqdm(initial=np.size(p, 0), total=N, desc="packing disks") as pbar:
            while np.size(p, 0) < N:
                pnew = rng.random((N, 2))
                pnew = self._remove_overlapping(pnew, r, p, boxsize=1)
                p = self._remove_overlapping(np.concatenate([p,pnew]), r, boxsize=1)
                pbar.n = min(N, np.size(p,0)); pbar.refresh()
        return self.Ngrid*p[:N,]

    def _edge_particles(self, p, r):
        """ expects particle positions in a [0,1] square """
        p = np.concatenate([p + [ 1, 0], p + [ 1, 1], p + [ 0, 1], p + [-1, 1],
                            p + [-1, 0], p + [-1,-1], p + [ 0,-1], p + [ 1,-1]])
        p = np.delete(p, np.any(p > 1+r, 1), 0)
        p = np.delete(p, np.any(p <  -r, 1), 0)
        return p

    def _remove_overlapping(self, pnew, r=None, p=None, boxsize=1):
        if r is None:
            r = self.rexcl
        kdt1 = KDTree(pnew, boxsize=boxsize)
        kdt2 = KDTree(pnew, boxsize=boxsize)
        if p is not None:
            kdt2 = KDTree(p, boxsize=boxsize)
        dm = kdt1.sparse_distance_matrix(kdt2, 2*r).nonzero()
        return np.delete(pnew, dm[0], 0)

    def has_overlap(self, boxsize=1):
        kdt = KDTree(self.ppos / self.Ngrid, boxsize=boxsize)
        dm = kdt.sparse_distance_matrix(kdt, 2*self.rpart/self.Ngrid) # rpart instead of rexcl
        return dm.count_nonzero() > 0

    def generate(self, p=None, Ngrid=None, r=None):
        if p is None:
            p = self.ppos
        if Ngrid is None:
            Ngrid = self.Ngrid
        if r is None:
            r = self.rpart
        p = np.concatenate([p, Ngrid*self._edge_particles(p/Ngrid, r/Ngrid)])
        m = np.zeros((Ngrid,Ngrid), dtype=bool)
        for pidx in tqdm(range(np.size(p,0)), desc="draw the mask"):
            m[draw.disk(p[pidx,], r, shape=m.shape)] = True
        return m

    def add_holes(self, Nadd, seed=None):
        N = self.Npart + Nadd
        pnew = self.RSA_PBC(N=N, p=self.ppos)
        self.__init__(self.Ngrid, N, self.rpart, self.rexcl, ppos=pnew, seed=self.seed)

    def remove_holes(self, Nrem, seed=None):
        N = self.Npart - Nrem
        rng = self.rng if seed is None else np.random.default_rng(seed)
        keep = rng.choice(range(self.Npart), self.Npart - Nrem, replace=False)
        pnew = self.ppos[keep,]
        self.__init__(self.Ngrid, N, self.rpart, self.rexcl, ppos=pnew)

    def shake(self, sigma):
        dp = self.rng.normal(0, sigma, size=np.shape(self.ppos))
        spos = np.mod(self.ppos + dp, self.Ngrid)
        self.__init__(self.Ngrid, self.Npart, self.rpart, self.rexcl, ppos=spos, seed=self.seed)

    def shift(self, sx, sy):
        self.mask = np.roll(self.mask, sx, axis=1) # along "x"
        self.mask = np.roll(self.mask, sy, axis=0) # along "y"

    def save(self, fname, path='.'):
        img = PIL.Image.fromarray(self.mask).convert("1")
        Path(path).mkdir(parents=True, exist_ok=True)
        img.save(os.path.join(path, fname), optimize=True)

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(~self.mask, cmap='Greys') # ~self.mask to invert colormap
