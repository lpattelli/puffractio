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

    def save(self, fname, path='.', upscale=1, flip=0):
        c = self.flip_pixels(npix=flip)
        c = self.upscale(c, upscale)
        Path(path).mkdir(parents=True, exist_ok=True)
        img = PIL.Image.fromarray(~c).convert("1") # ~c to avoid inverte colormap
        img.save(os.path.join(path, fname), optimize=True)

    def show(self, upscale=1, flip=0):
        c = self.flip_pixels(npix=flip)
        c = self.upscale(c, upscale)
        fig, ax = plt.subplots()
        ax.imshow(c, cmap='Greys')



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
        self.mask = self.generate_mask()
        self.f, self.fexcl, self.fpix = self.area_fraction()

    def area_fraction(self):
        f = lambda r: self.Npart * np.pi * r**2 / self.Ngrid**2
        return f(self.rpart), f(self.rexcl), np.sum(self.mask)/self.mask.size

    def RSA_PBC(self, N=None, r=None, p=None, seed=None): # TODO: implement some stop mechanism after N attempts
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

    def _edge_particles(self, p=None, r=None):
        """ expects particle positions in a [0,1] square """
        if p is None:
            p = self.ppos
        if r is None:
            r = self.rexcl
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
        dm = kdt.sparse_distance_matrix(kdt, 2*self.rpart/self.Ngrid) # with respect to rpart!
        return dm.count_nonzero() > 0

    def generate_mask(self, p=None, Ngrid=None, r=None):
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

    def add_holes(self, Nadd):
        N = self.Npart + Nadd
        pnew = self.RSA_PBC(N=N, p=self.ppos)
        self.__init__(self.Ngrid, N, self.rpart, self.rexcl, ppos=pnew, seed=self.seed)

    def remove_holes(self, Nrem): # TODO: actually implement seed
        pnew = self.ppos
        N = self.Npart - Nrem
        self.__init__(self.Ngrid, N, self.rpart, self.rexcl, ppos=pnew[:N,])

    def shake(self, sigma): # TODO: would be nice to add a flag to allow overlapped particles or not, and seed?
        dp = self.rng.normal(0, sigma, size=np.shape(self.ppos))
        spos = np.mod(self.ppos + dp, self.Ngrid)
        self.__init__(self.Ngrid, self.Npart, self.rpart, self.rexcl, ppos=spos, seed=self.seed)

    def save(self, fname, path='.'):
        img = PIL.Image.fromarray(self.mask).convert("1")
        Path(path).mkdir(parents=True, exist_ok=True)
        img.save(os.path.join(path, fname), optimize=True)

    def show(self):
        fig, ax = plt.subplots()
        ax.imshow(~self.mask, cmap='Greys') # note the negation!
